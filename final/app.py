from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, send_from_directory
import pandas as pd
import joblib
import os
import sys
import logging
from pathlib import Path
from whitelist import is_whitelisted
from instagram_analyzer import InstagramAnalyzer
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Constants
MODEL_PATH = Path('saved_model/cnn_lstm_model.pkl')
REQUIRED_FEATURES = [
    'followers_count', 'following_count', 'post_count',
    'has_profile_pic', 'is_private', 'is_verified'
]

# Initialize model and related data
model = None
scaler = None
feature_columns = []
lang_dict = {}

# Initialize global analyzer
# We do this globally so we don't spam Instagram servers on every API request
# and trigger 429 Too Many Requests blocks
try:
    analyzer = InstagramAnalyzer()
except Exception as e:
    logger.error(f"Failed to initialize InstagramAnalyzer: {e}")
    analyzer = None

def load_model():
    """Load the trained model and related data"""
    global model, feature_columns, lang_dict, scaler
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        model_data = joblib.load(MODEL_PATH)
        model = model_data.get('model')
        scaler = model_data.get('scaler', None)
        feature_columns = model_data.get('feature_columns', [])
        lang_dict = model_data.get('lang_dict', {})
        if model is None:
            raise ValueError("Model not found in the loaded data")
            
        # Patch for older models loaded in newer scikit-learn versions
        # Adds missing 'monotonic_cst' attribute to DecisionTreeClassifiers
        try:
            from sklearn.tree import DecisionTreeClassifier
            if hasattr(model, 'estimators_'):
                for estimator in model.estimators_:
                    if isinstance(estimator, DecisionTreeClassifier) and not hasattr(estimator, 'monotonic_cst'):
                        setattr(estimator, 'monotonic_cst', None)
            elif isinstance(model, DecisionTreeClassifier) and not hasattr(model, 'monotonic_cst'):
                setattr(model, 'monotonic_cst', None)
        except ImportError:
            pass
            
        logger.info("Successfully loaded model and related data")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def predict_gender(name):
    """Predict gender from name using simple heuristics
    Args:
        name (str): The full name to analyze
    Returns:
        int: -1 for female, 1 for male, 0 for unknown
    """
    if not name or not isinstance(name, str):
        return 0
    first_name = name.split(' ')[0].lower()
    female_endings = ['a', 'e', 'i', 'y']
    male_endings = ['o', 'r', 's', 't', 'n']
    if any(first_name.endswith(ending) for ending in female_endings):
        return -1  # female
    elif any(first_name.endswith(ending) for ending in male_endings):
        return 1   # male
    return 0       # unknown

@app.route('/')
def home():
    """Redirect to the Instagram analyzer page"""
    return redirect(url_for('instagram_analyzer'))

# Whitelist is now managed in whitelist.py
def validate_input_data(data):
    """Validate the input data for prediction
    Args:
        data (dict): Dictionary containing form data
    Returns:
        tuple: (is_valid, error_message, validated_data)
    """
    validated = {}
    # Required fields check
    for field in REQUIRED_FEATURES:
        if field not in data:
            return False, f"Missing required field: {field}", None
    # Type conversion and validation
    try:
        # Numeric fields
        for field in ['followers_count', 'following_count', 'post_count']:
            validated[field] = int(data[field])
            if validated[field] < 0:
                return False, f"{field} cannot be negative", None
        # Boolean fields
        for field in ['has_profile_pic', 'is_private', 'is_verified']:
            val = str(data[field]).lower()
            if val in ['true', '1', 'yes', 'y']:
                validated[field] = True
            elif val in ['false', '0', 'no', 'n']:
                validated[field] = False
            else:
                return False, f"Invalid boolean value for {field}: {data[field]}", None
        return True, "", validated
    except ValueError as e:
        return False, f"Invalid input format: {str(e)}", None

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None:
            logger.error("Model not loaded")
            return jsonify({
                'status': 'error',
                'message': 'Prediction model is not available. Please try again later.'
            }), 503  # Service Unavailable
        
        data = request.form.to_dict()
        user_id = data.get('user_id', '').strip()
        if not user_id:
            return jsonify({
                'status': 'error',
                'message': 'User ID is required'
            }), 400
            
        # Check whitelist
        if is_whitelisted(user_id):
            logger.info(f"User {user_id} is in whitelist")
            return render_template('result.html', 
                               result={
                                   'prediction': 'GENUINE',
                                   'confidence': 99.9,
                                   'is_whitelisted': True,
                                   'features': {
                                       'statuses_count': 0,
                                       'followers_count': 0,
                                       'friends_count': 0
                                   }
                               },
                               request=request)
        
        # Extract and validate features
        features = {
            'followers_count': int(data.get('followers_count', 0)),
            'following_count': int(data.get('following_count', 0)),
            'post_count': int(data.get('post_count', 0)),
            'has_profile_pic': 1 if data.get('has_profile_pic', 'true').lower() == 'true' else 0,
            'is_private': 1 if data.get('is_private', 'false').lower() == 'true' else 0,
            'is_verified': 1 if data.get('is_verified', 'false').lower() == 'true' else 0,
            'lang_code': lang_dict.get(data.get('language', 'en').lower(), 0)
        }
        
        # Heuristic checks for suspicious profiles
        is_suspicious = False
        
        # Check for suspicious following/follower ratio (only for non-verified accounts)
        # Fake profiles often follow a massive amount of people but have very low followers
        if not features['is_verified'] and features['followers_count'] > 0 and features['following_count'] > 1000 and (features['following_count'] / features['followers_count']) > 10:
            is_suspicious = True
            logger.info(f"Suspicious following/follower ratio for user {user_id}")
            
        # Check for very low post count relative to followers (only for non-verified accounts)
        if not features['is_verified'] and features['followers_count'] > 1000 and features['post_count'] < 10:
            is_suspicious = True
            logger.info(f"Suspiciously low post count for user {user_id}")
            
        # Check for private account with high follower count (only for non-verified accounts)
        if not features['is_verified'] and features['is_private'] and features['followers_count'] > 5000:
            is_suspicious = True
            logger.info(f"Suspicious private account with high follower count for user {user_id}")
            
        # Influencer check - verified accounts with high follower count
        is_influencer = features['is_verified'] and features['followers_count'] > 10000
        
        # If influencer, mark as genuine regardless of other checks
        if is_influencer:
            logger.info(f"Influencer profile detected for user {user_id}")
            return render_template('result.html',
                               result={
                                   'prediction': 'GENUINE',
                                   'confidence': 99.0,
                                   'is_whitelisted': False,
                                   'is_influencer': True,
                                   'features': {
                                       'statuses_count': features['post_count'],
                                       'followers_count': features['followers_count'],
                                       'friends_count': features['following_count']
                                   }
                               },
                               request=request)
            
        # If any suspicious flags are raised and not an influencer, mark as fake
        if is_suspicious:
            return render_template('result.html',
                               result={
                                   'prediction': 'FAKE',
                                   'confidence': 95.0,
                                   'is_whitelisted': False,
                                   'features': {
                                       'statuses_count': features['post_count'],
                                       'followers_count': features['followers_count'],
                                       'friends_count': features['following_count']
                                   }
                               },
                               request=request)
        is_valid, error_msg, validated_data = validate_input_data(features)
        if not is_valid or validated_data is None:
            logger.warning(f"Invalid input data: {error_msg}")
            return jsonify({
                'status': 'error',
                'message': error_msg
            }), 400
            
        # Fix mapping bug: frontend uses following_count/post_count, model needs friends_count/statuses_count
        if 'following_count' in validated_data:
            validated_data['friends_count'] = validated_data['following_count']
        if 'post_count' in validated_data:
            validated_data['statuses_count'] = validated_data['post_count']
            
        # Map frontend data to the new model structured features
        if 'has_profile_pic' in validated_data:
            validated_data['default_profile_image'] = 0 if validated_data['has_profile_pic'] else 1
            validated_data['default_profile'] = 0 if validated_data['has_profile_pic'] else 1
            
        if 'is_private' in validated_data:
            validated_data['protected'] = 1 if validated_data['is_private'] else 0
            
        if 'is_verified' in validated_data:
            validated_data['verified'] = 1 if validated_data['is_verified'] else 0
            
        # Try computing missing textual features from user_id if present
        if user_id:
            validated_data['screen_name_length'] = len(user_id)
            validated_data['screen_name_digits'] = sum(c.isdigit() for c in user_id)
            validated_data['name_length'] = len(user_id)  # Assuming name length is similar to username if absent
            validated_data['name_digits'] = sum(c.isdigit() for c in user_id)
            
        df = pd.DataFrame([validated_data])
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0  # Fill missing columns with default value
        df = df[feature_columns]
        
        # Scale features if a scaler is available
        if scaler is not None:
            df_scaled = scaler.transform(df)
        else:
            df_scaled = df
            
        prediction = model.predict(df_scaled)[0]
        proba = model.predict_proba(df_scaled)[0]
        
        # Normalize probabilities to ensure valid confidence 0-100%
        sum_proba = sum(proba)
        # Avoid division by zero
        normalized_prob = proba[prediction] / sum_proba if sum_proba > 0 else 1.0
        confidence = normalized_prob * 100
        
        label = 'GENUINE' if prediction == 0 else 'FAKE'
        logger.info(f"Prediction for user {user_id}: {label} (confidence: {confidence:.2f}%)")
        return render_template('result.html', 
                               result={
                                   'prediction': label,
                                   'confidence': round(confidence, 2),
                                   'is_whitelisted': False,
                                   'features': {
                                       'statuses_count': validated_data.get('statuses_count', 0),
                                       'followers_count': validated_data.get('followers_count', 0),
                                       'friends_count': validated_data.get('friends_count', 0)
                                   }
                               },
                               request=request)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'An error occurred during prediction',
            'details': str(e)
        }), 500

def init_app():
    """Initialize the Flask application"""
    os.makedirs('templates', exist_ok=True)
    os.makedirs('saved_model', exist_ok=True)
    if not load_model():
        logger.warning("Failed to load model on startup. Will try lazy loading.")

@app.route('/proxy-image/<path:image_url>')
def proxy_image(image_url):
    """Proxy Instagram images to avoid CORS issues."""
    import requests
    from flask import Response
    
    try:
        # Decode the URL
        from urllib.parse import unquote
        image_url = unquote(image_url)
        
        # Fetch the image
        response = requests.get(image_url, timeout=10)
        
        if response.status_code == 200:
            # Return the image with appropriate headers
            return Response(response.content, 
                          mimetype=response.headers.get('Content-Type', 'image/jpeg'),
                          headers={
                              'Cache-Control': 'public, max-age=3600',
                              'Access-Control-Allow-Origin': '*'
                          })
        else:
            # Return placeholder if image fetch fails
            placeholder_url = 'https://via.placeholder.com/150'
            placeholder_response = requests.get(placeholder_url)
            return Response(placeholder_response.content, 
                          mimetype='image/jpeg')
    except Exception as e:
        logger.error(f"Error proxying image: {str(e)}")
        # Return placeholder on error
        placeholder_url = 'https://via.placeholder.com/150'
        placeholder_response = requests.get(placeholder_url)
        return Response(placeholder_response.content, 
                      mimetype='image/jpeg')

@app.route('/instagram')
def instagram_analyzer():
    """Render the Instagram analyzer page."""
    return render_template('instagram.html')

@app.route('/api/analyze/instagram', methods=['POST'])
def analyze_instagram():
    """API endpoint to analyze an Instagram profile."""
    try:
        data = request.get_json()
        username = data.get('username', '').strip('@ ')
        
        if not username:
            return jsonify({'error': 'Username is required'}), 400
            
        # Get the profile data
        profile = None
        if analyzer:
            profile = analyzer.get_profile(username)
        
        if not profile:
            return jsonify({'error': 'Failed to fetch profile from Instagram. The account may be private, misspelled, or Instagram is temporarily blocking our requests (Rate Limit). Please try again later.'}), 404
            
        # Analyze the profile
        if analyzer:
            analyzer.analyze_profile(username)
        
        # Calculate confidence based on profile characteristics
        is_fake = profile.get('is_fake', False)
        confidence = 0
        
        # Base confidence
        if not is_fake:
            confidence = 85  # Base confidence for real profiles
            # Increase confidence for verified accounts
            if profile.get('is_verified', False):
                confidence = 95
            # Increase confidence for accounts with substantial followers
            elif profile.get('followers', 0) > 10000:
                confidence = 90
            # Increase confidence for accounts with posts
            elif profile.get('posts', 0) > 50:
                confidence = 88
        else:
            confidence = 75  # Base confidence for fake profiles
            # Increase confidence if obvious fake signs
            if profile.get('followers', 0) == 0:
                confidence = 85
            elif profile.get('posts', 0) == 0:
                confidence = 80
        
        # Prepare the response
        response = {
            'success': True,
            'profile': {
                'username': profile.get('username', ''),
                'full_name': profile.get('full_name', ''),
                'biography': profile.get('biography', ''),
                'external_url': profile.get('external_url', ''),
                'followers': profile.get('followers', 0),
                'following': profile.get('followees', 0),
                'posts': profile.get('posts', 0),
                'is_private': profile.get('is_private', False),
                'is_verified': profile.get('is_verified', False),
                'profile_pic_url': profile.get('profile_pic_url', '')
            },
            'analysis': {
                'is_fake': is_fake,
                'confidence': confidence / 100,  # Convert to decimal
                'reasons': profile.get('reasons', [])
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error analyzing Instagram profile: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_app()
    try:
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}")
        raise
