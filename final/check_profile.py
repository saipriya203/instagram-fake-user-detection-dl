import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the trained model and preprocessing objects
model_data = joblib.load('saved_model/cnn_lstm_model.pkl')
model = model_data['model']
lang_dict = model_data['lang_dict']
scaler = model_data.get('scaler')

def predict_gender(name):
    """Simple gender prediction from name"""
    if not name or not isinstance(name, str):
        return 0
    first_name = name.split(' ')[0].lower()
    # Simple heuristic based on name endings
    female_endings = ['a', 'e', 'i', 'y']
    male_endings = ['o', 'r', 's', 't', 'n']
    
    if any(first_name.endswith(ending) for ending in female_endings):
        return -1  # female
    elif any(first_name.endswith(ending) for ending in male_endings):
        return 1   # male
    return 0       # unknown

def check_profile():
    print("\n=== Fake Profile Detector ===")
    print("Please enter the following profile details (press Enter to skip any field):\n")
    
    # Get user input with default values
    username = input("Username: ")
    name = input("Full name: ")
    
    try:
        statuses = int(input("Number of posts/statuses (e.g., 100): ") or "0")
        followers = int(input("Number of followers (e.g., 500): ") or "0")
        following = int(input("Number of people following (e.g., 200): ") or "0")
        has_url = input("Has link/URL in bio? (y/n): ").lower() == 'y'
        has_location = input("Has location in bio? (y/n): ").lower() == 'y'
        has_default_pic = input("Is using a default/blank profile picture? (y/n): ").lower() == 'y'
    except ValueError:
        print("\nError: Please enter valid numbers for the counts.")
        return
    
    # Prepare features map
    feature_dict = {
        'statuses_count': statuses,
        'followers_count': followers,
        'friends_count': following,
        'has_url': 1 if has_url else 0,
        'has_location': 1 if has_location else 0,
        'default_profile_image': 1 if has_default_pic else 0,
        'default_profile': 1 if has_default_pic else 0,
    }
    
    # Add textual features extracted from names
    if username:
        feature_dict['screen_name_length'] = len(username)
        feature_dict['screen_name_digits'] = sum(c.isdigit() for c in username)
    if name:
        feature_dict['name_length'] = len(name)
        feature_dict['name_digits'] = sum(c.isdigit() for c in name)
        
    df = pd.DataFrame([feature_dict])
    
    # Fill any missing required feature columns with 0
    for col in model_data['feature_columns']:
        if col not in df.columns:
            df[col] = 0
            
    # Order columns as in training
    df = df[model_data['feature_columns']]
    
    if scaler is not None:
        features_scaled = scaler.transform(df)
    else:
        features_scaled = df
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    # Display results
    print("\n=== Profile Analysis ===")
    print(f"Name: {name}")
    print(f"Statuses: {statuses}, Followers: {followers}, Following: {following}")
    
    result = "FAKE" if prediction == 0 else "GENUINE"
    confidence = max(probability) * 100
    
    print(f"\nPrediction: {result}")
    print(f"Confidence: {confidence:.1f}%")
    
    if result == "FAKE":
        print("\n[!] Warning: This profile shows signs of being fake. Be cautious!")
        print("Common signs of fake profiles:")
        print("- Unusually high followers with few posts")
        print("- Very few followers compared to following")
        print("- Inconsistent activity patterns")
    else:
        print("\n[+] This profile appears to be genuine.")
        print("Common signs of genuine profiles:")
        print("- Balanced follower/following ratio")
        print("- Consistent posting activity")
        print("- Normal engagement metrics")

if __name__ == "__main__":
    try:
        while True:
            check_profile()
            if input("\nCheck another profile? (y/n): ").lower() != 'y':
                print("\nThank you for using Fake Profile Detector!")
                break
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user. Goodbye!")
