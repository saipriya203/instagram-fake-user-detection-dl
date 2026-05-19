
# coding: utf-8

### Detect fake profiles in online social networks using Random Forest

# In[54]:

import sys
import csv
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import gender_guesser.detector as gender
from sklearn.impute import SimpleImputer as Imputer
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from  sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# get_ipython().magic(u'matplotlib inline')  # Commented out as it's specific to Jupyter


####### function for reading dataset from csv files

# In[55]:


def read_datasets():
    """ Reads users profile from csv files """
    genuine_users = pd.read_csv("data/users.csv")
    fake_users = pd.read_csv("data/fusers.csv")
    # print genuine_users.columns
    # print genuine_users.describe()
    #print fake_users.describe()
    x=pd.concat([genuine_users,fake_users])   
    y=len(fake_users)*[0] + len(genuine_users)*[1]
    return x,y
    


####### function for predicting sex using name of person

# In[56]:

def predict_sex(name):
    sex_predictor = gender.Detector()
    first_name = name.str.split(' ').str.get(0)
    
    def get_gender(fname):
        if not isinstance(fname, str):
            return 0  # unknown
        result = sex_predictor.get_gender(fname)
        if result in ['female', 'mostly_female']:
            return -1  # female
        elif result in ['male', 'mostly_male']:
            return 1   # male
        else:
            return 0   # unknown
    
    return first_name.apply(get_gender)


####### function for feature engineering

# In[57]:

# Global variable to store language mapping
global_lang_dict = {}

def extract_features(x):
    global global_lang_dict
    
    # Feature Engineering based on EDA
    if 'name' in x.columns:
        x['name_length'] = x['name'].apply(lambda n: len(str(n)) if pd.notnull(n) else 0)
        x['name_digits'] = x['name'].apply(lambda n: sum(c.isdigit() for c in str(n)) if pd.notnull(n) else 0)
    
    if 'screen_name' in x.columns:
        x['screen_name_length'] = x['screen_name'].apply(lambda n: len(str(n)) if pd.notnull(n) else 0)
        x['screen_name_digits'] = x['screen_name'].apply(lambda n: sum(c.isdigit() for c in str(n)) if pd.notnull(n) else 0)
        
    if 'description' in x.columns:
        x['description_length'] = x['description'].apply(lambda d: len(str(d)) if pd.notnull(d) else 0)
        
    if 'location' in x.columns:
        x['has_location'] = x['location'].notnull().astype(int)
        
    if 'url' in x.columns:
        x['has_url'] = x['url'].notnull().astype(int)

    # Convert boolean/categorical fields to float numeric so they aren't dropped
    bool_cols = ['default_profile', 'default_profile_image', 'geo_enabled', 'profile_use_background_image', 'profile_background_tile', 'protected', 'verified']
    for col in bool_cols:
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors='coerce').fillna(0)
    
    # Drop noisy, text-heavy and irrelevant columns that prevent ANN/CNN from learning properly
    noisy_columns = [
        'id', 'name', 'screen_name', 'created_at', 'location', 
        'url', 'profile_image_url', 'time_zone',
        'profile_background_image_url_https', 'profile_text_color',
        'profile_image_url_https', 'profile_sidebar_border_color',
        'profile_sidebar_fill_color', 'profile_background_image_url',
        'profile_background_color', 'profile_link_color',
        'description', 'updated', 'dataset', 'lang', 'profile_banner_url'
    ]
    
    x = x.drop(columns=[col for col in noisy_columns if col in x.columns])
    
    # Select only numeric features suitable for CNN/ANN
    x = x.select_dtypes(include=[np.number])
    
    # Fill missing values with 0
    x = x.fillna(0)
    
    # We no longer need the language mapping since we dropped lang
    global_lang_dict = {}
    
    return x, global_lang_dict


####### function for ploting learning curve

# In[60]:

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


####### function for plotting confusion matrix

# In[61]:

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    target_names=['Fake','Genuine']
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


####### function for plotting ROC curve

# In[62]:

def plot_roc_curve(y_test, y_pred):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

    print("False Positive rate: ", false_positive_rate)
    print("True Positive rate: ", true_positive_rate)


    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
    label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


####### Function for training data using Random Forest

# In[63]:

def train(X_train, y_train, X_test):
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Use class_weight='balanced' to handle any class imbalance
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        oob_score=True
    )
    
    clf.fit(X_train_scaled, y_train)
    print("The best classifier is: ", clf)
    
    # Estimate score
    scores = model_selection.cross_val_score(clf, X_train_scaled, y_train, cv=5)
    print("Cross-validation scores:", scores)
    print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))
    
    # Print feature importances
    print("\nFeature importances:")
    for name, importance in zip(X_train.columns, clf.feature_importances_):
        print(f"{name}: {importance:.4f}")
    
    # Predict
    y_pred = clf.predict(X_test_scaled)
    
    # Save the scaler along with the model
    return y_test, y_pred, clf, scaler


# In[64]:

print("reading datasets.....\n")
x,y=read_datasets()
x.describe()


# In[65]:

print("extracting featues.....\n")
x, lang_dict = extract_features(x)
print(x.columns)
print(x.describe())


# In[66]:

print("spliting datasets in train and test dataset...\n")
X_train,X_test,y_train,y_test = train_test_split(x, y, test_size=0.20, random_state=44)


# In[67]:

print("training datasets.......\n")
y_test, y_pred, clf, scaler = train(X_train, y_train, X_test)


# In[68]:

print('Classification Accuracy on Test dataset: ' ,accuracy_score(y_test, y_pred))


# In[70]:

cm=confusion_matrix(y_test, y_pred)
print('Confusion matrix, without normalization')
print(cm)
plot_confusion_matrix(cm)


# In[71]:

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')


# In[72]:

print(classification_report(y_test, y_pred, target_names=['Fake','Genuine']))


# In[73]:

plot_roc_curve(y_test, y_pred)

# Save the model and preprocessing data
import joblib
import os

# Create directory if it doesn't exist
os.makedirs('saved_model', exist_ok=True)

# Ensure all required components exist
if not hasattr(clf, 'feature_importances_'):
    raise ValueError("Model is not properly trained!")

if scaler is None:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit on training data

# Save the model, scaler, and preprocessing data
model_data = {
    'model': clf,
    'scaler': scaler,
    'feature_columns': list(X_train.columns),
    'lang_dict': global_lang_dict,
    'model_info': 'Random Forest Classifier with StandardScaler'
}

print("\nModel components to be saved:")
print(f"- Model type: {type(clf).__name__}")
print(f"- Scaler type: {type(scaler).__name__ if scaler else 'None'}")
print(f"- Feature columns: {model_data['feature_columns']}")

joblib.dump(model_data, 'saved_model/cnn_lstm_model.pkl')
print("\nModel saved to 'saved_model/cnn_lstm_model.pkl'")
print("You can now use 'check_profile.py' to check individual profiles.")

