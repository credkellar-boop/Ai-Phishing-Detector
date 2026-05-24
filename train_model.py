import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def create_dummy_data():
    """Provides sample data. Replace with pd.read_csv('your_dataset.csv') for real use."""
    data = {
        'text': [
            "Urgent: Your bank account has been compromised. Click here to reset your password.",
            "Hey team, just a reminder about the meeting tomorrow at 10 AM.",
            "You have won a $1000 Walmart gift card! Claim it now at this link.",
            "Please find attached the invoice for your recent purchase.",
            "Warning: Unauthorized login attempt detected. Verify your identity immediately.",
            "Are we still on for lunch later?"
        ],
        'label': [1, 0, 1, 0, 1, 0] # 1 = Phishing, 0 = Legitimate
    }
    return pd.DataFrame(data)

def train_and_save_model():
    print("Loading data...")
    df = create_dummy_data() 
    
    X = df['text']
    y = df['label']
    
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    print("Saving model and vectorizer...")
    os.makedirs('model', exist_ok=True)
    joblib.dump(clf, 'model/phishing_model.pkl')
    joblib.dump(vectorizer, 'model/vectorizer.pkl')
    print("Done! You can now run the Streamlit app.")

if __name__ == "__main__":
    train_and_save_model()
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC

# Define the base models
clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
clf2 = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf3 = SVC(probability=True, random_state=42)

# Create the Ensemble (The "Voting" mechanism)
ensemble_clf = VotingClassifier(
    estimators=[('rf', clf1), ('gb', clf2), ('svc', clf3)],
    voting='soft' # Uses probability to make a more informed decision
)

# Train the ensemble
ensemble_clf.fit(X_train, y_train)

# Save this ensemble model instead
joblib.dump(ensemble_clf, 'model/phishing_model.pkl')
