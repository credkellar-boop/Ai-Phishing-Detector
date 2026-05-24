import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC

# Define the missing function to generate dummy data
def create_dummy_data():
    # Returns a simple pandas DataFrame with 'text' and 'label' columns
    data = {
        'text': ['example phishing email', 'legitimate message', 'win a free prize', 'meeting at 10'],
        'label': [1, 0, 1, 0]
    }
    return pd.DataFrame(data)

def train_and_save_model():
    print("Loading data...")
    df = create_dummy_data()

    # 1. Prepare Data
    X = df['text']
    y = df['label']

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)

    # 2. Split Data: These variables are now defined in this local scope
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    # 3. Define and Train Ensemble

    # We define all models and the ensemble within the same function
    clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf2 = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf3 = SVC(probability=True, random_state=42)

    ensemble_clf = VotingClassifier(
        estimators=[('rf', clf1), ('gb', clf2), ('svc', clf3)],
        voting='soft'
    )

    print("Training Ensemble...")
    # Now X_train and y_train are visible here, so flake8 will be happy!
    ensemble_clf.fit(X_train, y_train)

    # 4. Save
    os.makedirs('model', exist_ok=True)
    joblib.dump(ensemble_clf, 'model/phishing_model.pkl')
    joblib.dump(vectorizer, 'model/vectorizer.pkl')

if __name__ == "__main__":
    train_and_save_model()
