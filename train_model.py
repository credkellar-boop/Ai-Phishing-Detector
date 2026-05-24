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
