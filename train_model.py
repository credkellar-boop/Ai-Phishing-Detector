def train_and_save_model():
    print("Loading data...")
    df = create_dummy_data() 
    X = df['text']
    y = df['label']
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)
    
    # These must be defined here so the ensemble can see them
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
    
    # Define models
    clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf2 = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf3 = SVC(probability=True, random_state=42)
    
    ensemble_clf = VotingClassifier(
        estimators=[('rf', clf1), ('gb', clf2), ('svc', clf3)],
        voting='soft'
    )
    
    # Now this will work because X_train and y_train are defined
    ensemble_clf.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(ensemble_clf, 'model/phishing_model.pkl')
    joblib.dump(vectorizer, 'model/vectorizer.pkl')
