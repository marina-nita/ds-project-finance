X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#StandardScaler removes the mean and scales each feature to unit variance
scaler = StandardScaler()
