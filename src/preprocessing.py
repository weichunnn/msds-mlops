import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def load_data():
    data = pd.read_csv('data/iris.csv')
    data['target'] = data.target
    return data

def preprocess_data(data):
    X = data.drop('target', axis=1)
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    preprocessor = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )
    
    preprocessor.fit(X_train)
    
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    train_processed = pd.DataFrame(X_train_processed, columns=X.columns)
    test_processed = pd.DataFrame(X_test_processed, columns=X.columns)
    
    train_processed['target'] = y_train.values
    test_processed['target'] = y_test.values
    
    return preprocessor, train_processed, test_processed

if __name__ == "__main__":
    data = load_data()
    
    pipeline, train_data, test_data = preprocess_data(data)
    
    train_data.to_csv('data/iris_processed_train_data.csv', index=False)
    test_data.to_csv('data/iris_processed_test_data.csv', index=False)
    
    with open('data/iris_pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)