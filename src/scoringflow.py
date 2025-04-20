from metaflow import FlowSpec, step, Parameter
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ScoringFlow(FlowSpec):
    run_id = Parameter('run_id', help='MLFlow run ID of the model to use')
    mlflow_tracking_uri = Parameter('mlflow_tracking_uri', default='http://127.0.0.1:5000', help='MLflow tracking server URI')

    @step
    def start(self):
        print(f"Starting scoring flow with model from run {self.run_id}")
        self.next(self.load_data)
    
    @step
    def load_data(self):
        iris = load_iris()
        data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        data['target'] = iris.target
        
        _, self.data = train_test_split(data, test_size=0.3, random_state=42)
        print(f"Loaded {len(self.data)} samples for scoring")
        self.next(self.feature_engineering)
    
    @step
    def feature_engineering(self):        
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        
        selector = SelectKBest(f_classif, k=3)
        X_new = selector.fit_transform(X, y)
        
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        print(f"Selected features: {selected_features}")
        
        self.processed_X = pd.DataFrame(X_new, columns=selected_features)
        self.target_y = y
        
        self.next(self.load_model)
    
    @step
    def load_model(self):
        print(f"Loading model from MLFlow run {self.run_id}")
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        self.model = mlflow.sklearn.load_model(f"runs:/{self.run_id}/model")
        self.next(self.make_predictions)
    
    @step
    def make_predictions(self):
        self.predictions = self.model.predict(self.processed_X)
        self.probabilities = self.model.predict_proba(self.processed_X)
        
        print(f"Made predictions for {len(self.predictions)} samples")
        self.next(self.evaluate_model)
    
    @step
    def evaluate_model(self):
        self.accuracy = accuracy_score(self.target_y, self.predictions)
        self.precision = precision_score(self.target_y, self.predictions, average='weighted')
        self.recall = recall_score(self.target_y, self.predictions, average='weighted')
        self.f1 = f1_score(self.target_y, self.predictions, average='weighted')
        
        print(f"Scoring metrics:")
        print(f"  Accuracy:  {self.accuracy:.4f}")
        print(f"  Precision: {self.precision:.4f}")
        print(f"  Recall:    {self.recall:.4f}")
        print(f"  F1-Score:  {self.f1:.4f}")
        
        self.next(self.end)
    
    @step
    def end(self):
        print("Scoring flow complete!")

if __name__ == "__main__":
    ScoringFlow()