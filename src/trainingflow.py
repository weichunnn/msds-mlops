from metaflow import FlowSpec, step, Parameter
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

import mlflow

class TrainingFlow(FlowSpec):
    seed = Parameter('seed', default=42, help='Random seed')
    cv_folds = Parameter('cv_folds', default=5, help='Number of CV folds')

    n_estimators = Parameter('n_estimators', default=200, help='Number of trees in the forest')
    max_depth = Parameter('max_depth', default=3, help='Maximum depth of the trees')
    learning_rate = Parameter('learning_rate', default=0.01, help='Learning rate')

    mlflow_tracking_uri = Parameter('mlflow_tracking_uri', default='http://127.0.0.1:5000', help='MLflow tracking server URI')
    experiment_name = Parameter('experiment_name', default='iris-experiment', help='MLflow experiment name')
    
    @step
    def start(self):
        print("Starting the training flow")
        self.next(self.ingest_data)
    
    @step
    def ingest_data(self):
        iris = load_iris()
        data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        data['target'] = iris.target
        
        # split for training and testing
        # this represents a hold out set where the other 30% will be used in the scoringflow
        self.data, _ = train_test_split(data, test_size=0.3, random_state=self.seed)
        print(f"Loaded {len(self.data)} training samples from iris dataset")
        self.next(self.feature_engineering)
    
    @step
    def feature_engineering(self):
        
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        
        # following from best experiment in previous notebooks/labs
        selector = SelectKBest(f_classif, k=3)
        X_new = selector.fit_transform(X, y)
        
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        print(f"Selected features: {selected_features}")
        
        self.data = pd.DataFrame(X_new, columns=selected_features)
        self.data['target'] = y.values
            
        self.next(self.train_model)
    
    @step
    def train_model(self):
        print(f"Training model with parameters:")
        print(f"  CV folds: {self.cv_folds}")
        print(f"  Seed: {self.seed}")
        print(f"  n_estimators: {self.n_estimators}")
        print(f"  max_depth: {self.max_depth}")
        
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.seed
        )
        scores = cross_val_score(model, X, y, cv=self.cv_folds)
        
        # train final model
        model.fit(X, y)
        self.model = model
        self.accuracy = np.mean(scores)
        
        # calculate additional metrics
        y_pred = model.predict(X)
        self.precision = precision_score(y, y_pred, average='weighted')
        self.recall = recall_score(y, y_pred, average='weighted')
        self.f1 = f1_score(y, y_pred, average='weighted')
        
        print(f"Model trained with CV accuracy: {self.accuracy:.4f}")
        self.next(self.register_model)
    
    @step
    def register_model(self):    
        # set up MLflow tracking with the URI passed in
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        
        with mlflow.start_run() as run:
            # logging parameters on mlflow
            mlflow.log_param("seed", self.seed)
            mlflow.log_param("cv_folds", self.cv_folds)
            mlflow.log_param("n_estimators", self.n_estimators)
            mlflow.log_param("max_depth", self.max_depth)
            mlflow.log_param("learning_rate", self.learning_rate)
            
            # log metrics
            mlflow.log_metric("accuracy", self.accuracy)
            mlflow.log_metric("precision", self.precision)
            mlflow.log_metric("recall", self.recall)
            mlflow.log_metric("f1", self.f1)

            # log and register model
            mlflow.sklearn.log_model(self.model, "model")
            self.run_id = run.info.run_id
            mlflow.register_model(f"runs:/{run.info.run_id}/model", "IrisClassifier")
        
        print(f"Model registered with MLFlow, run_id: {self.run_id}")
        self.next(self.end)
    
    @step
    def end(self):
        print("Training flow complete!")
        print(f"Model accuracy: {self.accuracy:.4f}")
        print(f"MLFlow run ID: {self.run_id}")

if __name__ == "__main__":
    TrainingFlow()