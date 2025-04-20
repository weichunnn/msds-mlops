from metaflow import FlowSpec, step, Parameter, resources, kubernetes, conda_base, retry, timeout, catch
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np

import mlflow

@conda_base(libraries={'python': '3.12.9', 'scikit-learn': '1.5.1', 'pandas': '2.2.2', 'numpy': '1.26.4', 'mlflow': '2.15.1'})
class TrainingFlow(FlowSpec):
    # pipeline paramters
    seed = Parameter('seed', default=42, help='Random seed')
    cv_folds = Parameter('cv_folds', default=5, help='Number of CV folds')
    
    # prameters from best experiment
    n_estimators = Parameter('n_estimators', default=200, help='Number of trees in the forest')
    max_depth = Parameter('max_depth', default=3, help='Maximum depth of the trees')
    learning_rate = Parameter('learning_rate', default=0.01, help='Learning rate')
    
    @kubernetes(memory=1024, cpu=0.5, disk=1000)
    @resources(cpu=1, memory=500)
    @step
    def start(self):
        print("Starting the training flow")
        self.next(self.ingest_data)
    
    @kubernetes(memory=1024, cpu=0.5, disk=1000)
    @resources(cpu=1, memory=500)
    @timeout(minutes=10)
    @retry(times=3)
    @catch(var='ingest_error')
    @step
    def ingest_data(self):
        iris = load_iris()
        data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        data['target'] = iris.target
        
        # splitting data for train/test split
        self.data, _ = train_test_split(data, test_size=0.3, random_state=self.seed)
        print(f"Loaded {len(self.data)} training samples from iris dataset")
        
        if hasattr(self, 'ingest_error') and self.ingest_error:
            print(f"Error handled: {self.ingest_error}")
            
        self.next(self.feature_engineering)
    
    @kubernetes(memory=1024, cpu=0.5, disk=1000)
    @resources(cpu=1, memory=500)
    @timeout(minutes=15)
    @retry(times=2)
    @catch(var='feature_error')
    @step
    def feature_engineering(self):
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        
        # apply SelectKBest with k=3 (as per previous notebook)
        selector = SelectKBest(f_classif, k=3)
        X_new = selector.fit_transform(X, y)
        
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        print(f"Selected features: {selected_features}")
        
        self.data = pd.DataFrame(X_new, columns=selected_features)
        self.data['target'] = y.values
        
        if hasattr(self, 'feature_error') and self.feature_error:
            print(f"Error handled: {self.feature_error}")
            
        self.next(self.train_model)
    
    @kubernetes(memory=1024, cpu=0.5, disk=1000)
    @resources(cpu=1, memory=500)
    @timeout(minutes=30)
    @retry(times=2)
    @catch(var='train_error')
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
        
        print(f"Model trained with CV accuracy: {self.accuracy:.4f}")
        
        if hasattr(self, 'train_error') and self.train_error:
            print(f"Error handled: {self.train_error}")
            
        self.next(self.register_model)
    
    @kubernetes(memory=1024, cpu=0.5, disk=1000)
    @resources(cpu=1, memory=500)
    @timeout(minutes=15)
    @retry(times=3)
    @catch(var='register_error')
    @step
    def register_model(self):
        # set MLflow tracking URI and experiment
        mlflow.set_tracking_uri('https://mlflow-service-255971508831.us-west2.run.app')
        mlflow.set_experiment('iris-experiment-gcp')
        
        mlflow.start_run()
        
        # log parameters
        mlflow.log_param("seed", self.seed)
        mlflow.log_param("cv_folds", self.cv_folds)
        mlflow.log_param("n_estimators", self.n_estimators)
        mlflow.log_param("max_depth", self.max_depth)
        mlflow.log_param("learning_rate", self.learning_rate)
        
        mlflow.log_param("feature_selection", "SelectKBest_k3")
        mlflow.log_metric("accuracy", self.accuracy)
        
        # Save model to pickle file
        import os
        import pickle
        
        # Create models directory if it doesn't exist
        os.makedirs('../models', exist_ok=True)
        
        # Save model as pickle file
        model_path = '../models/model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
            
        # Log the saved model as an artifact
        mlflow.log_artifact(model_path, artifact_path='my_models')
        
        # Log the model
        model_info = mlflow.sklearn.log_model(self.model, "model")
        self.run_id = mlflow.active_run().info.run_id

        # Register the model in the MLflow Model Registry
        model_name = "iris-classifier"
        registered_model = mlflow.register_model(
            model_info.model_uri,
            model_name
        )
        
        # Set the model to staging
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=registered_model.version,
            stage="Staging"
        )
        
        self.model_name = model_name
        self.model_version = registered_model.version

        print(f"Model registered with MLFlow, run_id: {self.run_id}")
        print(f"Model registered in registry as '{model_name}' version {registered_model.version} in 'Staging' stage")
        print(f"Model saved to {model_path} and logged as artifact in 'my_models'")
        
        if hasattr(self, 'register_error') and self.register_error:
            print(f"Error handled: {self.register_error}")
            
        mlflow.end_run()
        self.next(self.end)
    
    @kubernetes(memory=1024, cpu=0.5, disk=1000)
    @resources(cpu=1, memory=100)
    @step
    def end(self):
        print("Training flow complete!")
        print(f"Model accuracy: {self.accuracy:.4f}")
        print(f"MLFlow run ID: {self.run_id}")

if __name__ == "__main__":
    TrainingFlow()