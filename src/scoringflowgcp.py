from metaflow import FlowSpec, step, Parameter, resources, kubernetes, conda_base, retry, timeout, catch
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import mlflow

@conda_base(libraries={'python': '3.12.9', 'scikit-learn': '1.5.1', 'pandas': '2.2.2', 'numpy': '1.26.4', 'mlflow': '2.15.1'})
class ScoringFlow(FlowSpec):
    run_id = Parameter('run_id', help='MLFlow run ID of the model to use')
    
    @kubernetes(memory=1024, cpu=0.5, disk=1000)
    @resources(cpu=1, memory=1000)
    @step
    def start(self):
        print(f"Starting scoring flow with model from run {self.run_id}")
        self.next(self.load_data)
    
    @kubernetes(memory=1024, cpu=0.5, disk=1000)
    @resources(cpu=1, memory=2000)
    @timeout(minutes=10)
    @retry(times=3)
    @catch(var='load_error')
    @step
    def load_data(self):
        # Load iris dataset for scoring
        iris = load_iris()
        data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        data['target'] = iris.target
        
        # Split data - we'll use the test portion for scoring
        _, self.data = train_test_split(data, test_size=0.3, random_state=42)
        print(f"Loaded {len(self.data)} samples for scoring")
        
        if hasattr(self, 'load_error') and self.load_error:
            print(f"Error handled: {self.load_error}")
            
        self.next(self.feature_engineering)
    
    @kubernetes(memory=1024, cpu=0.5, disk=1000)
    @resources(cpu=1, memory=2000)
    @timeout(minutes=15)
    @retry(times=2)
    @catch(var='feature_error')
    @step
    def feature_engineering(self):
        # Apply the same feature selection as in training
        print("Applying feature selection (SelectKBest_k3)")
        
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        
        # Apply SelectKBest with k=3 to match training
        selector = SelectKBest(f_classif, k=3)
        X_new = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        print(f"Selected features: {selected_features}")
        
        # Update dataframe to only include selected features
        self.processed_X = pd.DataFrame(X_new, columns=selected_features)
        self.target_y = y
        
        if hasattr(self, 'feature_error') and self.feature_error:
            print(f"Error handled: {self.feature_error}")
            
        self.next(self.load_model)
    
    @kubernetes(memory=1024, cpu=0.5, disk=1000)
    @resources(cpu=1, memory=2000)
    @timeout(minutes=10)
    @retry(times=3)
    @catch(var='model_error')
    @step
    def load_model(self):
        # Set MLflow tracking URI to match training flow
        mlflow.set_tracking_uri('https://mlflow-service-255971508831.us-west2.run.app')
        
        # Load the registered model from MLFlow
        print(f"Loading model from MLFlow run {self.run_id}")
        self.model = mlflow.sklearn.load_model(f"runs:/{self.run_id}/model")
        
        if hasattr(self, 'model_error') and self.model_error:
            print(f"Error handled: {self.model_error}")
            
        self.next(self.make_predictions)
    
    @kubernetes(memory=1024, cpu=0.5, disk=1000)
    @resources(cpu=1, memory=2000)
    @timeout(minutes=10)
    @retry(times=2)
    @catch(var='prediction_error')
    @step
    def make_predictions(self):
        # Make predictions on test data
        self.predictions = self.model.predict(self.processed_X)
        self.probabilities = self.model.predict_proba(self.processed_X)
        
        print(f"Made predictions for {len(self.predictions)} samples")
        
        if hasattr(self, 'prediction_error') and self.prediction_error:
            print(f"Error handled: {self.prediction_error}")
            
        self.next(self.evaluate_model)
    
    @kubernetes(memory=1024, cpu=0.5, disk=1000)
    @resources(cpu=1, memory=2000)
    @timeout(minutes=15)
    @retry(times=2)
    @catch(var='evaluation_error')
    @step
    def evaluate_model(self):
        # Calculate performance metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        self.accuracy = accuracy_score(self.target_y, self.predictions)
        self.precision = precision_score(self.target_y, self.predictions, average='weighted')
        self.recall = recall_score(self.target_y, self.predictions, average='weighted')
        self.f1 = f1_score(self.target_y, self.predictions, average='weighted')
        
        print(f"Scoring metrics:")
        print(f"  Accuracy:  {self.accuracy:.4f}")
        print(f"  Precision: {self.precision:.4f}")
        print(f"  Recall:    {self.recall:.4f}")
        print(f"  F1-Score:  {self.f1:.4f}")
        
        # Log scoring results to MLflow
        mlflow.set_experiment('iris-experiment-scoring-gcp')
        with mlflow.start_run():
            # Log reference to training run
            mlflow.log_param("training_run_id", self.run_id)
            
            # Log metrics
            mlflow.log_metric("accuracy", self.accuracy)
            mlflow.log_metric("precision", self.precision)
            mlflow.log_metric("recall", self.recall)
            mlflow.log_metric("f1", self.f1)
            
            print("Logged scoring results to MLflow")
        
        if hasattr(self, 'evaluation_error') and self.evaluation_error:
            print(f"Error handled: {self.evaluation_error}")
            
        self.next(self.end)
    
    @kubernetes(memory=1024, cpu=0.5, disk=1000)
    @resources(cpu=1, memory=1000)
    @step
    def end(self):
        print("Scoring flow complete!")

if __name__ == "__main__":
    ScoringFlow()