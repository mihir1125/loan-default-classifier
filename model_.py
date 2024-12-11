import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

class BaseModel:

    def load(self, filepath: str, verbose: bool = False) -> None:
        """Loads training data from a .xlsx file."""
        self.data = pd.read_excel(filepath)
        if verbose:
            print(f"Data loaded from {filepath}")

    def preprocess(self):
        """Preprocess the data."""

        X = self.data.drop(columns=['loan_status', 'transaction_date', 'customer_id'])
        y = self.data['loan_status']
        
        self.numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
        self.categorical_features = X.select_dtypes(include=['object']).columns
        
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Apply transformations
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )

        self.preprocessor = preprocessor
        
        X_processed = self.preprocessor.fit_transform(X)
        return X_processed, y
    
    def test(self, filepath: str, verbose: bool = False) -> None:
        """Loads and tests .xslsx data from given path."""
        self.traindata = pd.read_excel(filepath)
        if verbose:
            print(f"Test data loaded from {filepath}")
        
        X_test = self.traindata.drop(columns=['loan_status', 'transaction_date', 'customer_id'])
        y_test = self.traindata['loan_status']
        
        X_test = self.preprocessor.transform(X_test)

        y_pred = self.model.predict(X_test)
        print("Model Evaluation Summary:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    
    def predict(self, X) -> int:
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed)

class LRModel(BaseModel):
    def __init__(self, max_iter: int):
        self.max_iter = max_iter
    
    def train(self, X_train, y_train) -> None:
        self.model = LogisticRegression(max_iter = self.max_iter)
        self.model.fit(X_train, y_train)
    
class RFModel(BaseModel):
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
    
    def train(self, X_train, y_train) -> None:
        self.model = RandomForestClassifier(n_estimators = self.n_estimators, random_state = 42)
        self.model.fit(X_train, y_train)

class NNModel(BaseModel):
    def __init__(self, max_iter: int):
        self.max_iter = max_iter
    
    def train(self, X_train, y_train) -> None:
        self.model = MLPClassifier(max_iter=self.max_iter)
        self.model.fit(X_train, y_train)

class SVMModel(BaseModel):
    def train(self, X_train, y_train) -> None:
        self.model = SVC(gamma="auto")
        self.model.fit(X_train, y_train)