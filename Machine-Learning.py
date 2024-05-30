# machine_learning.py

"""
Machine Learning Module for Real-Time Video Streaming Optimization

This module contains functions for building, training, and evaluating machine learning models
to predict network conditions and optimize video streaming parameters in real-time.

Techniques Used:
- Regression
- Classification
- Reinforcement Learning

Algorithms Used:
- Linear Regression
- Random Forest
- Deep Q-Learning (DQN)

Libraries/Tools:
- TensorFlow
- PyTorch
- scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

class MachineLearning:
    def __init__(self):
        """
        Initialize the MachineLearning class.
        """
        self.models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }

    def load_data(self, filepath):
        """
        Load data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        return pd.read_csv(filepath, parse_dates=['timestamp'])

    def preprocess_data(self, data):
        """
        Preprocess the data for training.
        
        :param data: DataFrame, input data
        :return: DataFrame, preprocessed data
        """
        # Convert categorical variables to numerical
        data = pd.get_dummies(data, columns=['network_type', 'device_type'], drop_first=True)
        # Fill missing values
        data = data.fillna(data.mean())
        return data

    def split_data(self, data, target_column, test_size=0.2):
        """
        Split the data into training and testing sets.
        
        :param data: DataFrame, input data
        :param target_column: str, name of the target column
        :param test_size: float, proportion of the data to include in the test split
        :return: tuple, training and testing sets
        """
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_model(self, model_name, X_train, y_train):
        """
        Train the specified model.
        
        :param model_name: str, name of the model to train
        :param X_train: DataFrame, training features
        :param y_train: Series, training target
        """
        model = self.models.get(model_name)
        if model:
            model.fit(X_train, y_train)
            return model
        else:
            raise ValueError(f"Model {model_name} not found.")

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the specified model using various metrics.
        
        :param model: trained model
        :param X_test: DataFrame, testing features
        :param y_test: Series, testing target
        :return: dict, evaluation metrics
        """
        y_pred = model.predict(X_test)
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        return metrics

    def save_model(self, model_name, filepath):
        """
        Save the trained model to a file.
        
        :param model_name: str, name of the model to save
        :param filepath: str, path to save the model
        """
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found.")
        joblib.dump(model, filepath)

    def load_model(self, model_name, filepath):
        """
        Load a trained model from a file.
        
        :param model_name: str, name to assign to the loaded model
        :param filepath: str, path to the saved model
        """
        self.models[model_name] = joblib.load(filepath)

if __name__ == "__main__":
    data_filepath = 'data/processed/preprocessed_network_data.csv'
    target_column = 'bandwidth'

    ml = MachineLearning()
    data = ml.load_data(data_filepath)
    data = ml.preprocess_data(data)
    X_train, X_test, y_train, y_test = ml.split_data(data, target_column)

    # Train and evaluate linear regression model
    lr_model = ml.train_model('linear_regression', X_train, y_train)
    lr_metrics = ml.evaluate_model(lr_model, X_test, y_test)
    print("Linear Regression Evaluation:", lr_metrics)

    # Train and evaluate random forest model
    rf_model = ml.train_model('random_forest', X_train, y_train)
    rf_metrics = ml.evaluate_model(rf_model, X_test, y_test)
    print("Random Forest Evaluation:", rf_metrics)

    # Save the models
    ml.save_model('linear_regression', 'models/linear_regression_model.pkl')
    ml.save_model('random_forest', 'models/random_forest_model.pkl')
    print("Models saved to 'models/linear_regression_model.pkl' and 'models/random_forest_model.pkl'.")
