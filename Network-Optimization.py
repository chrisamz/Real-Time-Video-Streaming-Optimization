# network_optimization.py

"""
Network Optimization Module for Real-Time Video Streaming Optimization

This module contains functions for optimizing the use of network resources for video streaming.
It uses machine learning techniques to manage and optimize bandwidth allocation, congestion control,
and adaptive bitrate streaming.

Techniques Used:
- Bandwidth allocation
- Congestion control
- Adaptive bitrate streaming

Libraries/Tools:
- TensorFlow
- PyTorch
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

class NetworkOptimization:
    def __init__(self):
        """
        Initialize the NetworkOptimization class.
        """
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

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

    def train_model(self, X_train, y_train):
        """
        Train the network optimization model.
        
        :param X_train: DataFrame, training features
        :param y_train: Series, training target
        """
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the network optimization model.
        
        :param X_test: DataFrame, testing features
        :param y_test: Series, testing target
        :return: dict, evaluation metrics
        """
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return {'mae': mae, 'rmse': rmse}

    def optimize_bandwidth(self, data):
        """
        Optimize bandwidth allocation based on the trained model.
        
        :param data: DataFrame, input data
        :return: DataFrame, data with optimized bandwidth allocation
        """
        data['optimized_bandwidth'] = self.model.predict(data)
        return data

    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        :param filepath: str, path to save the model
        """
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        """
        Load a trained model from a file.
        
        :param filepath: str, path to the saved model
        """
        self.model = joblib.load(filepath)

if __name__ == "__main__":
    data_filepath = 'data/processed/preprocessed_network_data.csv'
    target_column = 'bandwidth'

    optimizer = NetworkOptimization()
    data = optimizer.load_data(data_filepath)
    data = optimizer.preprocess_data(data)
    X_train, X_test, y_train, y_test = optimizer.split_data(data, target_column)

    # Train and evaluate model
    optimizer.train_model(X_train, y_train)
    metrics = optimizer.evaluate_model(X_test, y_test)
    print("Model Evaluation:", metrics)

    # Save the model
    optimizer.save_model('models/network_optimization_model.pkl')
    print("Model saved to 'models/network_optimization_model.pkl'.")
