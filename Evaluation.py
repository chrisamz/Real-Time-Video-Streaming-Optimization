# evaluation.py

"""
Evaluation Module for Real-Time Video Streaming Optimization

This module contains functions for evaluating the performance of predictive models
and video compression techniques using appropriate metrics.

Techniques Used:
- Model Evaluation
- Video Compression Evaluation

Metrics Used:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import cv2
import math
import joblib

class ModelEvaluation:
    def __init__(self):
        """
        Initialize the ModelEvaluation class.
        """
        self.models = {}

    def load_data(self, filepath):
        """
        Load test data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        return pd.read_csv(filepath, parse_dates=['timestamp'])

    def load_model(self, model_name, filepath):
        """
        Load a trained model from a file.
        
        :param model_name: str, name to assign to the loaded model
        :param filepath: str, path to the saved model
        """
        self.models[model_name] = joblib.load(filepath)

    def evaluate_regression_model(self, model_name, X_test, y_test):
        """
        Evaluate the specified regression model using various metrics.
        
        :param model_name: str, name of the model to evaluate
        :param X_test: DataFrame, testing features
        :param y_test: Series, testing target
        :return: dict, evaluation metrics
        """
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found.")
        
        y_pred = model.predict(X_test)
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        return metrics

    def evaluate_compression(self, original_video_path, compressed_video_path):
        """
        Evaluate the quality of compressed video using PSNR and SSIM metrics.
        
        :param original_video_path: str, path to the original video file
        :param compressed_video_path: str, path to the compressed video file
        :return: dict, evaluation metrics
        """
        original_cap = cv2.VideoCapture(original_video_path)
        compressed_cap = cv2.VideoCapture(compressed_video_path)

        psnr_values = []
        ssim_values = []

        while True:
            ret1, frame1 = original_cap.read()
            ret2, frame2 = compressed_cap.read()

            if not ret1 or not ret2:
                break

            psnr_value = self.calculate_psnr(frame1, frame2)
            ssim_value = self.calculate_ssim(frame1, frame2)

            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)

        original_cap.release()
        compressed_cap.release()

        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)

        return {'psnr': avg_psnr, 'ssim': avg_ssim}

    def calculate_psnr(self, img1, img2):
        """
        Calculate the PSNR (Peak Signal-to-Noise Ratio) between two images.
        
        :param img1: numpy array, first image
        :param img2: numpy array, second image
        :return: float, PSNR value
        """
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    def calculate_ssim(self, img1, img2):
        """
        Calculate the SSIM (Structural Similarity Index) between two images.
        
        :param img1: numpy array, first image
        :param img2: numpy array, second image
        :return: float, SSIM value
        """
        return cv2.quality.QualitySSIM_compute(img1, img2)[0]

if __name__ == "__main__":
    test_data_filepath = 'data/processed/preprocessed_network_data_test.csv'
    target_column = 'bandwidth'

    evaluator = ModelEvaluation()
    data = evaluator.load_data(test_data_filepath)
    X_test = data.drop(columns=[target_column])
    y_test = data[target_column]

    # Load models
    evaluator.load_model('linear_regression', 'models/linear_regression_model.pkl')
    evaluator.load_model('random_forest', 'models/random_forest_model.pkl')

    # Evaluate regression models
    lr_metrics = evaluator.evaluate_regression_model('linear_regression', X_test, y_test)
    rf_metrics = evaluator.evaluate_regression_model('random_forest', X_test, y_test)

    print("Linear Regression Evaluation:", lr_metrics)
    print("Random Forest Evaluation:", rf_metrics)

    # Evaluate video compression
    original_video_path = 'data/raw/input_video.mp4'
    compressed_video_path = 'data/compressed/output_video_ffmpeg.mp4'
    compression_metrics = evaluator.evaluate_compression(original_video_path, compressed_video_path)

    print("Video Compression Evaluation:", compression_metrics)
