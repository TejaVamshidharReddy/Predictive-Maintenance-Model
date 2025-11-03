#!/usr/bin/env python3
"""
Predictive Maintenance Model

This script implements a machine learning model to predict equipment failures
based on sensor data and maintenance logs.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


class PredictiveMaintenanceModel:
    """
    A class to build and train a predictive maintenance model.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self, filepath):
        """
        Load sensor and maintenance data from CSV file.
        
        Args:
            filepath (str): Path to the CSV data file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        print(f"Loading data from {filepath}...")
        data = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {data.shape}")
        return data
    
    def preprocess_data(self, data):
        """
        Preprocess the data by handling missing values and feature engineering.
        
        Args:
            data (pd.DataFrame): Raw data
            
        Returns:
            tuple: Features (X) and target (y)
        """
        print("Preprocessing data...")
        
        # Handle missing values
        data = data.fillna(data.mean(numeric_only=True))
        
        # Separate features and target
        X = data.drop('failure', axis=1)
        y = data['failure']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"Features: {self.feature_names}")
        print(f"Target distribution:\n{y.value_counts()}")
        
        return X, y
    
    def train(self, X_train, y_train):
        """
        Train the Random Forest classifier.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        print("\nTraining model...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize and train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        print("Model training completed.")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            dict: Performance metrics
        """
        print("\nEvaluating model...")
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        print("\n=== Model Performance ===")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        
        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred, target_names=['No Failure', 'Failure']))
        
        return metrics, y_pred, y_pred_proba
    
    def get_feature_importance(self):
        """
        Get feature importance scores from the trained model.
        
        Returns:
            pd.DataFrame: Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Features for prediction
            
        Returns:
            tuple: Predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities
    
    def save_model(self, filepath='maintenance_model.pkl'):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, filepath)
        
        print(f"\nModel saved to {filepath}")
    
    def load_model(self, filepath='maintenance_model.pkl'):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        saved_data = joblib.load(filepath)
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']
        self.feature_names = saved_data['feature_names']
        
        print(f"Model loaded from {filepath}")


def main():
    """
    Main function to run the predictive maintenance pipeline.
    """
    print("=" * 60)
    print("Predictive Maintenance Model")
    print("=" * 60)
    
    # Initialize model
    pm_model = PredictiveMaintenanceModel()
    
    # Load data
    data_path = '../data/sensor_maintenance_data.csv'
    if not os.path.exists(data_path):
        print(f"\nError: Data file not found at {data_path}")
        print("Please ensure the data file exists in the data/ directory.")
        return
    
    data = pm_model.load_data(data_path)
    
    # Preprocess data
    X, y = pm_model.preprocess_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train model
    pm_model.train(X_train, y_train)
    
    # Evaluate model
    metrics, y_pred, y_pred_proba = pm_model.evaluate(X_test, y_test)
    
    # Display feature importance
    print("\n=== Feature Importance ===")
    importance_df = pm_model.get_feature_importance()
    print(importance_df.to_string(index=False))
    
    # Save model
    pm_model.save_model('../models/maintenance_model.pkl')
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
