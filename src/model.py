import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def train_model():
    """Train Random Forest model on Iris dataset"""
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Save metrics
    save_metrics(accuracy)
    
    # Generate confusion matrix
    generate_confusion_matrix(cm)
    
    return accuracy

def save_metrics(accuracy):
    """Save accuracy metric to JSON file"""
    metrics = {
        "accuracy": round(accuracy, 4)
    }
    
    os.makedirs("assets", exist_ok=True)
    
    with open("assets/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print("Metrics saved to assets/metrics.json")

def generate_confusion_matrix(cm):
    """Generate and save confusion matrix plot"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    os.makedirs("assets", exist_ok=True)
    plt.savefig("assets/confusion_matrix.png")
    plt.close()
    
    print("Confusion matrix saved to assets/confusion_matrix.png")

if __name__ == "__main__":
    train_model()