import pytest
import json
import os

def test_model_accuracy():
    """Test that model accuracy meets minimum threshold"""
    
    # Check if metrics file exists
    assert os.path.exists("assets/metrics.json"), "Metrics file not found"
    
    # Load metrics
    with open("assets/metrics.json", "r") as f:
        metrics = json.load(f)
    
    # Check accuracy threshold
    accuracy = metrics.get("accuracy", 0)
    assert accuracy > 0.8, f"Model accuracy {accuracy} is below threshold 0.8"
    
    print(f"✓ Model accuracy test passed: {accuracy}")

def test_confusion_matrix_exists():
    """Test that confusion matrix image was generated"""
    assert os.path.exists("assets/confusion_matrix.png"), "Confusion matrix not found"
    print("✓ Confusion matrix file exists")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])