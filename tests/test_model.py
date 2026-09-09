import os
import pytest
from PIL import Image
from ml_engine import classifier

@pytest.fixture
def sample_image(tmp_path):
    """Creates a temporary sample RGB image for testing."""
    img_path = str(tmp_path / "test_waste_sample.jpg")
    img = Image.new("RGB", (300, 300), color=(100, 180, 70))  # Greenish image
    img.save(img_path)
    return img_path

def test_classifier_prediction_keys(sample_image):
    result = classifier.predict(sample_image)
    
    assert "category" in result
    assert "confidence" in result
    assert "processing_time_ms" in result
    assert "all_scores" in result

def test_classifier_category_validity(sample_image):
    result = classifier.predict(sample_image)
    valid_categories = {"Plastic", "Paper", "Metal", "Glass", "Organic", "Other/Unknown"}
    
    assert result["category"] in valid_categories

def test_classifier_confidence_range(sample_image):
    result = classifier.predict(sample_image)
    
    assert 0.0 <= result["confidence"] <= 100.0

def test_classifier_performance_kpi(sample_image):
    result = classifier.predict(sample_image)
    # KPI requirement: classification under 3000ms (3 seconds)
    assert result["processing_time_ms"] < 3000.0
