import os
import pytest
from PIL import Image
from ml_engine import classifier
from training.prepare_dataset import (
    create_paper_image,
    create_plastic_image,
    create_metal_image,
    create_glass_image,
    create_organic_image
)

def test_classifier_prediction_keys(tmp_path):
    img_path = str(tmp_path / "test_sample.jpg")
    img = create_paper_image()
    img.save(img_path)

    result = classifier.predict(img_path)
    assert "category" in result
    assert "confidence" in result
    assert "is_low_confidence" in result
    assert "processing_time_ms" in result
    assert "all_scores" in result

def test_paper_classification(tmp_path):
    img_path = str(tmp_path / "paper_cardboard_sample.jpg")
    img = create_paper_image()
    img.save(img_path)

    result = classifier.predict(img_path)
    assert result["category"] == "Paper"
    assert result["confidence"] > 50.0

def test_plastic_classification(tmp_path):
    img_path = str(tmp_path / "plastic_bottle_sample.jpg")
    img = create_plastic_image()
    img.save(img_path)

    result = classifier.predict(img_path)
    assert result["category"] == "Plastic"
    assert result["confidence"] > 50.0

def test_metal_classification(tmp_path):
    img_path = str(tmp_path / "metal_can_sample.jpg")
    img = create_metal_image()
    img.save(img_path)

    result = classifier.predict(img_path)
    assert result["category"] == "Metal"
    assert result["confidence"] > 50.0

def test_glass_classification(tmp_path):
    img_path = str(tmp_path / "glass_bottle_sample.jpg")
    img = create_glass_image()
    img.save(img_path)

    result = classifier.predict(img_path)
    assert result["category"] == "Glass"
    assert result["confidence"] > 50.0

def test_organic_classification(tmp_path):
    img_path = str(tmp_path / "organic_food_sample.jpg")
    img = create_organic_image()
    img.save(img_path)

    result = classifier.predict(img_path)
    assert result["category"] == "Organic"
    assert result["confidence"] > 50.0

def test_inference_performance_kpi(tmp_path):
    img_path = str(tmp_path / "kpi_test.jpg")
    img = Image.new("RGB", (224, 224), color=(120, 120, 120))
    img.save(img_path)

    result = classifier.predict(img_path)
    # KPI requirement: classification under 3000ms (3 seconds)
    assert result["processing_time_ms"] < 3000.0
