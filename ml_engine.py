import os
import json
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "waste_classifier.pth")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")

# Default Fallback Classes if file missing
DEFAULT_CLASSES = ["Glass", "Metal", "Organic", "Paper", "Plastic"]

# Exact Preprocessing Transforms matching Training
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

class WasteClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.categories = self._load_class_names()
        self.model = None
        self.confidence_threshold = 0.55  # 55% minimum threshold for high certainty

    def _load_class_names(self):
        if os.path.exists(CLASS_NAMES_PATH):
            try:
                with open(CLASS_NAMES_PATH, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading class names: {e}")
        return DEFAULT_CLASSES

    def _load_model(self):
        """Lazy load trained MobileNetV3 waste classification model on CPU/CUDA."""
        if self.model is not None:
            return

        num_classes = len(self.categories)
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

        if os.path.exists(MODEL_PATH):
            try:
                state_dict = torch.load(MODEL_PATH, map_location=self.device)
                model.load_state_dict(state_dict)
                print(f"Loaded trained waste classification model from: {MODEL_PATH}")
            except Exception as e:
                print(f"Warning loading model state dict: {e}")

        model.to(self.device)
        model.eval()
        self.model = model

    def predict(self, image_path, original_filename=None):
        """
        Runs ML model inference on an input waste image.
        Returns:
            category (str)
            confidence (float, 0-100)
            processing_time_ms (float)
            all_scores (dict)
        """
        start_time = time.time()
        self._load_model()

        try:
            img_pil = Image.open(image_path).convert('RGB')
            tensor_img = inference_transforms(img_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor_img)
                probabilities = torch.softmax(logits, dim=1).squeeze(0)

            scores_dict = {}
            for idx, cls_name in enumerate(self.categories):
                scores_dict[cls_name] = float(round(probabilities[idx].item() * 100.0, 1))

            top_prob, top_idx = torch.max(probabilities, dim=0)
            top_category = self.categories[top_idx.item()]
            top_confidence = float(round(top_prob.item() * 100.0, 1))

            # Uncertainty / Low confidence handling
            is_low_confidence = top_prob.item() < self.confidence_threshold

            processing_time_ms = float(round((time.time() - start_time) * 1000.0, 1))

            return {
                'category': top_category,
                'confidence': top_confidence,
                'is_low_confidence': is_low_confidence,
                'processing_time_ms': processing_time_ms,
                'all_scores': scores_dict
            }

        except Exception as e:
            print(f"Inference error: {e}")
            processing_time_ms = float(round((time.time() - start_time) * 1000.0, 1))
            return {
                'category': "Other/Unknown",
                'confidence': 0.0,
                'is_low_confidence': True,
                'processing_time_ms': processing_time_ms,
                'all_scores': {cat: 0.0 for cat in self.categories}
            }

classifier = WasteClassifier()
