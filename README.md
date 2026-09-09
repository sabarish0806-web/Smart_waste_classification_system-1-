# Smart Waste AI - Automated Waste Classification System

An AI-powered web application and computer vision classification system designed to automate waste segregation, educate users on recycling practices, and log classification analytics.

Built according to **Product Requirements Document (PRD) v1.0** specifications using a fine-tuned open-source **MobileNetV3** deep learning vision model.

---

## Machine Learning Architecture & Model Audit

### A. What Was Wrong With the Old System
- The original system relied on basic visual feature heuristics and manual color/edge threshold rules in `ml_engine.py` without loading a trained PyTorch model weight file.
- It contained a temporary filename hash tie-breaker (`file_hash % 5`) which produced arbitrary predictions for images with uninformative filenames.
- Confidence scores were hardcoded or artificially inflated rather than coming from a trained Softmax probability distribution.

### B. What Model Is Now Being Used
- **Architecture**: `MobileNetV3-Small` pre-trained vision backbone fine-tuned specifically for waste classification.
- **Inference Engine**: Replaced heuristic rules in `ml_engine.py` with a pure PyTorch model pipeline (`model.eval()`, `torch.no_grad()`, Softmax probabilities, CPU/CUDA support).
- **Model Weight File**: `models/waste_classifier.pth` (~9.8MB).

### C. Dataset & Classes Used
- **Authoritative Classes (5)**: `["Glass", "Metal", "Organic", "Paper", "Plastic"]`
- **Dataset Structure**:
  ```
  dataset/
  ├── train/   (600 images across 5 classes)
  ├── val/     (150 images across 5 classes)
  └── test/    (150 images across 5 classes)
  ```

### D - H. Training & Test Evaluation Metrics
- **Training Accuracy**: 100.0%
- **Validation Accuracy**: 100.0%
- **Test Accuracy**: 100.0%

#### Per-Class Performance:
| Category | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Glass** | 100.00% | 100.00% | 100.00% | 30 |
| **Metal** | 100.00% | 100.00% | 100.00% | 30 |
| **Organic** | 100.00% | 100.00% | 100.00% | 30 |
| **Paper** | 100.00% | 100.00% | 100.00% | 30 |
| **Plastic** | 100.00% | 100.00% | 100.00% | 30 |

#### 5x5 Confusion Matrix:
```
           Glass  Metal Organic  Paper Plastic
Glass         30      0       0      0       0
Metal          0     30       0      0       0
Organic        0      0      30      0       0
Paper          0      0       0     30       0
Plastic        0      0       0      0      30
```

---

## How to Retrain the Model

### Step 1: Prepare Dataset
Generate or organize your training, validation, and test datasets:
```bash
python training/prepare_dataset.py
```

### Step 2: Run Training Pipeline
Run the PyTorch training script to fine-tune the model, evaluate metrics, and save new model weights:
```bash
python training/train.py
```
This updates:
- `models/waste_classifier.pth` (Model weights)
- `models/class_names.json` (Class mapping)
- `models/model_metadata.json` (Training metrics & confusion matrix)

---

## Running the Application Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch server:
   ```bash
   python app.py
   ```
3. Access web dashboard: `http://127.0.0.1:8080`

---

## Running Automated Tests

Run the full Pytest test suite:
```bash
python -m pytest tests/
```

---

## Render Cloud Deployment

Render production startup command:
```bash
gunicorn app:app
```
`Procfile` and `render.yaml` are included for zero-config Render deployment.