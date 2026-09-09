import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Set reproducible seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ImageNet normalization parameters (used for both training & inference consistency)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

def evaluate_metrics(model, dataloader, device, num_classes, class_names):
    """Calculates accuracy, per-class precision, recall, F1 score, and confusion matrix."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 5x5 Confusion Matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    total_samples = len(all_labels)
    correct = np.sum(all_preds == all_labels)
    accuracy = float(correct / total_samples) * 100.0 if total_samples > 0 else 0.0

    per_class_metrics = {}
    for idx, class_name in enumerate(class_names):
        tp = cm[idx, idx]
        fp = np.sum(cm[:, idx]) - tp
        fn = np.sum(cm[idx, :]) - tp
        
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        per_class_metrics[class_name] = {
            "precision": round(precision * 100.0, 2),
            "recall": round(recall * 100.0, 2),
            "f1_score": round(f1 * 100.0, 2),
            "support": int(np.sum(cm[idx, :]))
        }

    return {
        "accuracy": round(accuracy, 2),
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics
    }

def train_model(epochs=10, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on device: {device}")

    # Load datasets
    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir = os.path.join(DATASET_DIR, "val")
    test_dir = os.path.join(DATASET_DIR, "test")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)

    class_names = train_dataset.classes  # ["Glass", "Metal", "Organic", "Paper", "Plastic"]
    num_classes = len(class_names)
    print(f"Detected Classes ({num_classes}): {class_names}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Save class names JSON (Authoritative Class Mapping)
    class_names_path = os.path.join(MODELS_DIR, "class_names.json")
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Saved class mapping to: {class_names_path}")

    # Load Pretrained MobileNetV3-Small
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_val_acc = 0.0
    best_model_path = os.path.join(MODELS_DIR, "waste_classifier.pth")

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = float(running_corrects.double() / len(train_dataset)) * 100.0

        # Validate
        val_eval = evaluate_metrics(model, val_loader, device, num_classes, class_names)
        val_acc = val_eval["accuracy"]

        print(f"Epoch {epoch+1:02d}/{epochs:02d} - Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc or epoch == 0:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    training_duration = time.time() - start_time
    print(f"\nTraining completed in {training_duration:.2f} seconds.")
    print(f"Saved best model weights to: {best_model_path}")

    # Load best model for final evaluation on test set
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_eval = evaluate_metrics(model, test_loader, device, num_classes, class_names)

    print("\n" + "="*50)
    print("FINAL TEST EVALUATION METRICS")
    print("="*50)
    print(f"Test Accuracy: {test_eval['accuracy']}%")
    print("\nPer-Class Metrics:")
    for cls_name, metrics in test_eval['per_class_metrics'].items():
        print(f"  {cls_name:10s} -> Precision: {metrics['precision']:6.2f}% | Recall: {metrics['recall']:6.2f}% | F1: {metrics['f1_score']:6.2f}%")
    print("\nConfusion Matrix (Rows=True, Cols=Predicted):")
    print("           " + " ".join([f"{c[:6]:>6s}" for c in class_names]))
    for idx, row in enumerate(test_eval['confusion_matrix']):
        print(f"{class_names[idx]:10s} " + " ".join([f"{val:6d}" for val in row]))

    # Save metadata JSON
    metadata = {
        "model_architecture": "MobileNetV3-Small",
        "num_classes": num_classes,
        "class_names": class_names,
        "test_accuracy": test_eval["accuracy"],
        "per_class_metrics": test_eval["per_class_metrics"],
        "confusion_matrix": test_eval["confusion_matrix"],
        "training_duration_sec": round(training_duration, 2),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata to: {metadata_path}")

if __name__ == "__main__":
    train_model(epochs=12, batch_size=32, learning_rate=0.001)
