# ============================================
# Wildfire Classification Project - Enhanced
# ============================================

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, 
    precision_recall_curve, average_precision_score
)
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================
# Dataset Setup
# ============================================

# Shoutout Chat for all the help loading data.
# I was very confused and it was able to help me get my
# code working.

base_dir = "archive" 

def collect_files(base):
    paths = []
    labels = []
    classes = {"nowildfire": 0, "wildfire": 1}

    for split in ["train", "valid", "test"]:
        for cls in ["nowildfire", "wildfire"]:
            folder = os.path.join(base, split, cls)
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    paths.append(os.path.join(folder, fname))
                    labels.append(classes[cls])
    return paths, labels


def load_and_resize(path, size=(64, 64)):
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(size)
        return np.array(img).flatten()
    except:
        return None


image_paths, labels = collect_files(base_dir)
print(f"Total images: {len(image_paths)}")

X_list = []
y_list = []
for path, label in zip(image_paths, labels):
    arr = load_and_resize(path)
    if arr is not None:
        X_list.append(arr)
        y_list.append(label)

X = np.array(X_list) / 255.0
y = np.array(y_list)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

print("Data normalized using training set statistics")


# ============================================
# PyTorch Dataset
# ============================================
# Chat also helped me with splitting the data
# with tensors.

class WildfireDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = WildfireDataset(X_train, y_train)
val_dataset = WildfireDataset(X_val, y_val)
test_dataset = WildfireDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# ============================================
# Training Function with Early Stopping
# ============================================

def train_with_validation(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=5):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            pred = model(images).squeeze()
            loss = loss_fn(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                pred = model(images).squeeze()
                loss = loss_fn(pred, labels)
                val_loss += loss.item() / len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        

        # The early stopping was something chat recommended to me
        # I had it help me implement this portion, and explain why
        # this would be helpful.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break
    
    return train_losses, val_losses


def get_predictions(model, data_loader):
    model.eval()
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for images, _ in data_loader:
            probs = torch.sigmoid(model(images).squeeze()).numpy()
            all_probs.extend(probs)
            all_preds.extend((probs >= 0.5).astype(int))
    
    return np.array(all_preds), np.array(all_probs)


# ============================================
# Model 1: Logistic Regression
# ============================================

print("\n" + "="*50)
print("Training Logistic Regression")
print("="*50)

model1 = nn.Linear(X_train.shape[1], 1)
train_losses1, val_losses1 = train_with_validation(model1, train_loader, val_loader, epochs=50, lr=1e-2)

preds1, probs1 = get_predictions(model1, test_loader)
acc1 = accuracy_score(y_test, preds1)
print(f"Test Accuracy: {acc1:.4f}")


# ============================================
# Model 2: Deep Neural Network
# ============================================

print("\n" + "="*50)
print("Training Deep Neural Network")
print("="*50)

model2 = nn.Sequential(
    nn.Linear(X_train.shape[1], 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)
train_losses2, val_losses2 = train_with_validation(model2, train_loader, val_loader, epochs=50, lr=1e-3)

preds2, probs2 = get_predictions(model2, test_loader)
acc2 = accuracy_score(y_test, preds2)
print(f"Test Accuracy: {acc2:.4f}")


# ============================================
# Model 3: CNN
# ============================================

print("\n" + "="*50)
print("Training CNN")
print("="*50)

X_train_cnn = X_train.reshape(-1, 3, 64, 64)
X_val_cnn = X_val.reshape(-1, 3, 64, 64)
X_test_cnn = X_test.reshape(-1, 3, 64, 64)

train_dataset_cnn = WildfireDataset(X_train_cnn, y_train)
val_dataset_cnn = WildfireDataset(X_val_cnn, y_val)
test_dataset_cnn = WildfireDataset(X_test_cnn, y_test)

train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=32, shuffle=True)
val_loader_cnn = DataLoader(val_dataset_cnn, batch_size=32, shuffle=False)
test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=32, shuffle=False)

model3 = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(64 * 16 * 16, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1)
)
train_losses3, val_losses3 = train_with_validation(model3, train_loader_cnn, val_loader_cnn, epochs=30, lr=1e-3)

preds3, probs3 = get_predictions(model3, test_loader_cnn)
acc3 = accuracy_score(y_test, preds3)
print(f"Test Accuracy: {acc3:.4f}")


# ============================================
# Model 4: Random Forest
# ============================================

print("\n" + "="*50)
print("Training Random Forest")
print("="*50)

from sklearn.ensemble import RandomForestClassifier

model4 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model4.fit(X_train, y_train)
preds4 = model4.predict(X_test)
probs4 = model4.predict_proba(X_test)[:, 1]
acc4 = accuracy_score(y_test, preds4)
print(f"Test Accuracy: {acc4:.4f}")


# ============================================
# Evaluation Functions
# ============================================
# I also recieved a lot of help with the evaluation
# and graphing functions.

def evaluate_model(y_true, y_pred, y_proba, model_name):
    print(f"\n{'='*60}")
    print(f"Evaluation: {model_name}")
    print('='*60)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    print(f"Accuracy:      {accuracy:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1-Score:      {f1:.4f}")
    print(f"ROC-AUC:       {roc_auc:.4f}")
    print(f"Avg Precision: {avg_precision:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:5d}  |  FP: {fp:5d}")
    print(f"  FN: {fn:5d}  |  TP: {tp:5d}")
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"\nSpecificity: {specificity:.4f}")
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'specificity': specificity,
        'confusion_matrix': cm
    }


def plot_training_curves(losses_dict):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    models = ['Logistic Regression', 'Deep NN', 'CNN']
    for idx, (model_name, (train_losses, val_losses)) in enumerate(losses_dict.items()):
        axes[idx].plot(train_losses, label='Train Loss', linewidth=2)
        axes[idx].plot(val_losses, label='Val Loss', linewidth=2)
        axes[idx].set_title(models[idx], fontweight='bold')
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel('Loss')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    plt.show()


def plot_roc_curves(models_data, y_test):
    plt.figure(figsize=(10, 8))
    
    for model_data in models_data:
        fpr, tpr, _ = roc_curve(y_test, model_data['y_proba'])
        auc = roc_auc_score(y_test, model_data['y_proba'])
        plt.plot(fpr, tpr, label=f"{model_data['name']} (AUC = {auc:.3f})", linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300)
    plt.show()


def plot_precision_recall_curves(models_data, y_test):
    plt.figure(figsize=(10, 8))
    
    for model_data in models_data:
        precision, recall, _ = precision_recall_curve(y_test, model_data['y_proba'])
        avg_prec = average_precision_score(y_test, model_data['y_proba'])
        plt.plot(recall, precision, label=f"{model_data['name']} (AP = {avg_prec:.3f})", linewidth=2)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('pr_curves.png', dpi=300)
    plt.show()


def plot_confusion_matrices(results_list):
    n_models = len(results_list)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, result in enumerate(results_list):
        sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', 
                    cmap='Blues', ax=axes[idx], cbar=False,
                    xticklabels=['No Fire', 'Fire'],
                    yticklabels=['No Fire', 'Fire'])
        axes[idx].set_title(result['model'], fontweight='bold', fontsize=12)
        axes[idx].set_ylabel('True Label', fontsize=11)
        axes[idx].set_xlabel('Predicted Label', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300)
    plt.show()


def plot_metrics_comparison(results_list):
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    models = [r['model'] for r in results_list]
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, metric in enumerate(metrics):
        values = [r[metric] for r in results_list]
        axes[idx].bar(models, values, color=colors[:len(models)])
        axes[idx].set_title(metric.replace('_', ' ').upper(), fontweight='bold', fontsize=12)
        axes[idx].set_ylim([0, 1.05])
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300)
    plt.show()


def create_results_table(results_list):
    df = pd.DataFrame(results_list)
    df = df[['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'avg_precision', 'specificity']]
    df = df.round(4)
    
    print("\n" + "="*100)
    print("COMPREHENSIVE RESULTS TABLE")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)
    
    df.to_csv('model_results.csv', index=False)
    print("\nResults saved to 'model_results.csv'")
    
    return df


# ============================================
# Run All Evaluations
# ============================================

print("\n\n" + "="*60)
print("COMPREHENSIVE MODEL EVALUATION")
print("="*60)

results = []
results.append(evaluate_model(y_test, preds1, probs1, "Logistic Regression"))
results.append(evaluate_model(y_test, preds2, probs2, "Deep Neural Network"))
results.append(evaluate_model(y_test, preds3, probs3, "CNN"))
results.append(evaluate_model(y_test, preds4, probs4, "Random Forest"))

df_results = create_results_table(results)

models_data = [
    {'name': 'Logistic Regression', 'y_proba': probs1},
    {'name': 'Deep NN', 'y_proba': probs2},
    {'name': 'CNN', 'y_proba': probs3},
    {'name': 'Random Forest', 'y_proba': probs4}
]

losses_dict = {
    'model1': (train_losses1, val_losses1),
    'model2': (train_losses2, val_losses2),
    'model3': (train_losses3, val_losses3)
}
plot_training_curves(losses_dict)

print("\nGenerating visualizations...")
plot_roc_curves(models_data, y_test)
plot_precision_recall_curves(models_data, y_test)
plot_confusion_matrices(results)
plot_metrics_comparison(results)

print("\n✓ All visualizations saved!")
print("  - training_curves.png")
print("  - roc_curves.png")
print("  - pr_curves.png")
print("  - confusion_matrices.png")
print("  - metrics_comparison.png")

best_model_idx = np.argmax([r['f1'] for r in results])
print(f"\n Best Model (by F1-score): {results[best_model_idx]['model']}")
print(f"   F1-Score: {results[best_model_idx]['f1']:.4f}")
print(f"   ROC-AUC: {results[best_model_idx]['roc_auc']:.4f}")