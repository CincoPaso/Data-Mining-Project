# ============================================
# Wildfire Classification Project - Enhanced with CAM
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
from matplotlib.colors import LinearSegmentedColormap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================
# Dataset Setup
# ============================================
# Setting up the data was an area where I had a lot of help
# I used Claude and Chat GPT to help me create this data loader
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
path_list = []  # Keep track of paths for visualization
for path, label in zip(image_paths, labels):
    arr = load_and_resize(path)
    if arr is not None:
        X_list.append(arr)
        y_list.append(label)
        path_list.append(path)

X = np.array(X_list) / 255.0
y = np.array(y_list)

X_train, X_temp, y_train, y_temp, paths_train, paths_temp = train_test_split(
    X, y, path_list, test_size=0.40, random_state=42
)
X_val, X_test, y_val, y_test, paths_val, paths_test = train_test_split(
    X_temp, y_temp, paths_temp, test_size=0.50, random_state=42
)

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
        
        # Stopping early was something recommended by Chat GPT
        # Chat helped me input this small portion

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
# Model 3: CNN with CAM Support
# ============================================

print("\n" + "="*50)
print("Training CNN with CAM Support")
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

# Claude helped me adapt my original CNN to include CAM support
class CNNWithCAM(nn.Module):
    """CNN designed to support Class Activation Mapping"""
    def __init__(self):
        super(CNNWithCAM, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling for CAM
        self.fc = nn.Linear(64, 1)
        
        # Store feature maps for CAM
        self.feature_maps = None
        
    def forward(self, x):
        x = self.features(x)
        self.feature_maps = x  # Save for CAM
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


model3 = CNNWithCAM()
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
# Class Activation Mapping (CAM)
# ============================================
# Recieved guidance and input on this portion of the code
# from Claude the next 3 functions are all things that I 
# struggled with, so I got assistance.

def generate_cam(model, image_tensor):
    """Generate Class Activation Map for a single image"""
    model.eval()
    with torch.no_grad():
        # Forward pass
        output = model(image_tensor.unsqueeze(0))
        
        # Get feature maps (last conv layer output)
        feature_maps = model.feature_maps[0]  # Shape: [64, 16, 16]
        
        # Get weights from the final FC layer
        weights = model.fc.weight.data[0]  # Shape: [64]
        
        # Generate CAM by weighted combination of feature maps
        cam = torch.zeros(feature_maps.shape[1:])
        for i, w in enumerate(weights):
            cam += w * feature_maps[i]
        
        # Normalize CAM
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()


def visualize_cam_overlay(original_image, cam, alpha=0.5):
    """Create overlay of CAM on original image"""
    # Resize CAM to match original image
    cam_resized = np.array(Image.fromarray(cam).resize((64, 64), Image.BILINEAR))
    
    # Create heatmap colormap (blue -> green -> yellow -> red)
    cmap = plt.cm.jet
    cam_colored = cmap(cam_resized)[:, :, :3]  # Remove alpha channel
    
    # Overlay on original
    overlay = (1 - alpha) * original_image + alpha * cam_colored
    overlay = np.clip(overlay, 0, 1)
    
    return overlay, cam_colored


def plot_cam_examples(model, X_test_cnn, y_test, paths_test, n_samples=8):
    """Plot CAM visualizations for sample images"""
    fig, axes = plt.subplots(4, n_samples, figsize=(n_samples*3, 12))
    
    # Select mix of correct and incorrect predictions
    model.eval()
    with torch.no_grad():
        test_probs = torch.sigmoid(model(torch.FloatTensor(X_test_cnn))).numpy().flatten()
    test_preds = (test_probs >= 0.5).astype(int)
    
    # Get some correct wildfire, correct no-wildfire, and some misclassifications
    correct_fire = np.where((y_test == 1) & (test_preds == 1))[0]
    correct_nofire = np.where((y_test == 0) & (test_preds == 0))[0]
    
    samples = []
    samples.extend(np.random.choice(correct_fire, min(4, len(correct_fire)), replace=False))
    samples.extend(np.random.choice(correct_nofire, min(4, len(correct_nofire)), replace=False))
    
    for col, idx in enumerate(samples):
        # Original image
        original = X_test_cnn[idx].transpose(1, 2, 0)
        original = (original - original.min()) / (original.max() - original.min())
        
        # Generate CAM
        image_tensor = torch.FloatTensor(X_test_cnn[idx])
        cam = generate_cam(model, image_tensor)
        
        # Create overlay
        overlay, cam_colored = visualize_cam_overlay(original, cam, alpha=0.4)
        
        # Plot original
        axes[0, col].imshow(original)
        axes[0, col].set_title(f"Original\nTrue: {'Fire' if y_test[idx] else 'No Fire'}", 
                               fontsize=9, fontweight='bold')
        axes[0, col].axis('off')
        
        # Plot CAM heatmap
        axes[1, col].imshow(cam, cmap='jet')
        axes[1, col].set_title(f"CAM Heatmap\nPred: {test_probs[idx]:.2f}", 
                               fontsize=9, fontweight='bold')
        axes[1, col].axis('off')
        
        # Plot overlay
        axes[2, col].imshow(overlay)
        axes[2, col].set_title("CAM Overlay", fontsize=9, fontweight='bold')
        axes[2, col].axis('off')
        
        # Plot RGB channels separately
        rgb_composite = np.concatenate([
            np.expand_dims(original[:, :, 0], axis=1),
            np.expand_dims(original[:, :, 1], axis=1),
            np.expand_dims(original[:, :, 2], axis=1)
        ], axis=1)
        rgb_composite = rgb_composite.reshape(64, -1)
        axes[3, col].imshow(rgb_composite, cmap='gray')
        axes[3, col].set_title("R | G | B Channels", fontsize=9, fontweight='bold')
        axes[3, col].axis('off')
    
    plt.suptitle("Class Activation Mapping (CAM) - Wildfire Risk Visualization", 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('cam_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n✓ CAM visualization saved to 'cam_visualization.png'")


# ============================================
# Feature Channel Visualization
# ============================================

# Recieved guidance and input on this portion of the code
# from Claude 

def plot_feature_channels(X_test_cnn, y_test, n_samples=6):
    """Visualize different feature representations of satellite images"""
    fig, axes = plt.subplots(n_samples, 7, figsize=(21, n_samples*3))
    
    # Select diverse samples
    fire_samples = np.where(y_test == 1)[0][:n_samples//2]
    nofire_samples = np.where(y_test == 0)[0][:n_samples//2]
    samples = np.concatenate([fire_samples, nofire_samples])
    
    for row, idx in enumerate(samples):
        img = X_test_cnn[idx].transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        
        # Original RGB
        axes[row, 0].imshow(img)
        axes[row, 0].set_title("RGB Composite", fontsize=9, fontweight='bold')
        axes[row, 0].axis('off')
        
        # Individual channels
        axes[row, 1].imshow(img[:, :, 0], cmap='Reds')
        axes[row, 1].set_title("Red Channel", fontsize=9, fontweight='bold')
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(img[:, :, 1], cmap='Greens')
        axes[row, 2].set_title("Green Channel", fontsize=9, fontweight='bold')
        axes[row, 2].axis('off')
        
        axes[row, 3].imshow(img[:, :, 2], cmap='Blues')
        axes[row, 3].set_title("Blue Channel", fontsize=9, fontweight='bold')
        axes[row, 3].axis('off')
        
        # Computed indices
        # Normalized Difference (simulated vegetation index)
        nd = (img[:, :, 1] - img[:, :, 0]) / (img[:, :, 1] + img[:, :, 0] + 1e-8)
        axes[row, 4].imshow(nd, cmap='RdYlGn')
        axes[row, 4].set_title("Vegetation Index\n(G-R)/(G+R)", fontsize=9, fontweight='bold')
        axes[row, 4].axis('off')
        
        # Intensity (brightness)
        intensity = img.mean(axis=2)
        axes[row, 5].imshow(intensity, cmap='gray')
        axes[row, 5].set_title("Intensity", fontsize=9, fontweight='bold')
        axes[row, 5].axis('off')
        
        # Edge detection (Sobel-like)
        from scipy import ndimage
        edges = ndimage.sobel(intensity)
        axes[row, 6].imshow(edges, cmap='hot')
        axes[row, 6].set_title("Edge Detection", fontsize=9, fontweight='bold')
        axes[row, 6].axis('off')
        
        # Add label
        label_text = "WILDFIRE" if y_test[idx] == 1 else "NO WILDFIRE"
        color = 'red' if y_test[idx] == 1 else 'green'
        axes[row, 0].text(2, 8, label_text, color=color, fontsize=8, 
                         fontweight='bold', bbox=dict(boxstyle='round', 
                         facecolor='white', alpha=0.8))
    
    plt.suptitle("Satellite Image Feature Analysis - Multiple Channel Representations", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('feature_channels.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Feature channel visualization saved to 'feature_channels.png'")


# ============================================
# Evaluation Functions
# ============================================

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

# Recieved guidance here
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

# Recieved guidance here
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

print("\nGenerating standard visualizations...")
plot_roc_curves(models_data, y_test)
plot_precision_recall_curves(models_data, y_test)
plot_confusion_matrices(results)
plot_metrics_comparison(results)

# ============================================
# NEW: Generate CAM and Feature Visualizations
# ============================================

print("\n" + "="*60)
print("GENERATING CAM AND FEATURE VISUALIZATIONS")
print("="*60)

print("\nGenerating Class Activation Maps...")
plot_cam_examples(model3, X_test_cnn, y_test, paths_test, n_samples=8)

print("\nGenerating Feature Channel Analysis...")
plot_feature_channels(X_test_cnn, y_test, n_samples=6)

print("\n✓ All visualizations completed and saved!")
print("\nGenerated files:")
print("  - training_curves.png")
print("  - roc_curves.png")
print("  - pr_curves.png")
print("  - confusion_matrices.png")
print("  - metrics_comparison.png")
print("  - cam_visualization.png  [NEW]")
print("  - feature_channels.png  [NEW]")
print("  - model_results.csv")

best_model_idx = np.argmax([r['f1'] for r in results])
print(f"\n Best Model (by F1-score): {results[best_model_idx]['model']}")
print(f"   F1-Score: {results[best_model_idx]['f1']:.4f}")
print(f"   ROC-AUC: {results[best_model_idx]['roc_auc']:.4f}")