# -*- coding: utf-8 -*-
"""
@author: EHSAN_ab
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, 
                            confusion_matrix, classification_report)
import joblib
from matplotlib.colors import LinearSegmentedColormap

# ===========================================
# Configuration
# ===========================================
RESULT_PATH = r"final_superCO"
os.makedirs(RESULT_PATH, exist_ok=True)

# ===========================================
# Data Preparation
# ===========================================
print("Loading and preparing data...")
df = pd.read_csv(r"path_to_supecon_dataset", 
                encoding='unicode_escape')

# Feature and target engineering
X = df.iloc[:, 4:80].values  # 56 features
y = df['type'].values

# Encode labels
LE = LabelEncoder()
y_encoded = LE.fit_transform(y)
text_labels = LE.classes_

# Save LabelEncoder
joblib.dump(LE, os.path.join(RESULT_PATH, 'label_encoder.pkl'))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Preprocessing pipeline
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, os.path.join(RESULT_PATH, 'scaler.pkl'))

train_dataset = TensorDataset(
    torch.FloatTensor(X_train_scaled),
    torch.LongTensor(y_train)
)

test_dataset = TensorDataset(
    torch.FloatTensor(X_test_scaled),
    torch.LongTensor(y_test)
)

# ===========================================
# Model Architecture
# ===========================================
class ImbalancedClassifier(nn.Module):
    def __init__(self, input_size=56, num_classes=4):
        super(ImbalancedClassifier, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3))
        self.res_block = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2))
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, num_classes))
        class_counts = torch.bincount(torch.LongTensor(y_encoded))
        self.class_weights = 1. / class_counts.float()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base(x)
        x = x + self.res_block(x)
        return self.classifier(x)

# ===========================================
# Enhanced Training System
# ===========================================
class ScientificTrainer:
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss(weight=model.class_weights)
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1s = []

    def train(self, epochs=200):
        best_f1 = 0
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for inputs, targets in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            self.train_losses.append(epoch_loss/len(self.train_loader))
            self.scheduler.step()
            
            val_metrics = self.evaluate()
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            self.val_f1s.append(val_metrics['f1'])
            
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                torch.save(self.model.state_dict(), 
                         os.path.join(RESULT_PATH, 'best_classifier.pth'))

            print(f'Epoch {epoch+1}/{epochs}')
            print(f"Val Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")
            print(classification_report(val_metrics['true'], val_metrics['pred'], 
                                      target_names=text_labels))

        # Save training metrics
        metrics_df = pd.DataFrame({
            'epoch': range(1, epochs+1),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'val_accuracy': self.val_accuracies,
            'val_f1': self.val_f1s
        })
        metrics_df.to_csv(os.path.join(RESULT_PATH, 'training_metrics.csv'), index=False)
        return metrics_df

    def evaluate(self):
        self.model.eval()
        all_preds, all_true = [], []
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.numpy())
                all_true.extend(targets.numpy())
        
        return {
            'accuracy': accuracy_score(all_true, all_preds),
            'f1': f1_score(all_true, all_preds, average='weighted'),
            'loss': total_loss/len(self.test_loader),
            'true': all_true,
            'pred': all_preds
        }

# ===========================================
# Custom Plotting Functions
# ===========================================
def plot_loss_curve(metrics_df, result_path):
    """Plot training and validation loss curves with consistent styling"""
    # Set global plot parameters
    plt.rcParams["font.family"] = "serif"
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rc('axes', edgecolor='orange', linewidth=2)
    plt.rc('grid', color='lightgray', linestyle='--')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(13, 8))
    
    # Plot loss curves
    ax.plot(metrics_df['epoch'], metrics_df['train_loss'], 
            label='Training Loss', linewidth=3, color='blue')
    ax.plot(metrics_df['epoch'], metrics_df['val_loss'], 
            label='Validation Loss', linewidth=3, linestyle='--', color='red')
    
    # Formatting
    ax.set_title('Model Loss During Training', fontsize=25, pad=20)
    ax.set_ylabel('Loss', fontsize=25)
    ax.set_xlabel('Epoch', fontsize=25)
    ax.legend(fontsize=20, loc='upper right')
    ax.tick_params(axis='both', labelsize=20)
    
    # Grid configuration
    ax.grid(True, alpha=0.3)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth=0.7, color='lightgray')
    ax.grid(which='minor', linestyle=':', linewidth=0.5, color='lightgray')
    
    # Save plot
    plot_path = os.path.join(result_path, 'loss_curve.jpg')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Loss curve plot saved to: {plot_path}")
    return fig

def plot_confusion_matrix(true, pred, class_names, result_path):
    """Plot confusion matrix with consistent styling"""
    # Set global plot parameters
    plt.rcParams["font.family"] = "serif"
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'olive'
    plt.rcParams['axes.linewidth'] = 3
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    
    # Calculate confusion matrix
    cm = confusion_matrix(true, pred)
    
    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list('brg', ['#0000FF', '#00FFFF', '#FFFF00', '#FF0000'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Count Scale', rotation=270, labelpad=25, fontsize=20)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=20, fontweight='bold')
    
    # Set labels
    ax.set_title('Confusion Matrix', fontsize=24, pad=20)
    ax.set_xlabel('Predicted Label', fontsize=22, labelpad=15)
    ax.set_ylabel('True Label', fontsize=22, labelpad=15)
    
    # Set tick labels
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    # Add grid
    ax.grid(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(result_path, 'confusion_matrix.jpg')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix plot saved to: {plot_path}")
    return fig

def plot_metrics(metrics_df, result_path):
    """Plot accuracy and F1 score during training"""
    # Set global plot parameters
    plt.rcParams["font.family"] = "serif"
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rc('axes', edgecolor='purple', linewidth=2)
    plt.rc('grid', color='lightgray', linestyle='--')
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot accuracy
    color = 'tab:blue'
    ax1.set_xlabel('Epoch', fontsize=18)
    ax1.set_ylabel('Accuracy', color=color, fontsize=18)
    ax1.plot(metrics_df['epoch'], metrics_df['val_accuracy'], 
             color=color, linewidth=2.5, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Create second axis for F1 score
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('F1 Score', color=color, fontsize=18)
    ax2.plot(metrics_df['epoch'], metrics_df['val_f1'], 
             color=color, linewidth=2.5, linestyle='--', label='F1 Score')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add title
    plt.title('Validation Metrics During Training', fontsize=22, pad=15)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=14)
    
    # Add grid
    ax1.grid(True, which='major', alpha=0.3)
    ax1.grid(True, which='minor', alpha=0.15)
    ax1.minorticks_on()
    
    # Adjust layout
    fig.tight_layout()
    
    # Save plot
    plot_path = os.path.join(result_path, 'validation_metrics.jpg')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Validation metrics plot saved to: {plot_path}")
    return fig

# ===========================================
# Main Execution
# ===========================================
if __name__ == "__main__":
    # Set up data loaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model and trainer
    model = ImbalancedClassifier()
    trainer = ScientificTrainer(model, train_loader, test_loader)
    
    # Train model
    print("Starting model training...")
    metrics_df = trainer.train(epochs=200)
    print("Training completed.")
    
    # Plot loss curve
    print("\nPlotting loss curve...")
    plot_loss_curve(metrics_df, RESULT_PATH)
    
    # Plot validation metrics
    print("\nPlotting validation metrics...")
    plot_metrics(metrics_df, RESULT_PATH)
    
    # Final evaluation
    print("\nRunning final evaluation...")
    test_metrics = trainer.evaluate()
    
    print('\nTest Set Performance:')
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    
    # Save test results
    test_results = pd.DataFrame({
        'true': test_metrics['true'],
        'pred': test_metrics['pred']
    })
    test_results.to_csv(os.path.join(RESULT_PATH, 'test_results.csv'), index=False)
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(test_metrics['true'], test_metrics['pred'], 
                         text_labels, RESULT_PATH)
    
    print("\nAll operations completed successfully!")
