# System imports
import os
from tqdm import tqdm
import time

# Machine Learning imports
import torch
import torch.nn as nn
import torch.optim as optim 

# Processing imports
import numpy as np
import pandas as pd

# Evaluation imports
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, precision_score, recall_score, 
    f1_score, accuracy_score, roc_curve, auc, precision_recall_curve, log_loss
)

def list_files(path):
    """Return a list of .wav file paths under the given directory."""
    objects = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(".wav"):
                objects.append(os.path.join(root, file))
    return objects

def create_dataframe(keys):
    """Creates a pandas dataframe with file path as index, and 0-1 features 'Background' and 'Gunshot'"""
    return pd.DataFrame(index=keys, data={"label": [1 if "Gunshot" in key else 0 for key in keys]})

def train_model(model, train_loader, val_loader, output_path, output_path2, num_epochs, lr):
    torch.set_float32_matmul_precision('high')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
    model = model.to(device)
    model = torch.compile(model)
    best_f1 = 0.0
    for epoch in range(num_epochs):
        print(f"\n🚀 Starting Epoch [{epoch+1}/{num_epochs}]")
        # Training part
        model.train()
        train_loss = 0.0   
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
            images, labels = images.to(device), labels.to(device).to(torch.int64).squeeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation part
        model.eval()
        all_labels = []
        all_preds = []
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
                images, labels = images.to(device), labels.to(device).to(torch.int64).squeeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        score = f1_score(np.array(all_labels), np.array(all_preds))
        
        print(f"📊 Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | F1 Score: {score:.4f}")

        # Adjust learning rate if needed and save model if F1 improves
        scheduler.step(score)
        if score > best_f1:
            best_f1 = score
            print(f"✅ New Best Model Found! Saving at {output_path}")
            torch.save(model, output_path)

    torch.save(model, output_path2)
    print(f"\n🎯 Training Complete! Best Model saved with F1 Score: {best_f1:.4f}")
    return model



def evaluate_model(path, val_loader, threshold=0.5):
    """Evaluates a trained PyTorch model and saves metrics & plots to a local directory."""
    model = torch.load(path,map_location=device,weights_only=False)
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device).to(torch.int64).squeeze(1)
            
            outputs = model(images)
            # Compute probabilities using softmax
            probabilities = torch.softmax(outputs, dim=1)
            # Get predicted class
            _, preds = torch.max(outputs.data, 1)
            
            # Make sure to move tensors to CPU before converting to numpy
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probabilities[:, 1].cpu().numpy().tolist())
    
    # Convert lists to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    # Compute evaluation metrics (additional code for ROC, PR curves omitted)
    conf_matrix = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Save classification report to a text file
    timestamp = time.strftime("%Y%m%d-%H%M")
    eval_dir = f"evaluation/Run_{timestamp}"
    os.makedirs(eval_dir, exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=["Background", "Gunshot"])
    metrics_text = f"""
=== Classification Report ===
{report}

=== Performance Metrics ===
Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1 Score: {f1:.4f}
"""
    
    with open(os.path.join(eval_dir, "metrics.txt"), "w") as f:
        f.write(metrics_text)
    print(metrics_text)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Background", "Gunshot"],
                yticklabels=["Background", "Gunshot"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(eval_dir, "confusion_matrix.png"))
    plt.close()
    print(f"✅ Evaluation complete. Results saved in: {eval_dir}")