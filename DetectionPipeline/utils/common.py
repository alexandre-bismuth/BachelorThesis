# System imports
import os
import datetime
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
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, \
                            f1_score, accuracy_score, roc_curve, auc, precision_recall_curve

## ENVIRONMENT SETUP ##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")
#######################

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
    global device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5)
    model = model.to(device)
    model = torch.compile(model)
    best_f1 = 0.0
    for epoch in range(num_epochs):
        print(f"\n🚀 Starting Epoch [{epoch+1}/{num_epochs}]")
        # Training part
        model.train()
        train_loss = 0.0   
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
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
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
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

def evaluate_model(path, val_loader, threshold=0.5, optimized_f1=False, delta_fp_fn=False):
    """
    Evaluates the trained Pytorch model and saves metrics to a local directory.
    Note here that we are evaluating our model on the validation set, which is not
    ideal since the validation data is used to select the best model, but is necessary
    given that we have very few positive samples.
    """
    global device
    model = torch.load(path, map_location=device, weights_only=False)
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    if delta_fp_fn:
        all_file_paths = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            if len(batch) == 3:
                images, labels, file_paths = batch
            else:
                images, labels = batch
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

            if delta_fp_fn:
                assert len(batch) == 3, "The dataset your loader uses is not suited for delta analysis"
                all_file_paths.extend(file_paths)
    
    # Convert lists to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    # Compute evaluation metrics
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
    
    # Plot and save the confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Background", "Gunshot"],
                yticklabels=["Background", "Gunshot"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(eval_dir, "confusion_matrix.png"))
    plt.close()
    
    # --- ROC Curve ---
    # We plot it for comparison purposes but will not use it as a metric 
    # in our report since it is not well-suited for imabalanced datasets 
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(eval_dir, "roc_curve.png"))
    plt.close()
    
    # --- Precision-Recall Curve ---
    precision_vals, recall_vals, threshold_vals = precision_recall_curve(y_true, y_prob)
    auprc = auc(recall_vals, precision_vals)

    if optimized_f1:
        f1_vals = 2*precision_vals*recall_vals/(precision_vals+recall_vals)
        max_f1, max_f1_index = np.max(f1_vals), np.argmax(f1_vals)
        max_precision, max_recall = precision_vals[max_f1_index], recall_vals[max_f1_index]
        best_threshold = threshold_vals[max_f1_index]
        metrics_text += f"""
=== Performance Metrics - Optimised Threshold ===
Threshold: {best_threshold:.4f}
Precision: {max_precision:.4f}
Recall: {max_recall:.4f}
F1 Score: {max_f1:.4f}
"""
    if delta_fp_fn:
        false_positive_indices = np.where((y_true == 0) & (y_pred == 1))[0]
        false_positive_file_paths = [all_file_paths[i] for i in false_positive_indices]
        
        # Sort the file paths in chronological order based on the first 8 characters (hex timestamp)
        false_positive_file_paths = sorted(false_positive_file_paths, key=lambda fp: int(os.path.basename(fp)[:8], 16))
        timestamps_fp = [int(os.path.basename(fp)[:8], 16) for fp in false_positive_file_paths]

        assert len(timestamps_fp) >= 2, "There are not enough false positives to provide the mean time between false positives, please set the mean_time_fp parameter to False"
        
        time_diffs_fp = np.diff(timestamps_fp)
        mean_time_fp = np.mean(time_diffs_fp)
        median_time_fp = np.median(time_diffs_fp)
        max_time_fp, min_time_fp = np.max(time_diffs_fp), np.min(time_diffs_fp)

        metrics_text += f"""
=== False Positive analysis ===
Number of False Positives : {len(false_positive_file_paths)}
Mean Time Between False Positives: {mean_time_fp:.0f} seconds ({str(datetime.timedelta(seconds=int(mean_time_fp)))})
Median Time Between False Positives: {median_time_fp:.0f} seconds ({str(datetime.timedelta(seconds=int(median_time_fp)))})
Maximum Time Between False Positives: {max_time_fp:.0f} seconds ({str(datetime.timedelta(seconds=int(max_time_fp)))})
Minimum Time Between False Positives: {min_time_fp:.0f} seconds ({str(datetime.timedelta(seconds=int(min_time_fp)))})
"""
        false_negative_indices = np.where((y_true == 1) & (y_pred == 0))[0]
        false_negative_file_paths = [all_file_paths[i] for i in false_negative_indices]
        
        # Sort the file paths in chronological order based on the first 8 characters (hex timestamp)
        false_negative_file_paths = sorted(false_negative_file_paths, key=lambda fn: int(os.path.basename(fn)[:8], 16))
        timestamps_fn = [int(os.path.basename(fn)[:8], 16) for fn in false_negative_file_paths]
        
        assert len(timestamps_fn) >= 2, "There are not enough false negatives to provide the mean time between false negatives, please set the mean_time_fn parameter to False"
        
        time_diffs_fn = np.diff(timestamps_fn)
        mean_time_fn = np.mean(time_diffs_fn)
        median_time_fn = np.median(time_diffs_fn)
        max_time_fn, min_time_fn = np.max(time_diffs_fn), np.min(time_diffs_fn)

        metrics_text += f"""
=== False Negative analysis ===
Number of False Negatives : {len(false_negative_file_paths)}
Mean Time Between False Negatives: {mean_time_fn:.0f} seconds ({str(datetime.timedelta(seconds=int(mean_time_fn)))})
Median Time Between False Negatives: {median_time_fn:.0f} seconds ({str(datetime.timedelta(seconds=int(median_time_fn)))})
Maximum Time Between False Negatives: {max_time_fn:.0f} seconds ({str(datetime.timedelta(seconds=int(max_time_fn)))})
Minimum Time Between False Negatives: {min_time_fn:.0f} seconds ({str(datetime.timedelta(seconds=int(min_time_fn)))})
"""
    
    plt.figure(figsize=(6, 5))
    plt.plot(recall_vals, precision_vals, label=f"PR curve (area = {auprc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(eval_dir, "precision_recall_curve.png"))
    plt.close()

    with open(os.path.join(eval_dir, "metrics.txt"), "w") as f:
        f.write(metrics_text)
    print(metrics_text)

    
    print(f"✅ Evaluation complete. Results saved in: {eval_dir}")
