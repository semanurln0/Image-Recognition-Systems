import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import joblib
import random
import time
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc)

wildfire_step_counter = 1

def print_wildfire_step(description):
    global wildfire_step_counter
    print(f"> Wildfire Step {wildfire_step_counter}/10: {description}")
    wildfire_step_counter += 1

def print_progress_bar(current, total, label="Progress"):
    bar_length = 40
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    percent = (current / total) * 100
    print(f"\r{label} |{bar}| {percent:.0f}%%", end='\r')
    if current == total:
        print()

# --- Setup: Clean folders ---
for folder in ['plots_wildfire', 'training-models_wildfire', 'tables_wildfire']:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

print_wildfire_step("Folders 'plots_wildfire', 'training-models_wildfire', and 'tables_wildfire' are prepared.")

# --- Manual unzip expected ---
print_wildfire_step("Assuming wildfire dataset is manually unzipped.")

# --- Load Wildfire Dataset (with 40% balanced sampling) ---
print_wildfire_step("Loading wildfire dataset with user-defined sampling ratio...")

def load_wildfire_dataset(base_path):
    sampling_ratio = float(input("Enter sampling ratio (e.g., 0.4 for 40%): "))
    def load_split(split):
        split_path = os.path.join(base_path, split)
        x_split, y_split = [], []
        for label in ['wildfire', 'nowildfire']:
            class_path = os.path.join(split_path, label)
            if not os.path.exists(class_path):
                print(f"WARNING: Path not found {class_path}")
                continue
            image_files = sorted(os.listdir(class_path))
            total = len(image_files)
            if total == 0:
                print(f"WARNING: No images in {class_path}")
                continue
            
            sample_size = int(total * sampling_ratio)
            sampled_files = random.sample(image_files, sample_size)
            for idx, img_name in enumerate(sampled_files):
                print_progress_bar(idx + 1, sample_size, label=label.upper())
                img_path = os.path.join(class_path, img_name)
                try:
                    img = Image.open(str(img_path)).convert('RGB')
                    img = img.resize((64, 64))
                    img_array = np.array(img) / 255.0
                    x_split.append(img_array)
                    y_split.append(label)
                except Exception as e:
                    print(f"Error reading {img_path}: {e}")
                    continue
        return np.array(x_split), np.array(y_split)

    x_train_data, y_train_data = load_split('train')
    x_valid_data, y_valid_data = load_split('valid')
    x_test_data, y_test_data = load_split('test')

    return x_train_data, y_train_data, x_valid_data, y_valid_data, x_test_data, y_test_data

wildfire_base_path = "wildfire_dataset"
x_train_w, y_train_w, x_valid_w, y_valid_w, x_test_w, y_test_w = load_wildfire_dataset(wildfire_base_path)

x_all = np.concatenate((x_train_w, x_valid_w))
y_all = np.concatenate((y_train_w, y_valid_w))

print_wildfire_step("Wildfire dataset loaded and combined.")

# --- Preprocess ---
print_wildfire_step("Preprocessing wildfire data with scaling and PCA...")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_all)
x_flat = x_all.reshape(len(x_all), -1)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_flat)

pca = PCA(n_components=100)
x_pca = pca.fit_transform(x_scaled)

joblib.dump(scaler, os.path.join('training-models_wildfire', 'scaler.pkl'))
joblib.dump(pca, os.path.join('training-models_wildfire', 'pca.pkl'))
joblib.dump(label_encoder, os.path.join('training-models_wildfire', 'label_encoder.pkl'))
joblib.dump(x_flat, os.path.join('training-models_wildfire', 'x_all_flat.pkl'))
joblib.dump(y_encoded, os.path.join('training-models_wildfire', 'y_all_encoded.pkl'))

x_train_final, x_test_final, y_train_final, y_test_final = train_test_split(
    x_pca, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# --- Model Training ---
print_wildfire_step("Training models for wildfire classification...")

base_svm = LinearSVC(dual=False, max_iter=2000, tol=1e-4)
models = {
    "SVM": CalibratedClassifierCV(base_svm, cv=5),
    "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
timing_summary = []

for name, model in models.items():
    print(f"> Training {name} model (wildfire)...")
    scores = []
    start_time = time.time()
    
    for tr_idx, val_idx in kf.split(x_train_final):
        x_fold_train, x_fold_val = x_train_final[tr_idx], x_train_final[val_idx]
        y_fold_train, y_fold_val = y_train_final[tr_idx], y_train_final[val_idx]
        model.fit(x_fold_train, y_fold_train)
        y_pred_fold = model.predict(x_fold_val)
        scores.append(accuracy_score(y_fold_val, y_pred_fold))

    cv_accuracy = np.mean(scores)

    model.fit(x_train_final, y_train_final)
    training_time = time.time() - start_time
    timing_summary.append([name, training_time])
    
    joblib.dump(model, os.path.join('training-models_wildfire', f'{name}_model.pkl'))

    # --- Save Sample Predictions (original image space) ---
    
    sample_indices = np.random.choice(len(x_test_w), size=5, replace=False)
    img_test = x_test_w[sample_indices]
    y_test_labels = y_test_w[sample_indices]
    flat_test = x_test_w.reshape(len(x_test_w), -1)
    x_test_transformed = pca.transform(scaler.transform(flat_test[sample_indices]))
    y_pred_labels = label_encoder.inverse_transform(model.predict(x_test_transformed))

    plt.figure(figsize=(15, 5))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(img_test[i])
        plt.title(f"P:{y_pred_labels[i]}\nT:{y_test_labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'plots_wildfire/{name}_sample_predictions.png')
    plt.close()


    y_pred = model.predict(x_test_final)

    acc = accuracy_score(y_test_final, y_pred)
    prec = precision_score(y_test_final, y_pred, average='macro')
    rec = recall_score(y_test_final, y_pred, average='macro')
    f1 = f1_score(y_test_final, y_pred, average='macro')

    cm = confusion_matrix(y_test_final, y_pred)
    specificities = []
    for i in range(len(label_encoder.classes_)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp)
        specificities.append(specificity)
    avg_specificity = np.mean(specificities)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'{name} - Wildfire Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join('plots_wildfire', f'{name}_confusion_matrix.png'))
    plt.close()

    metrics_df = pd.DataFrame({
        'Metric': ['Cross-Validation Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity'],
        'Score': [cv_accuracy, acc, prec, rec, f1, avg_specificity]
    })
    metrics_df.to_csv(os.path.join('tables_wildfire', f'{name}_metrics.csv'), index=False)

# Save comparison tables as PNG
print_wildfire_step("Generating comparison tables as images...")

metrics_summary = []
for name in models.keys():
    csv_path = os.path.join('tables_wildfire', f'{name}_metrics.csv')
    metrics_df = pd.read_csv(csv_path)
    metrics_summary.append([name] + list(metrics_df['Score']))

metrics_df = pd.DataFrame(metrics_summary, 
                         columns=['Model', 'Cross-Validation Accuracy', 'Test Accuracy', 
                                 'Precision', 'Recall', 'F1 Score', 'Specificity'])
metrics_df.to_csv('tables_wildfire/metrics_summary.csv', index=False)

timing_df = pd.DataFrame(timing_summary, columns=['Model', 'Training Time (s)'])

def plot_table(df, title, filename):
    plt.figure(figsize=(10, len(df)*0.8))
    ax = plt.gca()
    ax.axis('off')
    tbl = ax.table(cellText=df.round(4).values, colLabels=df.columns, 
                  loc='center', cellLoc='center')
    tbl.scale(1, 1.5)
    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_table(metrics_df, "Wildfire Classifier - Model Performance", 
          "plots_wildfire/metrics_comparison.png")
plot_table(timing_df, "Wildfire Classifier - Training Time Comparison", 
          "plots_wildfire/training_time_comparison.png")

print_wildfire_step("Finished wildfire model training and evaluation.")

# --- ROC Curve with Logistic Regression ---
print_wildfire_step("Calculating and plotting wildfire ROC curve with Logistic Regression...")

# Prepare data
x_train2_flat = x_train_w.reshape(len(x_train_w), -1)
x_test2_flat = x_test_w.reshape(len(x_test_w), -1)

# Scale features
scaler2 = StandardScaler()
x_train2_scaled = scaler2.fit_transform(x_train2_flat)
x_test2_scaled = scaler2.transform(x_test2_flat)

# Apply PCA
pca2 = PCA(n_components=100)
x_train2_pca = pca2.fit_transform(x_train2_scaled)
x_test2_pca = pca2.transform(x_test2_scaled)

# Train logistic regression
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(x_train2_pca, y_train_w)

# Get probabilities for ROC curve
y_score = model_lr.predict_proba(x_test2_pca)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

# Binarize the labels for ROC curve calculation
y_test_bin = label_binarize(y_test_w, classes=label_encoder.classes_)

# Calculate ROC curve and ROC area for both classes
for i in range(len(label_encoder.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(10, 8))
colors = ['red', 'blue']
for i, (color, class_name) in enumerate(zip(colors, label_encoder.classes_)):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'{class_name} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Wildfire ROC Curve (Logistic Regression)')
plt.legend(loc='lower right')
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join('plots_wildfire', 'roc_curve.png'))
plt.close()

# --- Metrics & Time Comparison Plots ---
print_wildfire_step("Generating metrics and training time comparison plots...")

all_metrics = []
all_times = []

for model_name in ['SVM', 'Random_Forest', 'KNN']:
    metrics_csv = os.path.join('tables_wildfire', f'{model_name}_metrics.csv')
    if os.path.exists(metrics_csv):
        df = pd.read_csv(metrics_csv)
        df['Model'] = model_name
        all_metrics.append(df)

metrics_df = pd.concat(all_metrics, ignore_index=True)
timing_df = pd.DataFrame(timing_summary, columns=['Model', 'Training Time (s)'])

plt.figure(figsize=(10, 6))
sns.barplot(data=metrics_df, x='Metric', y='Score', hue='Model')
plt.title('Metrics Comparison Across Models')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join('plots_wildfire', 'metrics_comparison.png'))
plt.close()

plt.figure(figsize=(8, 6))
sns.barplot(data=timing_df, x='Model', y='Training Time (s)')
plt.title('Training Time Comparison')
plt.ylabel('Seconds')
plt.tight_layout()
plt.savefig(os.path.join('plots_wildfire', 'training_time_comparison.png'))
plt.close()

print_wildfire_step("Wildfire classification task fully completed. All outputs saved.")
