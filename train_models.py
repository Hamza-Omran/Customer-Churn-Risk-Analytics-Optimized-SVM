import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import joblib
import time
import sys
import os

sys.path.append(os.path.dirname(__file__))
from src.models.gradient_descent_svm import GradientDescentSVM
from src.models.subgradient_descent_svm import SubgradientDescentSVM

print("Loading and preprocessing data...")

df = pd.read_csv('data/raw/churn_prediction.csv')

print(f"Original shape: {df.shape}")

df = df.dropna()
df = df[df['age'] > 0]
df = df[df['age'] < 100]

print(f"After cleaning: {df.shape}")

if 'vintage' in df.columns and 'age' in df.columns:
    df['tenure_ratio'] = df['vintage'] / df['age']
    print("Created tenure_ratio feature")

transaction_cols = ['current_month_credit', 'previous_month_credit', 
                   'current_month_debit', 'previous_month_debit']
if all(col in df.columns for col in transaction_cols):
    df['activity_score'] = df[transaction_cols].sum(axis=1) / (df.get('vintage', 1) + 1)
    print("Created activity_score feature")

label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

for col in categorical_cols:
    if col != 'churn':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

if 'customer_id' in df.columns:
    df = df.drop('customer_id', axis=1)

X = df.drop('churn', axis=1)
y = df['churn']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

feature_names = X.columns.tolist()

joblib.dump(scaler, 'deployment/models/scaler.pkl')
print("Saved scaler")

results = {}

print("\n" + "="*50)
print("Training sklearn SVM...")
print("="*50)

param_grid = {
    'C': [1, 10, 100],
    'gamma': ['scale', 'auto'],
    'class_weight': ['balanced', None]
}

start_time = time.time()
svm_sklearn = SVC(kernel='rbf', random_state=42, probability=True)
grid_search = GridSearchCV(svm_sklearn, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
sklearn_time = time.time() - start_time

best_svm = grid_search.best_estimator_
train_acc_sklearn = accuracy_score(y_train, best_svm.predict(X_train_scaled))
val_acc_sklearn = accuracy_score(y_val, best_svm.predict(X_val_scaled))
test_acc_sklearn = accuracy_score(y_test, best_svm.predict(X_test_scaled))

results['sklearn'] = {
    'model': best_svm,
    'train_acc': train_acc_sklearn,
    'val_acc': val_acc_sklearn,
    'test_acc': test_acc_sklearn,
    'time': sklearn_time,
    'params': grid_search.best_params_
}

joblib.dump(best_svm, 'deployment/models/sklearn_svm.pkl')

print(f"Best params: {grid_search.best_params_}")
print(f"Train Acc: {train_acc_sklearn:.4f}")
print(f"Val Acc: {val_acc_sklearn:.4f}")
print(f"Test Acc: {test_acc_sklearn:.4f}")
print(f"Training time: {sklearn_time:.2f}s")

print("\n" + "="*50)
print("Training Gradient Descent SVM...")
print("="*50)

start_time = time.time()
gd_svm = GradientDescentSVM(learning_rate=0.001, lambda_param=0.01, n_iterations=1000)
gd_svm.fit(X_train_scaled, y_train.values)
gd_time = time.time() - start_time

train_acc_gd = gd_svm.score(X_train_scaled, y_train.values)
val_acc_gd = gd_svm.score(X_val_scaled, y_val.values)
test_acc_gd = gd_svm.score(X_test_scaled, y_test.values)

results['gd'] = {
    'model': gd_svm,
    'train_acc': train_acc_gd,
    'val_acc': val_acc_gd,
    'test_acc': test_acc_gd,
    'time': gd_time,
    'losses': gd_svm.losses
}

joblib.dump(gd_svm, 'deployment/models/gd_svm.pkl')

print(f"Train Acc: {train_acc_gd:.4f}")
print(f"Val Acc: {val_acc_gd:.4f}")
print(f"Test Acc: {test_acc_gd:.4f}")
print(f"Training time: {gd_time:.2f}s")

print("\n" + "="*50)
print("Training Subgradient Descent SVM...")
print("="*50)

start_time = time.time()
subgd_svm = SubgradientDescentSVM(learning_rate=0.001, lambda_param=0.0005, 
                                   n_iterations=1000, batch_size=32)
subgd_svm.fit(X_train_scaled, y_train.values)
subgd_time = time.time() - start_time

train_acc_subgd = subgd_svm.score(X_train_scaled, y_train.values)
val_acc_subgd = subgd_svm.score(X_val_scaled, y_val.values)
test_acc_subgd = subgd_svm.score(X_test_scaled, y_test.values)

results['subgd'] = {
    'model': subgd_svm,
    'train_acc': train_acc_subgd,
    'val_acc': val_acc_subgd,
    'test_acc': test_acc_subgd,
    'time': subgd_time,
    'losses': subgd_svm.losses
}

joblib.dump(subgd_svm, 'deployment/models/subgd_svm.pkl')

print(f"Train Acc: {train_acc_subgd:.4f}")
print(f"Val Acc: {val_acc_subgd:.4f}")
print(f"Test Acc: {test_acc_subgd:.4f}")
print(f"Training time: {subgd_time:.2f}s")

print("\n" + "="*50)
print("Creating Visualizations...")
print("="*50)

os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)
os.makedirs('results/comparisons', exist_ok=True)

comparison_df = pd.DataFrame({
    'Model': ['sklearn SVM', 'Gradient Descent', 'Subgradient Descent'],
    'Train Accuracy': [train_acc_sklearn, train_acc_gd, train_acc_subgd],
    'Validation Accuracy': [val_acc_sklearn, val_acc_gd, val_acc_subgd],
    'Test Accuracy': [test_acc_sklearn, test_acc_gd, test_acc_subgd],
    'Training Time (s)': [sklearn_time, gd_time, subgd_time]
})

comparison_df.to_csv('results/comparisons/model_comparison.csv', index=False)
print(comparison_df)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(comparison_df))
width = 0.25

axes[0].bar(x - width, comparison_df['Train Accuracy'], width, label='Train', alpha=0.8)
axes[0].bar(x, comparison_df['Validation Accuracy'], width, label='Validation', alpha=0.8)
axes[0].bar(x + width, comparison_df['Test Accuracy'], width, label='Test', alpha=0.8)
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Accuracy Comparison')
axes[0].set_xticks(x)
axes[0].set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
axes[0].legend()
axes[0].set_ylim([0.85, 1.0])
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(comparison_df['Model'], comparison_df['Training Time (s)'], alpha=0.8, color='coral')
axes[1].set_ylabel('Time (seconds)')
axes[1].set_title('Training Time Comparison')
axes[1].set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/model_comparison.png")

plt.figure(figsize=(10, 6))
plt.plot(results['gd']['losses'], label='Gradient Descent', linewidth=2)
plt.plot(range(0, len(results['subgd']['losses'])*10, 10), results['subgd']['losses'], 
         label='Subgradient Descent', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Convergence')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/convergence.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/convergence.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, pred) in enumerate([
    ('sklearn SVM', best_svm.predict(X_test_scaled)),
    ('Gradient Descent', gd_svm.predict(X_test_scaled)),
    ('Subgradient Descent', subgd_svm.predict(X_test_scaled))
]):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(f'{name}')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('results/figures/confusion_matrices.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/confusion_matrices.png")

if hasattr(best_svm, 'predict_proba'):
    fpr_sklearn, tpr_sklearn, _ = roc_curve(y_test, best_svm.predict_proba(X_test_scaled)[:, 1])
    roc_auc_sklearn = auc(fpr_sklearn, tpr_sklearn)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_sklearn, tpr_sklearn, label=f'sklearn SVM (AUC = {roc_auc_sklearn:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/roc_curve.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/roc_curve.png")

print("\n" + "="*50)
print("Training Complete!")
print("="*50)
print(f"\nAll models saved to deployment/models/")
print(f"Visualizations saved to results/figures/")
print(f"Metrics saved to results/comparisons/")
