import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

print("="*70)
print("LOAN APPROVAL PREDICTION - CHECKPOINT 1")
print("="*70)

# Load and explore data
df = pd.read_csv('data.csv')
print(f"\nDataset: {df.shape[0]:,} samples × {df.shape[1]} features")
print(f"Target Distribution:\n{df['loan_status'].value_counts()}")
print(f"Proportion:\n{df['loan_status'].value_counts(normalize=True)}")

numerical_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
                  'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['loan_status'].value_counts().plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Loan Status Distribution', fontweight='bold')
df['loan_status'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
axes[1].set_ylabel('')
plt.tight_layout()
plt.savefig('01_target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
axes = axes.ravel()
for idx, col in enumerate(numerical_cols):
    axes[idx].hist(df[col], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[idx].set_title(col, fontweight='bold')
plt.tight_layout()
plt.savefig('02_numerical_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.ravel()
for idx, col in enumerate(categorical_cols):
    df[col].value_counts().plot(kind='bar', ax=axes[idx], color='coral')
    axes[idx].set_title(col, fontweight='bold')
    axes[idx].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('03_categorical_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 8))
correlation_matrix = df[numerical_cols + ['loan_status']].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig('04_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.ravel()
for idx, col in enumerate(categorical_cols):
    pd.crosstab(df[col], df['loan_status'], normalize='index').plot(kind='bar', ax=axes[idx], 
                                                                      color=['#2ecc71', '#e74c3c'])
    axes[idx].set_title(f'Loan Status by {col}', fontweight='bold')
    axes[idx].legend(['Approved', 'Default'])
    axes[idx].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('05_status_by_categories.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
axes = axes.ravel()
for idx, col in enumerate(numerical_cols):
    df.boxplot(column=col, by='loan_status', ax=axes[idx])
    axes[idx].set_title(col, fontweight='bold')
plt.suptitle('')
plt.tight_layout()
plt.savefig('06_boxplots_by_status.png', dpi=300, bbox_inches='tight')
plt.close()

# Preprocessing
df_processed = df.drop('id', axis=1) if 'id' in df.columns else df.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col].astype(str))

df_processed['income_to_loan_ratio'] = df_processed['person_income'] / (df_processed['loan_amnt'] + 1)
df_processed['age_employment_ratio'] = df_processed['person_age'] / (df_processed['person_emp_length'] + 1)

X = df_processed.drop('loan_status', axis=1)
y = df_processed['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# Model training and evaluation
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
}

results = {}
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred_test = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    results[model_name] = {
        'model': model,
        'test_acc': accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test),
        'recall': recall_score(y_test, y_pred_test),
        'f1': f1_score(y_test, y_pred_test),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'y_pred': y_pred_test
    }

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['test_acc'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1'] for m in results.keys()],
    'ROC-AUC': [results[m]['roc_auc'] for m in results.keys()]
})

print("\n" + "="*70)
print("MODEL PERFORMANCE COMPARISON")
print("="*70)
print(comparison_df.to_string(index=False))

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
colors = ['#3498db', '#e74c3c', '#2ecc71']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]
    bars = ax.bar(comparison_df['Model'], comparison_df[metric], color=colors)
    ax.set_title(metric, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', 
                ha='center', va='bottom', fontsize=9)

axes[1, 2].axis('off')
plt.tight_layout()
plt.savefig('07_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, (model_name, result) in enumerate(results.items()):
    cm = confusion_matrix(y_test, result['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Approved', 'Default'],
                yticklabels=['Approved', 'Default'])
    axes[idx].set_title(model_name, fontweight='bold')
    axes[idx].set_ylabel('True')
    axes[idx].set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('08_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n" + "="*70)
print("TOP 10 FEATURE IMPORTANCE")
print("="*70)
print(feature_importance.head(10).to_string(index=False))

plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance', fontweight='bold')
plt.title('Feature Importance - Random Forest', fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('09_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*70)
print("CLASSIFICATION REPORTS")
print("="*70)
for model_name, result in results.items():
    print(f"\n{model_name}:")
    print(classification_report(y_test, result['y_pred'], target_names=['Approved', 'Default']))

print("="*70)
print("CHECKPOINT 1 COMPLETE - 9 visualizations generated")
print("="*70)