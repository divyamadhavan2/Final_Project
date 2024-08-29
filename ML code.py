#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("E:/Divya/SCCT/code/Data/Student_performance_data.csv")

# Check for and handle missing or infinite values
# Fill NaN values with the median
df = df.fillna(df.median())

# Replace infinite values with NaN and then fill with the median
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.fillna(df.median())

# Numerical and categorical features defined
selected_categorical_features = ['ParentalSupport', 'Extracurricular', 'Music', 'ParentalEducation', 'Sports','Tutoring']
numerical_features = ['StudyTimeWeekly', 'Absences','GPA']

# Define the target variable as GradeClass
target = 'GradeClass'

# Combine selected numerical features with selected categorical features
X_combined = pd.concat([df[numerical_features], df[selected_categorical_features].reset_index(drop=True)], axis=1)

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df[target])

# Create and fit a preprocessor for categorical and numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), selected_categorical_features)
    ]
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_encoded, test_size=0.2, random_state=42)

# Define pipelines for each model
pipelines = {
    'decision_tree': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ]),
    'random_forest': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ]),
    'xgboost': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(random_state=42, eval_metric='mlogloss'))
    ]),
    'logistic_regression': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ]),
    'svm': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(random_state=42, probability=True))
    ])
}

# Define parameter grids for each model
param_grids = {
    'decision_tree': {
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    'random_forest': {
        'classifier__n_estimators': [50, 100, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__class_weight': ['balanced', 'balanced_subsample', None]
    },
    'xgboost': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 6, 10]
    },
    'logistic_regression': {
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__penalty': ['l2']
    },
    'svm': {
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__kernel': ['linear', 'rbf']
    }
}

# Function to train and evaluate models
def train_and_evaluate_model(model_name):
    print(f"Training {model_name}...")
    pipeline = pipelines[model_name]
    param_grid = param_grids[model_name]
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best Cross-validation Score for {model_name}: {grid_search.best_score_:.4f}")
    
    # Predict on the test set
    y_pred = best_model.predict(X_test)
    
    # Evaluate the model
    print(f"{model_name.capitalize()} Metrics:")
    print(f'Training Accuracy: {accuracy_score(y_train, best_model.predict(X_train)):.4f}')
    print(f'Test Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(f'Precision: {precision_score(y_test, y_pred, average="weighted", zero_division=0):.4f}')
    print(f'Recall: {recall_score(y_test, y_pred, average="weighted"):.4f}')
    print(f'F1 Score: {f1_score(y_test, y_pred, average="weighted", zero_division=0):.4f}')
    print(f'ROC-AUC: {roc_auc_score(y_test, best_model.predict_proba(X_test), multi_class="ovr"):.4f}')
    
    # Print the classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {model_name.capitalize()}')
    plt.show()
    print("\n")
    
    return best_model

# Train and evaluate all models
best_models = {}
for model_name in pipelines.keys():
    best_models[model_name] = train_and_evaluate_model(model_name)


# In[4]:


import matplotlib.pyplot as plt
import numpy as np

# Fit the preprocessor first
preprocessor.fit(X_train)

# Generate feature names after preprocessing
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(selected_categorical_features)
num_feature_names = numerical_features
feature_names = np.concatenate([num_feature_names, cat_feature_names])

# Function to plot feature importances for tree-based models
def plot_feature_importance(model, feature_names, model_name):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(8, 5))  # Reduced figure size
    plt.title(f"Feature Importances for {model_name}", fontsize=12)
    bars = plt.bar(range(len(indices)), importances[indices], color="r", align="center", width=0.6)  # Reduced bar width
    plt.xticks(range(len(indices)), np.array(feature_names)[indices], rotation=90, fontsize=8)  # Smaller font size
    plt.xlim([-1, len(indices)])
    plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels

    # Adding percentage values on top of bars
    for bar, importance in zip(bars, importances[indices]):
        percentage = f'{importance * 100:.2f}%'
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), percentage,
                 ha='center', va='bottom', fontsize=8)  # Smaller text for values

    plt.show()

# Function to plot coefficients for models like Logistic Regression and SVM
def plot_coefficients(model, feature_names, model_name):
    coefficients = model.coef_.flatten()
    indices = np.argsort(np.abs(coefficients))[::-1]
    
    plt.figure(figsize=(8, 4))  # Reduced figure size
    plt.title(f"Feature Coefficients for {model_name}", fontsize=12)
    bars = plt.bar(range(len(indices)), coefficients[indices], color="b", align="center", width=0.6)  # Reduced bar width
    plt.xticks(range(len(indices)), np.array(feature_names)[indices], rotation=90, fontsize=8)  # Smaller font size
    plt.xlim([-1, len(indices)])
    plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels

    # Adding percentage values on top of bars
    for bar, coefficient in zip(bars, coefficients[indices]):
        percentage = f'{coefficient:.2f}'
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), percentage,
                 ha='center', va='bottom', fontsize=8)  # Smaller text values

    plt.show()

# Plotting for decision tree, random forest, and xgboost
for model_name in ['decision_tree', 'random_forest', 'xgboost']:
    model = best_models[model_name].named_steps['classifier']
    plot_feature_importance(model, feature_names, model_name)

# Function to plot feature importances derived from linear model coefficients (Logistic Regression, SVM)
def plot_linear_model_feature_importance(model, feature_names, model_name):
    importance = np.abs(model.coef_).flatten()
    importance = importance / np.sum(importance)
    indices = np.argsort(importance)[::-1]
    valid_indices = [i for i in indices if i < len(feature_names)]
    
    plt.figure(figsize=(8, 4))  # Reduced figure size
    plt.title(f"Feature Importances for {model_name}", fontsize=12)
    bars = plt.bar(range(len(valid_indices)), importance[valid_indices], color="b", align="center", width=0.6)  # Reduced bar width
    
    plt.xticks(range(len(valid_indices)), [feature_names[i] for i in valid_indices], rotation=90, fontsize=8)  # Smaller font size
    plt.xlim([-1, len(valid_indices)])
    plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels

    # Adding percentage values on top of bars
    for bar, imp in zip(bars, importance[valid_indices]):
        percentage = f'{imp * 100:.2f}%'
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), percentage,
                 ha='center', va='bottom', fontsize=8)  # Smaller text for values

    plt.show()

# Plotting feature importance for Logistic Regression
if 'logistic_regression' in best_models:
    model = best_models['logistic_regression'].named_steps['classifier']
    plot_linear_model_feature_importance(model, feature_names, 'logistic_regression')

# Plotting feature importance for SVM
if 'svm' in best_models:
    model = best_models['svm'].named_steps['classifier']
    plot_linear_model_feature_importance(model, feature_names, 'svm')


# In[ ]:





# In[ ]:





# In[ ]:




