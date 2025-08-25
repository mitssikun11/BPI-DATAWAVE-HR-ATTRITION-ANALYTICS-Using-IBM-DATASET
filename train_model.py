#!/usr/bin/env python3
"""
IBM HR Analytics Employee Attrition Prediction
Main training script for building and evaluating machine learning models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve, precision_recall_curve)
from sklearn.utils.class_weight import compute_class_weight

# Advanced ML Libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Utilities
import joblib
import shap
from datetime import datetime
import os

class HRAttritionPredictor:
    """
    Comprehensive HR Attrition Prediction System
    """
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.feature_importance = {}
        self.results = {}
        
    def load_data(self, data_path=None):
        """Load and prepare the dataset"""
        if data_path:
            self.data_path = data_path
            
        # For demonstration, we'll create sample data structure
        # In real scenario, load from CSV: self.data = pd.read_csv(data_path)
        
        print("Loading and preparing data...")
        
        # Create sample data structure (replace with actual data loading)
        np.random.seed(42)
        n_samples = 1000
        
        # Generate sample HR data
        self.data = pd.DataFrame({
            'Age': np.random.randint(22, 65, n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'Department': np.random.choice(['Sales', 'Research', 'HR', 'Engineering'], n_samples),
            'JobRole': np.random.choice(['Sales Executive', 'Research Scientist', 'HR Representative', 'Engineer'], n_samples),
            'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
            'MonthlyIncome': np.random.randint(3000, 15000, n_samples),
            'YearsAtCompany': np.random.randint(1, 20, n_samples),
            'YearsInCurrentRole': np.random.randint(1, 15, n_samples),
            'JobSatisfaction': np.random.randint(1, 5, n_samples),
            'WorkLifeBalance': np.random.randint(1, 5, n_samples),
            'JobInvolvement': np.random.randint(1, 5, n_samples),
            'PerformanceRating': np.random.randint(1, 5, n_samples),
            'BusinessTravel': np.random.choice(['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'], n_samples),
            'Overtime': np.random.choice(['Yes', 'No'], n_samples),
            'DistanceFromHome': np.random.randint(1, 30, n_samples),
            'NumCompaniesWorked': np.random.randint(0, 10, n_samples),
            'TrainingTimesLastYear': np.random.randint(0, 6, n_samples),
            'Attrition': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # 20% attrition rate
        })
        
        print(f"Dataset loaded with {len(self.data)} samples and {len(self.data.columns)} features")
        return self.data
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n=== EXPLORATORY DATA ANALYSIS ===")
        
        # Basic info
        print(f"Dataset shape: {self.data.shape}")
        print(f"Missing values:\n{self.data.isnull().sum()}")
        print(f"Data types:\n{self.data.dtypes}")
        
        # Target distribution
        attrition_rate = self.data['Attrition'].value_counts(normalize=True)
        print(f"\nAttrition Rate: {attrition_rate[1]:.2%}")
        
        # Create visualizations
        self._create_eda_plots()
        
    def _create_eda_plots(self):
        """Create exploratory data analysis plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Attrition by Age
        self.data.boxplot(column='Age', by='Attrition', ax=axes[0,0])
        axes[0,0].set_title('Age Distribution by Attrition')
        
        # Attrition by Department
        dept_attrition = self.data.groupby('Department')['Attrition'].mean()
        dept_attrition.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Attrition Rate by Department')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Monthly Income distribution
        self.data['MonthlyIncome'].hist(bins=30, ax=axes[0,2])
        axes[0,2].set_title('Monthly Income Distribution')
        
        # Years at Company
        self.data.boxplot(column='YearsAtCompany', by='Attrition', ax=axes[1,0])
        axes[1,0].set_title('Years at Company by Attrition')
        
        # Job Satisfaction
        self.data.groupby('JobSatisfaction')['Attrition'].mean().plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Attrition Rate by Job Satisfaction')
        
        # Work Life Balance
        self.data.groupby('WorkLifeBalance')['Attrition'].mean().plot(kind='bar', ax=axes[1,2])
        axes[1,2].set_title('Attrition Rate by Work Life Balance')
        
        plt.tight_layout()
        plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def preprocess_data(self):
        """Preprocess the data for machine learning"""
        print("\n=== DATA PREPROCESSING ===")
        
        # Create a copy for preprocessing
        df = self.data.copy()
        
        # Handle categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'Attrition':  # Skip target variable
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        
        # Separate features and target
        X = df.drop('Attrition', axis=1)
        y = df['Attrition']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numerical features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Class distribution in training set: {np.bincount(self.y_train)}")
        
    def train_models(self):
        """Train multiple machine learning models"""
        print("\n=== MODEL TRAINING ===")
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(self.y_train), 
            y=self.y_train
        )
        class_weight_dict = dict(zip(np.unique(self.y_train), class_weights))
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                class_weight='balanced',
                max_iter=1000
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            ),
            'XGBoost': xgb.XGBClassifier(
                random_state=42,
                scale_pos_weight=len(self.y_train[self.y_train==0]) / len(self.y_train[self.y_train==1])
            ),
            'LightGBM': lgb.LGBMClassifier(
                random_state=42,
                class_weight='balanced'
            ),
            'CatBoost': CatBoostClassifier(
                random_state=42,
                verbose=False
            )
        }
        
        # Train models
        for name, model in models.items():
            print(f"Training {name}...")
            
            if name in ['Logistic Regression']:
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Store model and predictions
            self.models[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # Calculate metrics
            self._calculate_metrics(name, y_pred, y_pred_proba)
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
        
        print("All models trained successfully!")
        
    def _calculate_metrics(self, model_name, y_pred, y_pred_proba):
        """Calculate and store model performance metrics"""
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
        }
        
        self.results[model_name] = metrics
        
        print(f"{model_name} - Accuracy: {metrics['accuracy']:.3f}, "
              f"Precision: {metrics['precision']:.3f}, "
              f"Recall: {metrics['recall']:.3f}, "
              f"F1: {metrics['f1_score']:.3f}, "
              f"ROC-AUC: {metrics['roc_auc']:.3f}")
        
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        print("\n=== MODEL EVALUATION ===")
        
        # Create comparison plot
        self._plot_model_comparison()
        
        # Create ROC curves
        self._plot_roc_curves()
        
        # Create confusion matrices
        self._plot_confusion_matrices()
        
        # Feature importance analysis
        self._plot_feature_importance()
        
        # Detailed classification reports
        self._print_classification_reports()
        
    def _plot_model_comparison(self):
        """Plot model performance comparison"""
        metrics_df = pd.DataFrame(self.results).T
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for i, metric in enumerate(['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']):
            row, col = i // 3, i % 3
            metrics_df[metric].plot(kind='bar', ax=axes[row, col])
            axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
            axes[row, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _plot_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for name, model_data in self.models.items():
            fpr, tpr, _ = roc_curve(self.y_test, model_data['probabilities'])
            auc = roc_auc_score(self.y_test, model_data['probabilities'])
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.models)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (name, model_data) in enumerate(self.models.items()):
            cm = confusion_matrix(self.y_test, model_data['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{name} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        tree_models = {k: v for k, v in self.feature_importance.items() 
                      if k in ['Random Forest', 'XGBoost', 'LightGBM']}
        
        if tree_models:
            fig, axes = plt.subplots(1, len(tree_models), figsize=(15, 5))
            if len(tree_models) == 1:
                axes = [axes]
            
            feature_names = self.X_train.columns
            
            for i, (name, importance) in enumerate(tree_models.items()):
                # Get top 10 features
                top_indices = np.argsort(importance)[-10:]
                axes[i].barh(range(10), importance[top_indices])
                axes[i].set_yticks(range(10))
                axes[i].set_yticklabels([feature_names[j] for j in top_indices])
                axes[i].set_title(f'{name} Feature Importance')
            
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
    def _print_classification_reports(self):
        """Print detailed classification reports"""
        print("\n=== DETAILED CLASSIFICATION REPORTS ===")
        
        for name, model_data in self.models.items():
            print(f"\n{name}:")
            print(classification_report(self.y_test, model_data['predictions']))
            
    def save_models(self):
        """Save trained models and preprocessing objects"""
        print("\n=== SAVING MODELS ===")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save models
        for name, model_data in self.models.items():
            model_path = f'models/{name.lower().replace(" ", "_")}.joblib'
            joblib.dump(model_data['model'], model_path)
            print(f"Saved {name} to {model_path}")
        
        # Save preprocessing objects
        joblib.dump(self.scaler, 'models/scaler.joblib')
        joblib.dump(self.label_encoders, 'models/label_encoders.joblib')
        
        # Save results
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv('models/model_results.csv')
        print("Saved model results to models/model_results.csv")
        
    def generate_business_insights(self):
        """Generate business insights and recommendations"""
        print("\n=== BUSINESS INSIGHTS ===")
        
        # Find best model
        best_model = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        print(f"Best performing model: {best_model[0]} (F1-Score: {best_model[1]['f1_score']:.3f})")
        
        # Key insights from feature importance
        if 'Random Forest' in self.feature_importance:
            importance = self.feature_importance['Random Forest']
            feature_names = self.X_train.columns
            top_features = feature_names[np.argsort(importance)[-5:]]
            
            print("\nTop 5 factors contributing to employee attrition:")
            for i, feature in enumerate(reversed(top_features), 1):
                print(f"{i}. {feature}")
        
        # Recommendations
        print("\n=== RECOMMENDATIONS ===")
        print("1. Focus on employee satisfaction and work-life balance")
        print("2. Implement retention programs for high-risk employees")
        print("3. Regular employee feedback and engagement surveys")
        print("4. Career development and growth opportunities")
        print("5. Competitive compensation and benefits packages")
        
    def run_complete_pipeline(self):
        """Run the complete machine learning pipeline"""
        print("=== IBM HR ANALYTICS EMPLOYEE ATTRITION PREDICTION ===")
        print("Starting complete pipeline...")
        
        # Load data
        self.load_data()
        
        # Explore data
        self.explore_data()
        
        # Preprocess data
        self.preprocess_data()
        
        # Train models
        self.train_models()
        
        # Evaluate models
        self.evaluate_models()
        
        # Save models
        self.save_models()
        
        # Generate insights
        self.generate_business_insights()
        
        print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
        print("Check the generated plots and saved models in the project directory.")

if __name__ == "__main__":
    # Initialize the predictor
    predictor = HRAttritionPredictor()
    
    # Run the complete pipeline
    predictor.run_complete_pipeline() 