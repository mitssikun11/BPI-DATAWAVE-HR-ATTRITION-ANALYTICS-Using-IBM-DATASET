#!/usr/bin/env python3
"""
IBM HR Analytics Competition Trainer for BPI DATA Wave
Specialized for the IBM HR Employee Attrition dataset
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

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Additional imports
import joblib
import os
from datetime import datetime
import json

class HRCompetitionTrainer:
    """
    Specialized competition trainer for IBM HR Analytics dataset
    """
    
    def __init__(self, data_path="WA_Fn-UseC_-HR-Employee-Attrition.csv"):
        self.data_path = data_path
        self.data = None
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self.best_model = None
        self.best_score = 0
        
        # HR-specific settings
        self.cv_folds = 5
        self.random_state = 42
        self.test_size = 0.2
        
        # HR dataset specific columns
        self.target_column = 'Attrition'
        self.id_column = 'EmployeeNumber'
        
    def load_hr_data(self):
        """
        Load and prepare the IBM HR Analytics dataset
        """
        print("=== LOADING IBM HR ANALYTICS DATASET ===")
        
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"✓ Dataset loaded successfully: {self.data.shape}")
            print(f"✓ Columns: {list(self.data.columns)}")
            
            # Basic data info
            print(f"\n=== DATASET INFO ===")
            print(f"Shape: {self.data.shape}")
            print(f"Memory usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            print(f"Target distribution: {self.data[self.target_column].value_counts().to_dict()}")
            
            return self.data
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def analyze_hr_data(self):
        """
        Analyze the HR dataset structure
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        print("\n=== HR DATASET ANALYSIS ===")
        
        # Identify column types
        numerical_cols = []
        categorical_cols = []
        
        for col in self.data.columns:
            if col in [self.target_column, self.id_column]:
                continue
                
            if self.data[col].dtype in ['int64', 'float64']:
                numerical_cols.append(col)
            else:
                categorical_cols.append(col)
        
        print(f"Numerical columns: {len(numerical_cols)}")
        print(f"Categorical columns: {len(categorical_cols)}")
        
        # Missing values
        missing_data = self.data.isnull().sum()
        if missing_data.sum() > 0:
            print(f"\nMissing values: {missing_data[missing_data > 0].to_dict()}")
        else:
            print("\n✓ No missing values found")
        
        # Target analysis
        print(f"\n=== TARGET ANALYSIS ===")
        target_counts = self.data[self.target_column].value_counts()
        print(f"Target distribution: {target_counts.to_dict()}")
        print(f"Attrition rate: {target_counts['Yes'] / len(self.data) * 100:.2f}%")
        
        return {
            'numerical_columns': numerical_cols,
            'categorical_columns': categorical_cols,
            'target_distribution': target_counts.to_dict()
        }
    
    def preprocess_hr_data(self):
        """
        Preprocess the HR dataset for modeling
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return None, None, None, None
        
        print("\n=== PREPROCESSING HR DATA ===")
        
        # Create a copy for preprocessing
        data_processed = self.data.copy()
        
        # Remove ID column if present
        if self.id_column in data_processed.columns:
            data_processed = data_processed.drop(columns=[self.id_column])
        
        # Handle target variable (convert Yes/No to 1/0)
        data_processed[self.target_column] = (data_processed[self.target_column] == 'Yes').astype(int)
        
        # Identify column types
        numerical_cols = []
        categorical_cols = []
        
        for col in data_processed.columns:
            if col == self.target_column:
                continue
                
            if data_processed[col].dtype in ['int64', 'float64']:
                numerical_cols.append(col)
            else:
                categorical_cols.append(col)
        
        # Handle missing values
        if numerical_cols:
            num_imputer = SimpleImputer(strategy='median')
            data_processed[numerical_cols] = num_imputer.fit_transform(data_processed[numerical_cols])
        
        if categorical_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            data_processed[categorical_cols] = cat_imputer.fit_transform(data_processed[categorical_cols])
        
        # Encode categorical variables
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            data_processed[col] = le.fit_transform(data_processed[col].astype(str))
            label_encoders[col] = le
        
        # Scale numerical features
        if numerical_cols:
            scaler = StandardScaler()
            data_processed[numerical_cols] = scaler.fit_transform(data_processed[numerical_cols])
        
        # Split features and target
        X = data_processed.drop(columns=[self.target_column])
        y = data_processed[self.target_column]
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"✓ Training set shape: {X_train.shape}")
        print(f"✓ Test set shape: {X_test.shape}")
        print(f"✓ Target distribution (train): {y_train.value_counts().to_dict()}")
        print(f"✓ Target distribution (test): {y_test.value_counts().to_dict()}")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train, X_test, y_train, y_test
    
    def train_hr_models(self):
        """
        Train multiple models for HR attrition prediction
        """
        print("\n=== TRAINING HR ATTRITION MODELS ===")
        
        # Define models optimized for HR data
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=self.random_state, n_jobs=-1,
                class_weight='balanced'
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=self.random_state,
                scale_pos_weight=3  # Handle class imbalance
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=self.random_state,
                class_weight='balanced'
            ),
            'CatBoost': cb.CatBoostClassifier(
                iterations=200, depth=6, learning_rate=0.1,
                random_state=self.random_state, verbose=False,
                class_weights=[1, 3]  # Handle class imbalance
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=self.random_state
            ),
            'Logistic Regression': LogisticRegression(
                C=1.0, max_iter=1000, random_state=self.random_state,
                class_weight='balanced'
            )
        }
        
        # Train each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            metrics = self._calculate_hr_metrics(y_pred, y_pred_proba)
            
            # Store results
            self.models[name] = model
            self.results[name] = metrics
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
            
            print(f"{name} - ROC AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
        
        # Find best model
        self._find_best_model()
        
        return self.models, self.results
    
    def _calculate_hr_metrics(self, y_pred, y_pred_proba):
        """
        Calculate comprehensive metrics for HR attrition prediction
        """
        return {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='weighted'),
            'recall': recall_score(self.y_test, y_pred, average='weighted'),
            'f1': f1_score(self.y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'average_precision': average_precision_score(self.y_test, y_pred_proba)
        }
    
    def _find_best_model(self):
        """
        Find the best performing model for HR attrition
        """
        best_score = 0
        best_model_name = None
        
        for name, metrics in self.results.items():
            # HR-specific scoring (focus on recall for attrition prediction)
            score = metrics['roc_auc'] * 0.3 + metrics['f1'] * 0.3 + metrics['recall'] * 0.4
            
            if score > best_score:
                best_score = score
                best_model_name = name
        
        self.best_model = self.models[best_model_name]
        self.best_score = best_score
        
        print(f"\n=== BEST HR MODEL ===")
        print(f"Model: {best_model_name}")
        print(f"HR Competition Score: {best_score:.4f}")
        print(f"ROC AUC: {self.results[best_model_name]['roc_auc']:.4f}")
        print(f"F1 Score: {self.results[best_model_name]['f1']:.4f}")
        print(f"Recall: {self.results[best_model_name]['recall']:.4f}")
    
    def create_hr_ensemble(self):
        """
        Create an ensemble model for HR attrition prediction
        """
        print("\n=== CREATING HR ENSEMBLE MODEL ===")
        
        # Select top performing models for ensemble
        top_models = []
        model_scores = []
        
        for name, metrics in self.results.items():
            score = metrics['roc_auc']
            model_scores.append((name, score))
        
        # Sort by ROC AUC score
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 3 models for ensemble
        top_model_names = [name for name, score in model_scores[:3]]
        top_models = [(name, self.models[name]) for name in top_model_names]
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=top_models,
            voting='soft'
        )
        
        # Train ensemble
        ensemble.fit(self.X_train, self.y_train)
        
        # Evaluate ensemble
        y_pred = ensemble.predict(self.X_test)
        y_pred_proba = ensemble.predict_proba(self.X_test)[:, 1]
        
        metrics = self._calculate_hr_metrics(y_pred, y_pred_proba)
        
        self.models['HR Ensemble'] = ensemble
        self.results['HR Ensemble'] = metrics
        
        print(f"HR Ensemble Model - ROC AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
        
        # Update best model if ensemble is better
        ensemble_score = metrics['roc_auc'] * 0.3 + metrics['f1'] * 0.3 + metrics['recall'] * 0.4
        if ensemble_score > self.best_score:
            self.best_model = ensemble
            self.best_score = ensemble_score
            print("HR Ensemble model is now the best model!")
    
    def generate_hr_report(self):
        """
        Generate comprehensive HR competition report
        """
        print("\n=== GENERATING HR COMPETITION REPORT ===")
        
        # Create results summary
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(4)
        
        print("\n=== HR MODEL PERFORMANCE COMPARISON ===")
        print(results_df)
        
        # Save results
        results_df.to_csv('submissions/hr_model_performance.csv')
        
        # Generate visualizations
        self._create_hr_plots()
        
        # Save best model
        self._save_best_model()
        
        # Generate submission file
        self._generate_hr_submission()
        
        print("\n=== HR COMPETITION REPORT GENERATED ===")
        print("Files saved:")
        print("- submissions/hr_model_performance.csv")
        print("- submissions/hr_best_model.pkl")
        print("- submissions/hr_predictions.csv")
        print("- submissions/hr_feature_importance.csv")
    
    def _create_hr_plots(self):
        """
        Create HR-specific visualizations
        """
        # Create submissions directory
        os.makedirs('submissions', exist_ok=True)
        
        # 1. Model comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC AUC comparison
        model_names = list(self.results.keys())
        roc_scores = [self.results[name]['roc_auc'] for name in model_names]
        
        axes[0, 0].bar(model_names, roc_scores)
        axes[0, 0].set_title('HR Model ROC AUC Scores')
        axes[0, 0].set_ylabel('ROC AUC')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1 Score comparison
        f1_scores = [self.results[name]['f1'] for name in model_names]
        axes[0, 1].bar(model_names, f1_scores)
        axes[0, 1].set_title('HR Model F1 Scores')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Recall comparison
        recall_scores = [self.results[name]['recall'] for name in model_names]
        axes[1, 0].bar(model_names, recall_scores)
        axes[1, 0].set_title('HR Model Recall Scores')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Precision comparison
        precision_scores = [self.results[name]['precision'] for name in model_names]
        axes[1, 1].bar(model_names, precision_scores)
        axes[1, 1].set_title('HR Model Precision Scores')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('submissions/hr_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC curves
        plt.figure(figsize=(10, 8))
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('HR Attrition ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig('submissions/hr_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Feature importance (for best model)
        if hasattr(self.best_model, 'feature_importances_'):
            feature_names = self.X_train.columns
            importance = self.best_model.feature_importances_
            
            # Create feature importance DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            feature_importance_df.to_csv('submissions/hr_feature_importance.csv', index=False)
            
            # Plot top 20 features
            top_features = feature_importance_df.head(20)
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 20 HR Features for Attrition Prediction')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('submissions/hr_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _save_best_model(self):
        """
        Save the best performing HR model
        """
        if self.best_model is not None:
            joblib.dump(self.best_model, 'submissions/hr_best_model.pkl')
            print("Best HR model saved to submissions/hr_best_model.pkl")
    
    def _generate_hr_submission(self):
        """
        Generate HR competition submission file
        """
        if self.best_model is not None:
            # Generate predictions on test set
            predictions = self.best_model.predict_proba(self.X_test)[:, 1]
            
            # Create submission DataFrame
            submission_df = pd.DataFrame({
                'id': range(len(predictions)),
                'attrition_probability': predictions
            })
            
            submission_df.to_csv('submissions/hr_predictions.csv', index=False)
            print("HR predictions saved to submissions/hr_predictions.csv")
    
    def run_complete_hr_pipeline(self):
        """
        Run the complete HR competition pipeline
        """
        print("=== IBM HR ANALYTICS COMPETITION PIPELINE ===")
        print(f"Dataset: {self.data_path}")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Load HR data
        self.load_hr_data()
        
        # Step 2: Analyze HR data
        self.analyze_hr_data()
        
        # Step 3: Preprocess HR data
        self.preprocess_hr_data()
        
        # Step 4: Train HR models
        self.train_hr_models()
        
        # Step 5: Create HR ensemble
        self.create_hr_ensemble()
        
        # Step 6: Generate HR report
        self.generate_hr_report()
        
        print("\n=== HR PIPELINE COMPLETED SUCCESSFULLY ===")
        print("Check the 'submissions/' directory for all HR outputs!")

# Example usage
if __name__ == "__main__":
    trainer = HRCompetitionTrainer()
    trainer.run_complete_hr_pipeline() 