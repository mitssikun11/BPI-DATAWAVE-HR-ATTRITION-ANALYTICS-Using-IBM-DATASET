#!/usr/bin/env python3
"""
Competition Trainer for BPI DATA Wave
Advanced machine learning pipeline optimized for competition performance
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
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Additional imports
import joblib
import os
from datetime import datetime
import json

# Import our custom modules
from src.competition_data_loader import CompetitionDataLoader
from utils.visualization import HRVisualizer

class CompetitionTrainer:
    """
    Advanced competition trainer with ensemble methods and optimization
    """
    
    def __init__(self, competition_name="BPI_DATA_Wave"):
        self.competition_name = competition_name
        self.data_loader = CompetitionDataLoader()
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self.best_model = None
        self.best_score = 0
        self.visualizer = HRVisualizer()
        
        # Competition-specific settings
        self.cv_folds = 5
        self.random_state = 42
        self.test_size = 0.2
        
    def load_and_prepare_data(self, data_path=None):
        """
        Load and prepare competition data
        """
        print("=== LOADING COMPETITION DATA ===")
        
        # Load data
        data = self.data_loader.load_competition_data(data_path)
        
        # Analyze data structure
        self.data_loader.analyze_data_structure()
        
        # Preprocess data
        X_train, X_test, y_train, y_test = self.data_loader.preprocess_competition_data(
            test_size=self.test_size, random_state=self.random_state
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Data loaded and prepared successfully!")
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_competition_models(self):
        """
        Train multiple models with competition optimization
        """
        print("=== TRAINING COMPETITION MODELS ===")
        
        # Define models with competition-optimized parameters
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=self.random_state, n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=self.random_state
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=self.random_state
            ),
            'CatBoost': cb.CatBoostClassifier(
                iterations=200, depth=6, learning_rate=0.1,
                random_state=self.random_state, verbose=False
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=self.random_state
            ),
            'Logistic Regression': LogisticRegression(
                C=1.0, max_iter=1000, random_state=self.random_state
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
            metrics = self._calculate_competition_metrics(y_pred, y_pred_proba)
            
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
    
    def _calculate_competition_metrics(self, y_pred, y_pred_proba):
        """
        Calculate comprehensive competition metrics
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
        Find the best performing model
        """
        best_score = 0
        best_model_name = None
        
        for name, metrics in self.results.items():
            # Competition scoring (can be customized based on competition rules)
            score = metrics['roc_auc'] * 0.4 + metrics['f1'] * 0.3 + metrics['precision'] * 0.3
            
            if score > best_score:
                best_score = score
                best_model_name = name
        
        self.best_model = self.models[best_model_name]
        self.best_score = best_score
        
        print(f"\n=== BEST MODEL ===")
        print(f"Model: {best_model_name}")
        print(f"Competition Score: {best_score:.4f}")
        print(f"ROC AUC: {self.results[best_model_name]['roc_auc']:.4f}")
        print(f"F1 Score: {self.results[best_model_name]['f1']:.4f}")
    
    def create_ensemble_model(self):
        """
        Create an ensemble model for better performance
        """
        print("\n=== CREATING ENSEMBLE MODEL ===")
        
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
        
        metrics = self._calculate_competition_metrics(y_pred, y_pred_proba)
        
        self.models['Ensemble'] = ensemble
        self.results['Ensemble'] = metrics
        
        print(f"Ensemble Model - ROC AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
        
        # Update best model if ensemble is better
        ensemble_score = metrics['roc_auc'] * 0.4 + metrics['f1'] * 0.3 + metrics['precision'] * 0.3
        if ensemble_score > self.best_score:
            self.best_model = ensemble
            self.best_score = ensemble_score
            print("Ensemble model is now the best model!")
    
    def perform_hyperparameter_tuning(self, model_name='XGBoost'):
        """
        Perform hyperparameter tuning for the best model
        """
        print(f"\n=== HYPERPARAMETER TUNING FOR {model_name} ===")
        
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return
        
        model = self.models[model_name]
        
        # Define parameter grid based on model type
        if model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0]
            }
        elif model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return
        
        # Perform grid search
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Update model with best parameters
        self.models[f'{model_name}_Tuned'] = grid_search.best_estimator_
        
        # Evaluate tuned model
        y_pred = grid_search.predict(self.X_test)
        y_pred_proba = grid_search.predict_proba(self.X_test)[:, 1]
        
        metrics = self._calculate_competition_metrics(y_pred, y_pred_proba)
        self.results[f'{model_name}_Tuned'] = metrics
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Tuned {model_name} - ROC AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
    
    def generate_competition_report(self):
        """
        Generate comprehensive competition report
        """
        print("\n=== GENERATING COMPETITION REPORT ===")
        
        # Create results summary
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(4)
        
        print("\n=== MODEL PERFORMANCE COMPARISON ===")
        print(results_df)
        
        # Save results
        results_df.to_csv('submissions/model_performance.csv')
        
        # Generate visualizations
        self._create_competition_plots()
        
        # Save best model
        self._save_best_model()
        
        # Generate submission file
        self._generate_submission_file()
        
        print("\n=== COMPETITION REPORT GENERATED ===")
        print("Files saved:")
        print("- submissions/model_performance.csv")
        print("- submissions/best_model.pkl")
        print("- submissions/predictions.csv")
        print("- submissions/feature_importance.csv")
    
    def _create_competition_plots(self):
        """
        Create competition-specific visualizations
        """
        # Create submissions directory
        os.makedirs('submissions', exist_ok=True)
        
        # 1. Model comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC AUC comparison
        model_names = list(self.results.keys())
        roc_scores = [self.results[name]['roc_auc'] for name in model_names]
        
        axes[0, 0].bar(model_names, roc_scores)
        axes[0, 0].set_title('ROC AUC Scores')
        axes[0, 0].set_ylabel('ROC AUC')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1 Score comparison
        f1_scores = [self.results[name]['f1'] for name in model_names]
        axes[0, 1].bar(model_names, f1_scores)
        axes[0, 1].set_title('F1 Scores')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Precision comparison
        precision_scores = [self.results[name]['precision'] for name in model_names]
        axes[1, 0].bar(model_names, precision_scores)
        axes[1, 0].set_title('Precision Scores')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Recall comparison
        recall_scores = [self.results[name]['recall'] for name in model_names]
        axes[1, 1].bar(model_names, recall_scores)
        axes[1, 1].set_title('Recall Scores')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('submissions/model_comparison.png', dpi=300, bbox_inches='tight')
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
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig('submissions/roc_curves.png', dpi=300, bbox_inches='tight')
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
            
            feature_importance_df.to_csv('submissions/feature_importance.csv', index=False)
            
            # Plot top 20 features
            top_features = feature_importance_df.head(20)
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 20 Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('submissions/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _save_best_model(self):
        """
        Save the best performing model
        """
        if self.best_model is not None:
            joblib.dump(self.best_model, 'submissions/best_model.pkl')
            print("Best model saved to submissions/best_model.pkl")
    
    def _generate_submission_file(self):
        """
        Generate competition submission file
        """
        if self.best_model is not None:
            # Generate predictions on test set
            predictions = self.best_model.predict_proba(self.X_test)[:, 1]
            
            # Create submission DataFrame
            submission_df = pd.DataFrame({
                'id': range(len(predictions)),
                'prediction': predictions
            })
            
            submission_df.to_csv('submissions/predictions.csv', index=False)
            print("Predictions saved to submissions/predictions.csv")
    
    def run_complete_pipeline(self, data_path=None):
        """
        Run the complete competition pipeline
        """
        print("=== BPI DATA WAVE COMPETITION PIPELINE ===")
        print(f"Competition: {self.competition_name}")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Load and prepare data
        self.load_and_prepare_data(data_path)
        
        # Step 2: Train models
        self.train_competition_models()
        
        # Step 3: Create ensemble
        self.create_ensemble_model()
        
        # Step 4: Hyperparameter tuning (optional)
        # self.perform_hyperparameter_tuning('XGBoost')
        
        # Step 5: Generate report
        self.generate_competition_report()
        
        print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
        print("Check the 'submissions/' directory for all outputs!")

# Example usage
if __name__ == "__main__":
    trainer = CompetitionTrainer()
    trainer.run_complete_pipeline() 