#!/usr/bin/env python3
"""
Test script for IBM HR Analytics pipeline
"""
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_hr_pipeline():
    """
    Test the complete HR pipeline with the actual dataset
    """
    print("=== TESTING IBM HR ANALYTICS PIPELINE ===")
    
    try:
        # Test 1: Import modules
        print("1. Testing imports...")
        from hr_competition_trainer import HRCompetitionTrainer
        print("✓ All imports successful")
        
        # Test 2: Load HR data
        print("\n2. Testing HR data loading...")
        trainer = HRCompetitionTrainer()
        data = trainer.load_hr_data()
        print(f"✓ HR data loaded successfully: {data.shape}")
        
        # Test 3: Analyze HR data
        print("\n3. Testing HR data analysis...")
        analysis = trainer.analyze_hr_data()
        print(f"✓ HR data analyzed: {analysis['numerical_columns']} numerical, {analysis['categorical_columns']} categorical")
        
        # Test 4: Preprocess HR data
        print("\n4. Testing HR data preprocessing...")
        X_train, X_test, y_train, y_test = trainer.preprocess_hr_data()
        print(f"✓ HR data preprocessed: Train {X_train.shape}, Test {X_test.shape}")
        
        # Test 5: Quick model training
        print("\n5. Testing HR model training...")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        from sklearn.metrics import roc_auc_score, f1_score, recall_score
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        print(f"✓ HR model trained successfully:")
        print(f"  - ROC AUC: {roc_auc:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Recall: {recall:.4f}")
        
        # Test 6: File creation
        print("\n6. Testing HR file creation...")
        import joblib
        os.makedirs('submissions', exist_ok=True)
        
        # Save test model
        joblib.dump(model, 'submissions/hr_test_model.pkl')
        
        # Create test predictions
        import pandas as pd
        predictions_df = pd.DataFrame({
            'id': range(len(y_pred_proba)),
            'attrition_probability': y_pred_proba
        })
        predictions_df.to_csv('submissions/hr_test_predictions.csv', index=False)
        
        print("✓ HR test files created successfully")
        
        print("\n=== ALL HR TESTS PASSED! ===")
        print("The HR pipeline is working correctly with the actual dataset.")
        print("\nNext steps:")
        print("1. Run 'python hr_competition_trainer.py' for full HR pipeline")
        print("2. Check the submissions/ directory for HR-specific outputs")
        print("3. Review the HR feature importance and model performance")
        
        return True
        
    except Exception as e:
        print(f"\n❌ HR test failed with error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the HR dataset file exists: WA_Fn-UseC_-HR-Employee-Attrition.csv")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Check that all files are in the correct locations")
        return False

def analyze_hr_dataset():
    """
    Analyze the HR dataset structure
    """
    print("\n=== ANALYZING HR DATASET ===")
    
    try:
        import pandas as pd
        
        # Load the dataset
        data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
        
        print(f"✓ Dataset loaded: {data.shape}")
        print(f"✓ Columns: {list(data.columns)}")
        
        # Target analysis
        target_counts = data['Attrition'].value_counts()
        print(f"✓ Target distribution: {target_counts.to_dict()}")
        print(f"✓ Attrition rate: {target_counts['Yes'] / len(data) * 100:.2f}%")
        
        # Column types
        numerical_cols = []
        categorical_cols = []
        
        for col in data.columns:
            if col in ['Attrition', 'EmployeeNumber']:
                continue
                
            if data[col].dtype in ['int64', 'float64']:
                numerical_cols.append(col)
            else:
                categorical_cols.append(col)
        
        print(f"✓ Numerical columns: {len(numerical_cols)}")
        print(f"✓ Categorical columns: {len(categorical_cols)}")
        
        # Missing values
        missing_data = data.isnull().sum()
        if missing_data.sum() > 0:
            print(f"⚠️  Missing values: {missing_data[missing_data > 0].to_dict()}")
        else:
            print("✓ No missing values found")
        
        return True
        
    except Exception as e:
        print(f"❌ HR dataset analysis failed: {e}")
        return False

def main():
    """
    Run all HR tests
    """
    print("Starting IBM HR Analytics pipeline tests...\n")
    
    # Test dataset analysis
    dataset_test = analyze_hr_dataset()
    
    # Test full pipeline
    pipeline_test = test_hr_pipeline()
    
    if dataset_test and pipeline_test:
        print("\n🎉 All HR tests completed successfully!")
        print("Your HR pipeline is ready to use with the actual dataset!")
    else:
        print("\n⚠️  Some HR tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 