#!/usr/bin/env python3
"""
Test script for BPI DATA Wave competition pipeline
"""
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_competition_pipeline():
    """
    Test the complete competition pipeline
    """
    print("=== TESTING BPI DATA WAVE COMPETITION PIPELINE ===")
    
    try:
        # Test 1: Import modules
        print("1. Testing imports...")
        from src.competition_data_loader import CompetitionDataLoader
        from competition_trainer import CompetitionTrainer
        print("✓ All imports successful")
        
        # Test 2: Data loader
        print("\n2. Testing data loader...")
        loader = CompetitionDataLoader()
        data = loader.load_competition_data()
        print(f"✓ Data loaded successfully: {data.shape}")
        
        # Test 3: Data preprocessing
        print("\n3. Testing data preprocessing...")
        X_train, X_test, y_train, y_test = loader.preprocess_competition_data()
        print(f"✓ Data preprocessed: Train {X_train.shape}, Test {X_test.shape}")
        
        # Test 4: Model training (quick test)
        print("\n4. Testing model training...")
        trainer = CompetitionTrainer()
        trainer.X_train = X_train
        trainer.X_test = X_test
        trainer.y_train = y_train
        trainer.y_test = y_test
        
        # Train just one model for testing
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        from sklearn.metrics import roc_auc_score, f1_score
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"✓ Model trained successfully: ROC AUC = {roc_auc:.4f}, F1 = {f1:.4f}")
        
        # Test 5: File creation
        print("\n5. Testing file creation...")
        import joblib
        os.makedirs('submissions', exist_ok=True)
        
        # Save test model
        joblib.dump(model, 'submissions/test_model.pkl')
        
        # Create test predictions
        import pandas as pd
        predictions_df = pd.DataFrame({
            'id': range(len(y_pred_proba)),
            'prediction': y_pred_proba
        })
        predictions_df.to_csv('submissions/test_predictions.csv', index=False)
        
        print("✓ Test files created successfully")
        
        print("\n=== ALL TESTS PASSED! ===")
        print("The competition pipeline is working correctly.")
        print("\nNext steps:")
        print("1. Run 'python competition_trainer.py' for full pipeline")
        print("2. Add your real competition data to the 'data/' directory")
        print("3. Customize the models and preprocessing as needed")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check that all files are in the correct locations")
        print("3. Verify Python version (3.8+ recommended)")
        return False

def test_data_generation():
    """
    Test synthetic data generation
    """
    print("\n=== TESTING DATA GENERATION ===")
    
    try:
        from src.competition_data_loader import CompetitionDataLoader
        
        loader = CompetitionDataLoader()
        data = loader._generate_competition_data()
        
        print(f"✓ Synthetic data generated: {data.shape}")
        print(f"✓ Columns: {list(data.columns)}")
        print(f"✓ Target column: {loader.target_column}")
        print(f"✓ Numerical columns: {len(loader.numerical_columns)}")
        print(f"✓ Categorical columns: {len(loader.categorical_columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        return False

def main():
    """
    Run all tests
    """
    print("Starting BPI DATA Wave competition pipeline tests...\n")
    
    # Test data generation
    data_test = test_data_generation()
    
    # Test full pipeline
    pipeline_test = test_competition_pipeline()
    
    if data_test and pipeline_test:
        print("\n🎉 All tests completed successfully!")
        print("Your competition pipeline is ready to use!")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 