# complete_debug.py - Full diagnostic script to find the exact issue
import pandas as pd
import numpy as np
import pickle
import os
import sys

def check_training_data():
    """Check the actual training data used"""
    print("🔍 CHECKING TRAINING DATA")
    print("="*50)
    
    try:
        # Load processed training data
        train_data = pd.read_csv('data/processed_train.csv')
        print(f"✅ Training data shape: {train_data.shape}")
        
        # Check target distribution
        target_dist = train_data['target'].value_counts()
        print(f"📊 Target distribution: {target_dist.to_dict()}")
        print(f"📊 Target balance: {target_dist[1]/(target_dist[0]+target_dist[1]):.3f}")
        
        # Check feature ranges
        feature_cols = [col for col in train_data.columns if col != 'target']
        print(f"\n📈 Feature statistics (first 10):")
        for col in feature_cols[:10]:
            print(f"   {col}: mean={train_data[col].mean():.3f}, std={train_data[col].std():.3f}, range=[{train_data[col].min():.2f}, {train_data[col].max():.2f}]")
        
        # Check for any constant features
        constant_features = [col for col in feature_cols if train_data[col].std() == 0]
        if constant_features:
            print(f"⚠️  Constant features (std=0): {constant_features}")
        
        # Check correlations with target
        correlations = train_data.corr()['target'].abs().sort_values(ascending=False)
        print(f"\n🔗 Top 5 correlations with target:")
        for feature, corr in correlations.head(6).items():  # 6 because target is included
            if feature != 'target':
                print(f"   {feature}: {corr:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_model_file():
    """Check the actual saved model"""
    print("\n🤖 CHECKING SAVED MODEL")
    print("="*50)
    
    try:
        with open('models/trained_models.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"✅ Models loaded: {list(model_data['models'].keys())}")
        print(f"✅ Best model: {model_data['best_model_name']}")
        print(f"✅ Features: {len(model_data['feature_names'])}")
        print(f"✅ Feature order: {model_data['feature_names']}")
        
        # Check model performance
        for name, info in model_data['models'].items():
            print(f"   {name}: CV AUC = {info['cv_score']:.3f}")
        
        return model_data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_preprocessing_directly():
    """Test preprocessing function directly"""
    print("\n⚙️  TESTING PREPROCESSING FUNCTION")
    print("="*50)
    
    try:
        # Import the API
        sys.path.append('src')
        from app import HeartDiseaseAPI
        
        api = HeartDiseaseAPI()
        
        if api.models is None:
            print("❌ Models not loaded")
            return False
        
        # Test case: Young, healthy female (should be LOW risk)
        young_patient = {
            "age": "30",
            "sex": "female",
            "chest_pain": "3",           # Asymptomatic (GOOD)
            "blood_pressure": "110",     # Normal
            "cholesterol": "175",        # Normal
            "fasting_sugar": "no",       # Normal
            "rest_ecg": "0",            # Normal
            "max_heart_rate": "180",     # High (GOOD for young person)
            "exercise_angina": "no",     # No chest pain (GOOD)
            "st_depression": "0.1",      # Minimal
            "slope": "0",               # Upsloping (GOOD)
            "vessels": "0",             # No blockages (GOOD)
            "thalassemia": "0"          # Normal
        }
        
        print("🧪 Testing young healthy female...")
        print(f"Input: {young_patient}")
        
        # Process input
        processed = api.preprocess_input(young_patient)
        if processed is None:
            print("❌ Preprocessing failed")
            return False
        
        print(f"✅ Processed shape: {processed.shape}")
        print(f"✅ Processed values: {processed[0]}")
        
        # Make prediction with each model
        print(f"\n📊 Individual model predictions:")
        for model_name, model_info in api.models.items():
            model = model_info['model']
            prob = model.predict_proba(processed)[0][1]
            pred = model.predict(processed)[0]
            print(f"   {model_name}: {prob:.3f} ({prob*100:.1f}%) - Prediction: {pred}")
        
        # Check if this is reasonable for a 30-year-old
        best_prob = api.best_model.predict_proba(processed)[0][1]
        print(f"\n🏆 Best model probability: {best_prob:.3f} ({best_prob*100:.1f}%)")
        
        if best_prob > 0.5:
            print("🚨 PROBLEM: Young healthy patient getting high risk!")
            print("   This indicates a serious preprocessing or model issue.")
            
            # Let's check the feature values step by step
            print(f"\n🔍 Detailed feature analysis:")
            for i, (feature_name, value) in enumerate(zip(api.feature_names, processed[0])):
                print(f"   {i:2d}. {feature_name:15s} = {value:8.3f}")
                
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_actual_data():
    """Test with actual low-risk cases from training data"""
    print("\n📋 TESTING WITH ACTUAL TRAINING DATA")
    print("="*50)
    
    try:
        # Load training data
        train_data = pd.read_csv('data/processed_train.csv')
        
        # Find actual low-risk cases (young people with target=0)
        young_low_risk = train_data[(train_data['age'] <= 0.5) & (train_data['target'] == 0)]  # Scaled age
        
        if len(young_low_risk) == 0:
            print("⚠️ No young low-risk cases found in training data")
            # Try all low-risk cases
            low_risk = train_data[train_data['target'] == 0].head(3)
        else:
            low_risk = young_low_risk.head(3)
        
        print(f"📊 Found {len(low_risk)} low-risk cases to test:")
        
        # Load model
        with open('models/trained_models.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        best_model = model_data['best_model']
        feature_names = model_data['feature_names']
        
        for idx, row in low_risk.iterrows():
            # Extract features (remove target)
            features = row[feature_names].values.reshape(1, -1)
            
            # Predict
            prob = best_model.predict_proba(features)[0][1]
            pred = best_model.predict(features)[0]
            
            print(f"   Case {idx}: target={row['target']}, predicted_prob={prob:.3f}, prediction={pred}")
            
            if prob > 0.5 and row['target'] == 0:
                print(f"   🚨 Model predicting high risk for actual low-risk case!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def suggest_specific_fixes():
    """Suggest specific fixes based on findings"""
    print("\n🔧 SUGGESTED FIXES")
    print("="*50)
    
    print("Based on the diagnosis, here are the most likely issues:")
    print()
    print("1. **DATA SCALING ISSUE**:")
    print("   - The features might not be properly scaled/standardized")
    print("   - Solution: Retrain models with proper scaling")
    print()
    print("2. **FEATURE ORDER MISMATCH**:")
    print("   - Preprocessing creates features in different order than training")
    print("   - Solution: Check feature_names order in model file")
    print()
    print("3. **POOR TRAINING DATA**:")
    print("   - Training data doesn't have clear patterns")
    print("   - Solution: Regenerate training data with better separation")
    print()
    print("4. **MODEL OVERFITTING**:")
    print("   - Models learned noise instead of real patterns")
    print("   - Solution: Retrain with better validation")
    print()
    print("🚀 **RECOMMENDED ACTION**:")
    print("   1. Run: python src/data_preprocessing.py")
    print("   2. Run: python src/train_models.py") 
    print("   3. Restart: python src/app.py")
    print("   4. Test again with young patient")

def main():
    """Run complete diagnosis"""
    print("🏥 HEART DISEASE MODEL - COMPLETE DIAGNOSIS")
    print("="*60)
    
    all_checks = [
        ("Training Data", check_training_data),
        ("Model File", check_model_file),
        ("Preprocessing", test_preprocessing_directly),
        ("Actual Data", test_with_actual_data)
    ]
    
    results = {}
    for check_name, check_func in all_checks:
        print(f"\n{'='*20} {check_name.upper()} {'='*20}")
        try:
            result = check_func()
            results[check_name] = result
        except Exception as e:
            print(f"❌ {check_name} failed: {e}")
            results[check_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 DIAGNOSIS SUMMARY")
    print("="*60)
    
    for check_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {check_name}: {status}")
    
    suggest_specific_fixes()

if __name__ == "__main__":
    main()