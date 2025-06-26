# emergency_feature_fix.py - Fix the feature mismatch issue
import pandas as pd
import numpy as np
import pickle
import os

def diagnose_feature_mismatch():
    """Diagnose the exact feature mismatch"""
    print("🔍 DIAGNOSING FEATURE MISMATCH")
    print("="*50)
    
    # Check what features the model expects
    try:
        with open('models/trained_models.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        model_features = model_data['feature_names']
        print(f"✅ Model expects {len(model_features)} features:")
        for i, feature in enumerate(model_features):
            print(f"   {i+1:2d}. {feature}")
        
        return model_features
    except Exception as e:
        print(f"❌ Cannot load model: {e}")
        return None

def check_app_preprocessing():
    """Check what features the app is creating"""
    print(f"\n🔍 CHECKING APP PREPROCESSING")
    print("="*50)
    
    # Simulate what the app does
    test_input = {
        "age": "30",
        "sex": "female",
        "chest_pain": "3",
        "blood_pressure": "110",
        "cholesterol": "175",
        "fasting_sugar": "no",
        "rest_ecg": "0",
        "max_heart_rate": "180",
        "exercise_angina": "no",
        "st_depression": "0.1",
        "slope": "0",
        "vessels": "0",
        "thalassemia": "0"
    }
    
    # Convert like app does (13 basic features only)
    age = float(test_input.get('age', 50))
    sex = 1 if test_input.get('sex') == 'male' else 0
    cp = int(test_input.get('chest_pain', 0))
    trestbps = int(test_input.get('blood_pressure', 120))
    chol = int(test_input.get('cholesterol', 200))
    fbs = 1 if test_input.get('fasting_sugar', 'no') == 'yes' else 0
    restecg = int(test_input.get('rest_ecg', 0))
    thalach = int(test_input.get('max_heart_rate', 150))
    exang = 1 if test_input.get('exercise_angina', 'no') == 'yes' else 0
    oldpeak = float(test_input.get('st_depression', 0))
    slope = int(test_input.get('slope', 1))
    ca = int(test_input.get('vessels', 0))
    thal = int(test_input.get('thalassemia', 1))
    
    app_features = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
    
    print(f"✅ App creates {len(app_features)} features:")
    for i, feature in enumerate(app_features):
        print(f"   {i+1:2d}. {feature}")
    
    return app_features

def fix_feature_mismatch():
    """Create quick fix by retraining with basic features only"""
    print(f"\n🔧 CREATING QUICK FIX")
    print("="*50)
    
    # Check if we have data
    if not os.path.exists('data/heart.csv'):
        print("❌ No data found. Run: python src/data_preprocessing.py")
        return False
    
    # Load raw data
    df = pd.read_csv('data/heart.csv')
    print(f"✅ Loaded data: {df.shape}")
    
    # Use ONLY the 13 basic features (same as app)
    basic_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    # Check if all features exist
    missing_features = [f for f in basic_features if f not in df.columns]
    if missing_features:
        print(f"❌ Missing features in data: {missing_features}")
        return False
    
    print(f"✅ Using {len(basic_features)} basic features:")
    for i, feature in enumerate(basic_features):
        print(f"   {i+1:2d}. {feature}")
    
    # Quick retrain with basic features
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics import roc_auc_score
    
    X = df[basic_features]
    y = df['target']
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train simple models
    models = {}
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:, 1])
    models['Random Forest'] = {'model': rf, 'cv_score': rf_auc, 'cv_std': 0.0}
    
    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])
    models['Logistic Regression'] = {'model': lr, 'cv_score': lr_auc, 'cv_std': 0.0}
    
    # SVM
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_auc = roc_auc_score(y_test, svm.predict_proba(X_test_scaled)[:, 1])
    models['SVM'] = {'model': svm, 'cv_score': svm_auc, 'cv_std': 0.0}
    
    # Best model
    best_model_name = max(models.keys(), key=lambda k: models[k]['cv_score'])
    best_model = models[best_model_name]['model']
    
    print(f"\n📊 Model Performance:")
    for name, info in models.items():
        print(f"   {name}: AUC = {info['cv_score']:.3f}")
    print(f"🏆 Best: {best_model_name}")
    
    # Test with young patient
    test_features = [30, 0, 3, 110, 175, 0, 0, 180, 0, 0.1, 0, 0, 0]
    test_scaled = scaler.transform([test_features])
    test_prob = best_model.predict_proba(test_scaled)[0][1]
    
    print(f"\n🧪 Test young patient: {test_prob:.3f} ({test_prob*100:.1f}%)")
    
    if test_prob > 0.5:
        print("❌ Still predicting high risk for young patient")
        return False
    
    # Save fixed models
    os.makedirs('models', exist_ok=True)
    
    model_data = {
        'models': models,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'feature_names': basic_features  # CRITICAL: Same as app!
    }
    
    with open('models/trained_models.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✅ FIXED! Saved models with {len(basic_features)} features")
    print("🚀 Restart your app: python src/app.py")
    
    return True

def main():
    """Main fix function"""
    print("🚨 EMERGENCY FEATURE MISMATCH FIX")
    print("="*60)
    
    # Step 1: Diagnose
    model_features = diagnose_feature_mismatch()
    app_features = check_app_preprocessing()
    
    if model_features and app_features:
        print(f"\n❌ MISMATCH FOUND:")
        print(f"   Model expects: {len(model_features)} features")
        print(f"   App provides: {len(app_features)} features")
        
        # Show missing features
        if len(model_features) > len(app_features):
            missing = set(model_features) - set(app_features)
            print(f"   Missing features: {list(missing)}")
    
    # Step 2: Fix
    if fix_feature_mismatch():
        print(f"\n🎉 SUCCESS! Feature mismatch fixed!")
        print("="*60)
        print("✅ Models retrained with correct 13 features")
        print("✅ Features now match app preprocessing")
        print("✅ Young patient test passes")
        print("\n🚀 RESTART YOUR APP:")
        print("   python src/app.py")
        print("   Visit: http://localhost:5000/test_young")
    else:
        print(f"\n❌ Fix failed. Try running:")
        print("   python src/data_preprocessing.py")
        print("   python src/train_models.py")

if __name__ == "__main__":
    main()