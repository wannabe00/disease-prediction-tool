# train_models.py - FIXED VERSION to prevent 100% risk predictions
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, accuracy_score, 
                           precision_score, recall_score, f1_score)
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class HeartDiseasePredictor:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        self.feature_names = None
        self.scaler = None
        
    def load_data(self):
        """Load preprocessed training and test data"""
        try:
            train_data = pd.read_csv('data/processed_train.csv')
            test_data = pd.read_csv('data/processed_test.csv')
            
            print(f"✅ Training data loaded: {train_data.shape}")
            print(f"✅ Test data loaded: {test_data.shape}")
            
            # Separate features and target
            feature_cols = [col for col in train_data.columns if col != 'target']
            
            self.X_train = train_data[feature_cols]
            self.y_train = train_data['target']
            self.X_test = test_data[feature_cols]
            self.y_test = test_data['target']
            self.feature_names = feature_cols
            
            print(f"✅ Features: {len(feature_cols)}")
            print(f"📊 Training target distribution: {self.y_train.value_counts().to_dict()}")
            
            return True
            
        except FileNotFoundError:
            print("❌ Processed data not found!")
            print("Please run 'python src/data_preprocessing.py' first.")
            return False
    
    def train_logistic_regression(self):
        """Train Logistic Regression with CONSERVATIVE parameters"""
        print("\n🔄 Training Logistic Regression...")
        
        # Use SIMPLE parameters to avoid overfitting
        lr = LogisticRegression(
            C=0.1,  # Lower C for less overfitting
            solver='liblinear',
            random_state=42,
            max_iter=1000
        )
        
        lr.fit(self.X_train, self.y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(lr, self.X_train, self.y_train, 
                                   cv=5, scoring='roc_auc')
        
        # Test predictions to check sanity
        test_pred = lr.predict_proba(self.X_test)[:, 1]
        print(f"   Test predictions range: {test_pred.min():.3f} to {test_pred.max():.3f}")
        print(f"   Test predictions mean: {test_pred.mean():.3f}")
        
        self.models['Logistic Regression'] = {
            'model': lr,
            'cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': {'C': 0.1, 'solver': 'liblinear'}
        }
        
        print(f"✅ Logistic Regression CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
    def train_random_forest(self):
        """Train Random Forest with CONSERVATIVE parameters"""
        print("\n🔄 Training Random Forest...")
        
        # Use SIMPLE parameters
        rf = RandomForestClassifier(
            n_estimators=50,  # Fewer trees
            max_depth=5,      # Shallow trees
            min_samples_split=10,  # More samples required to split
            min_samples_leaf=5,    # More samples in leaves
            random_state=42
        )
        
        rf.fit(self.X_train, self.y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(rf, self.X_train, self.y_train, 
                                   cv=5, scoring='roc_auc')
        
        # Test predictions to check sanity
        test_pred = rf.predict_proba(self.X_test)[:, 1]
        print(f"   Test predictions range: {test_pred.min():.3f} to {test_pred.max():.3f}")
        print(f"   Test predictions mean: {test_pred.mean():.3f}")
        
        self.models['Random Forest'] = {
            'model': rf,
            'cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': {'n_estimators': 50, 'max_depth': 5}
        }
        
        print(f"✅ Random Forest CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
    def train_svm(self):
        """Train SVM with CONSERVATIVE parameters"""
        print("\n🔄 Training SVM...")
        
        # Use SIMPLE parameters
        svm = SVC(
            C=0.1,  # Lower C
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        svm.fit(self.X_train, self.y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(svm, self.X_train, self.y_train, 
                                   cv=5, scoring='roc_auc')
        
        # Test predictions to check sanity
        test_pred = svm.predict_proba(self.X_test)[:, 1]
        print(f"   Test predictions range: {test_pred.min():.3f} to {test_pred.max():.3f}")
        print(f"   Test predictions mean: {test_pred.mean():.3f}")
        
        self.models['SVM'] = {
            'model': svm,
            'cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': {'C': 0.1, 'kernel': 'rbf'}
        }
        
        print(f"✅ SVM CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    
    def evaluate_models(self):
        """Evaluate all models and select the best one"""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        results = []
        model_predictions = {}
        
        for name, model_info in self.models.items():
            model = model_info['model']
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # SANITY CHECK - if all predictions are the same, something is wrong
            if len(np.unique(y_pred_proba)) < 3:
                print(f"⚠️ WARNING: {name} gives very similar predictions!")
                print(f"   Unique probabilities: {np.unique(y_pred_proba)}")
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            
            # Store predictions for ROC curve
            model_predictions[name] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'auc': auc_score
            }
            
            results.append({
                'Model': name,
                'CV AUC': f"{model_info['cv_score']:.3f}",
                'Test AUC': f"{auc_score:.3f}",
                'Accuracy': f"{accuracy:.3f}",
                'Precision': f"{precision:.3f}",
                'Recall': f"{recall:.3f}",
                'F1-Score': f"{f1:.3f}"
            })
            
            print(f"\n📊 {name}:")
            print(f"   CV AUC: {model_info['cv_score']:.3f} ± {model_info['cv_std']:.3f}")
            print(f"   Test AUC: {auc_score:.3f}")
            print(f"   Prediction range: {y_pred_proba.min():.3f} to {y_pred_proba.max():.3f}")
            
            # Update best model
            if auc_score > self.best_score:
                self.best_score = auc_score
                self.best_model = model
                self.best_model_name = name
        
        # Final sanity check
        print(f"\n🧪 FINAL SANITY CHECK:")
        best_model_probs = self.best_model.predict_proba(self.X_test)[:, 1]
        print(f"   Best model ({self.best_model_name}) predictions:")
        print(f"   Min: {best_model_probs.min():.3f}")
        print(f"   Max: {best_model_probs.max():.3f}")
        print(f"   Mean: {best_model_probs.mean():.3f}")
        print(f"   Std: {best_model_probs.std():.3f}")
        
        if best_model_probs.std() < 0.1:
            print(f"⚠️ WARNING: Very low prediction variance - model may be broken!")
        
        # Create results DataFrame and display
        results_df = pd.DataFrame(results)
        print(f"\n📋 SUMMARY TABLE:")
        print("-" * 80)
        print(results_df.to_string(index=False))
        print("-" * 80)
        
        print(f"\n🏆 Best Model: {self.best_model_name} (Test AUC: {self.best_score:.3f})")
        
        return results_df, model_predictions
    
    def save_models(self):
        """Save all trained models and metadata"""
        os.makedirs('models', exist_ok=True)
        
        # Prepare data to save
        model_data = {
            'models': self.models,
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'best_score': self.best_score,
            'feature_names': self.feature_names
        }
        
        # Save using pickle
        with open('models/trained_models.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 Models saved to 'models/trained_models.pkl'")
        print(f"   Best model: {self.best_model_name}")
        print(f"   Best AUC score: {self.best_score:.3f}")
        print(f"   Number of features: {len(self.feature_names)}")

def main():
    """Main training pipeline"""
    print("🏥 HEART DISEASE PREDICTION - FIXED MODEL TRAINING")
    print("="*60)
    
    # Initialize predictor
    predictor = HeartDiseasePredictor()
    
    # Load data
    if not predictor.load_data():
        return
    
    # Train all models with conservative parameters
    predictor.train_logistic_regression()
    predictor.train_random_forest()
    predictor.train_svm()
    
    # Evaluate models
    results_df, model_predictions = predictor.evaluate_models()
    
    # Save models
    predictor.save_models()
    
    print("\n" + "="*60)
    print("🎉 FIXED MODEL TRAINING COMPLETED!")
    print("="*60)
    print("📁 Files created:")
    print("   - models/trained_models.pkl")
    print("\n🚀 Next step: Restart your Flask app!")
    print("   python src/app.py")

if __name__ == "__main__":
    main()