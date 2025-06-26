# train_models.py - Member 2: Shota Tsertsvadze (FIXED VERSION)
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
            # Load processed data
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
            print(f"✅ Training samples: {len(self.X_train)}")
            print(f"✅ Test samples: {len(self.X_test)}")

            return True

        except FileNotFoundError:
            print("❌ Processed data not found!")
            print("Please run 'python src/data_preprocessing.py' first.")
            return False

    def train_logistic_regression(self):
        """Train Logistic Regression model with hyperparameter tuning"""
        print("\n🔄 Training Logistic Regression...")

        # Hyperparameter tuning
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs'],
            'max_iter': [1000, 2000]
        }

        lr = LogisticRegression(random_state=42)
        grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        best_lr = grid_search.best_estimator_

        # Cross-validation
        cv_scores = cross_val_score(best_lr, self.X_train, self.y_train,
                                    cv=5, scoring='roc_auc')

        self.models['Logistic Regression'] = {
            'model': best_lr,
            'cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': grid_search.best_params_
        }

        print(f"✅ Logistic Regression CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"   Best params: {grid_search.best_params_}")

    def train_random_forest(self):
        """Train Random Forest model with hyperparameter tuning"""
        print("\n🔄 Training Random Forest...")

        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        best_rf = grid_search.best_estimator_

        # Cross-validation
        cv_scores = cross_val_score(best_rf, self.X_train, self.y_train,
                                    cv=5, scoring='roc_auc')

        self.models['Random Forest'] = {
            'model': best_rf,
            'cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': grid_search.best_params_
        }

        print(f"✅ Random Forest CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"   Best params: {grid_search.best_params_}")

    def train_svm(self):
        """Train Support Vector Machine model with hyperparameter tuning"""
        print("\n🔄 Training SVM...")

        # Hyperparameter tuning
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto', 0.001, 0.01]
        }

        svm = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        best_svm = grid_search.best_estimator_

        # Cross-validation
        cv_scores = cross_val_score(best_svm, self.X_train, self.y_train,
                                    cv=5, scoring='roc_auc')

        self.models['SVM'] = {
            'model': best_svm,
            'cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': grid_search.best_params_
        }

        print(f"✅ SVM CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"   Best params: {grid_search.best_params_}")

    def evaluate_models(self):
        """Evaluate all models and select the best one"""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)

        results = []
        model_predictions = {}

        for name, model_info in self.models.items():
            model = model_info['model']

            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]

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
            print(f"   Accuracy: {accuracy:.3f}")
            print(f"   Precision: {precision:.3f}")
            print(f"   Recall: {recall:.3f}")
            print(f"   F1-Score: {f1:.3f}")

            print(f"\n   Classification Report:")
            print(classification_report(self.y_test, y_pred,
                                        target_names=['No Disease', 'Disease'],
                                        digits=3))

            # Update best model
            if auc_score > self.best_score:
                self.best_score = auc_score
                self.best_model = model
                self.best_model_name = name

        # Create results DataFrame and display
        results_df = pd.DataFrame(results)
        print(f"\n📋 SUMMARY TABLE:")
        print("-" * 80)
        print(results_df.to_string(index=False))
        print("-" * 80)

        print(f"\n🏆 Best Model: {self.best_model_name} (Test AUC: {self.best_score:.3f})")

        return results_df, model_predictions

    def plot_roc_curves(self, model_predictions):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(12, 8))

        colors = ['blue', 'red', 'green', 'purple', 'orange']

        for i, (name, pred_info) in enumerate(model_predictions.items()):
            fpr, tpr, _ = roc_curve(self.y_test, pred_info['y_pred_proba'])
            auc = pred_info['auc']

            plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.3f})',
                     color=colors[i % len(colors)])

        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC = 0.5)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison - Heart Disease Prediction', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)

        # Add best model annotation
        plt.text(0.6, 0.2, f'Best Model: {self.best_model_name}\nAUC: {self.best_score:.3f}',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                 fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✅ ROC curves saved as 'roc_curves.png'")

    def plot_confusion_matrices(self, model_predictions):
        """Plot confusion matrices for all models"""
        n_models = len(model_predictions)
        fig, axes = plt.subplots(1, n_models, figsize=(15, 4))
        if n_models == 1:
            axes = [axes]

        for i, (name, pred_info) in enumerate(model_predictions.items()):
            cm = confusion_matrix(self.y_test, pred_info['y_pred'])

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                        xticklabels=['No Disease', 'Disease'],
                        yticklabels=['No Disease', 'Disease'])
            axes[i].set_title(f'{name}\nAUC: {pred_info["auc"]:.3f}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')

        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✅ Confusion matrices saved as 'confusion_matrices.png'")

    def get_feature_importance(self):
        """Get feature importance from the best model"""
        if self.best_model is None:
            return None

        if self.best_model_name == 'Random Forest':
            importances = self.best_model.feature_importances_
            feature_imp = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            print(f"\n🔍 TOP 10 MOST IMPORTANT FEATURES ({self.best_model_name}):")
            print("-" * 50)
            print(feature_imp.head(10).to_string(index=False))

            # Plot feature importance
            plt.figure(figsize=(12, 8))
            top_features = feature_imp.head(10)
            bars = plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 10 Feature Importance - {self.best_model_name}', fontweight='bold')
            plt.gca().invert_yaxis()

            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height() / 2,
                         f'{width:.3f}', ha='left', va='center')

            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()

            print("✅ Feature importance plot saved as 'feature_importance.png'")
            return feature_imp

        elif self.best_model_name == 'Logistic Regression':
            coefficients = self.best_model.coef_[0]
            feature_imp = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': coefficients,
                'abs_coefficient': np.abs(coefficients)
            }).sort_values('abs_coefficient', ascending=False)

            print(f"\n🔍 TOP 10 MOST IMPORTANT FEATURES ({self.best_model_name}):")
            print("-" * 60)
            print(feature_imp[['feature', 'coefficient']].head(10).to_string(index=False))

            return feature_imp
        else:
            print(f"Feature importance not available for {self.best_model_name}")
            return None

    def save_models(self):
        """Save all trained models and metadata"""
        # Ensure models directory exists
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

    def generate_model_report(self, results_df):
        """Generate a comprehensive model report"""
        report = f"""

HEART DISEASE PREDICTION MODEL REPORT
{'=' * 60}

DATASET SUMMARY:
   Training samples: {len(self.X_train)}
   Test samples: {len(self.X_test)}
   Features: {len(self.feature_names)}

MODEL PERFORMANCE:
{results_df.to_string(index=False)}

BEST MODEL: {self.best_model_name}
   Test AUC: {self.best_score:.3f}

MODEL INTERPRETABILITY:
   - Random Forest: Provides feature importance scores
   - Logistic Regression: Provides coefficient interpretation
   - SVM: Black-box model with high accuracy

DEPLOYMENT READY:
   - Models saved and serialized
   - Feature preprocessing pipeline included
   - REST API integration available

⚠️  DISCLAIMER:
   This model is for educational purposes only.
   Always consult healthcare professionals for medical decisions.

{'=' * 60}
        """

        print(report)

        # Save report to file
        with open('model_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)

        print("📄 Model report saved as 'model_report.txt'")


def main():
    """Main training pipeline"""
    print("🏥 HEART DISEASE PREDICTION - MODEL TRAINING")
    print("=" * 60)

    # Initialize predictor
    predictor = HeartDiseasePredictor()

    # Load data
    if not predictor.load_data():
        return

    # Train all models
    predictor.train_logistic_regression()
    predictor.train_random_forest()
    predictor.train_svm()

    # Evaluate models
    results_df, model_predictions = predictor.evaluate_models()

    # Create visualizations
    predictor.plot_roc_curves(model_predictions)
    predictor.plot_confusion_matrices(model_predictions)
    feature_imp = predictor.get_feature_importance()

    # Save models
    predictor.save_models()

    # Generate report
    predictor.generate_model_report(results_df)

    print("\n" + "=" * 60)
    print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("📁 Files created:")
    print("   - models/trained_models.pkl")
    print("   - roc_curves.png")
    print("   - confusion_matrices.png")
    print("   - feature_importance.png (if Random Forest is best)")
    print("   - model_report.txt")
    print("\n🚀 Next step: Run 'python src/app.py' to start the web application!")


if __name__ == "__main__":
    main()