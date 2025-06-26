# data_preprocessing.py - FIXED VERSION with better data generation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

def create_realistic_heart_data():
    """Create realistic heart disease dataset with clear patterns"""
    print("📊 Creating realistic heart disease dataset...")
    
    np.random.seed(42)
    n_samples = 303
    
    # Create correlated features that make medical sense
    data = []
    
    for i in range(n_samples):
        # Start with random base risk
        base_risk = np.random.uniform(0, 1)
        
        # Generate age with bias toward higher risk for older patients
        if base_risk > 0.7:  # High risk patients tend to be older
            age = np.random.normal(60, 8)
        elif base_risk > 0.4:  # Moderate risk
            age = np.random.normal(50, 10)
        else:  # Low risk patients tend to be younger
            age = np.random.normal(40, 8)
        
        age = int(np.clip(age, 29, 77))
        
        # Sex - males have higher baseline risk
        if base_risk > 0.6:
            sex = np.random.choice([0, 1], p=[0.2, 0.8])  # More males in high risk
        else:
            sex = np.random.choice([0, 1], p=[0.6, 0.4])  # More females in low risk
        
        # Chest pain - typical angina is highest risk
        if base_risk > 0.7:
            cp = np.random.choice([0, 1, 2, 3], p=[0.6, 0.2, 0.15, 0.05])  # More typical angina
        elif base_risk > 0.4:
            cp = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2])
        else:
            cp = np.random.choice([0, 1, 2, 3], p=[0.1, 0.1, 0.3, 0.5])  # More asymptomatic
        
        # Blood pressure - correlated with age and risk
        bp_base = 120 + (age - 50) * 0.8 + base_risk * 40
        trestbps = int(np.clip(np.random.normal(bp_base, 15), 94, 200))
        
        # Cholesterol - correlated with age and risk
        chol_base = 200 + (age - 50) * 1.5 + base_risk * 60
        chol = int(np.clip(np.random.normal(chol_base, 40), 126, 564))
        
        # Fasting blood sugar
        fbs = 1 if (base_risk > 0.6 and np.random.random() < 0.3) else 0
        
        # Rest ECG
        if base_risk > 0.6:
            restecg = np.random.choice([0, 1, 2], p=[0.3, 0.6, 0.1])
        else:
            restecg = np.random.choice([0, 1, 2], p=[0.7, 0.3, 0.0])
        
        # Max heart rate - inversely correlated with risk and age
        hr_base = 220 - age - base_risk * 30
        thalach = int(np.clip(np.random.normal(hr_base, 20), 71, 202))
        
        # Exercise angina - higher in high risk
        exang = 1 if (base_risk > 0.5 and np.random.random() < 0.6) else 0
        
        # ST depression - higher in high risk
        if base_risk > 0.6:
            oldpeak = np.random.exponential(2.0)
        else:
            oldpeak = np.random.exponential(0.5)
        oldpeak = np.clip(oldpeak, 0, 6.2)
        
        # Slope - downsloping is worse
        if base_risk > 0.6:
            slope = np.random.choice([0, 1, 2], p=[0.1, 0.4, 0.5])
        else:
            slope = np.random.choice([0, 1, 2], p=[0.5, 0.4, 0.1])
        
        # Number of vessels - more vessels blocked = higher risk
        if base_risk > 0.7:
            ca = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2])
        elif base_risk > 0.4:
            ca = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
        else:
            ca = np.random.choice([0, 1, 2, 3], p=[0.8, 0.15, 0.04, 0.01])
        
        # Thalassemia - reversible defect is high risk
        if base_risk > 0.6:
            thal = np.random.choice([0, 1, 2, 3], p=[0.0, 0.1, 0.7, 0.2])
        else:
            thal = np.random.choice([0, 1, 2, 3], p=[0.1, 0.1, 0.3, 0.5])
        
        # Calculate actual risk based on medical factors
        risk_factors = [
            age > 55,           # Age risk
            sex == 1,           # Male
            cp <= 1,            # Typical/atypical angina
            trestbps > 140,     # High BP
            chol > 240,         # High cholesterol
            fbs == 1,           # High blood sugar
            thalach < 120,      # Low max heart rate
            exang == 1,         # Exercise angina
            oldpeak > 1.5,      # High ST depression
            slope == 2,         # Downsloping
            ca > 0,             # Blocked vessels
            thal == 2           # Reversible defect
        ]
        
        risk_score = sum(risk_factors)
        
        # Convert to probability with some noise
        probability = 1 / (1 + np.exp(-(risk_score - 4 + np.random.normal(0, 1))))
        target = 1 if probability > 0.5 else 0
        
        data.append({
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal,
            'target': target
        })
    
    df = pd.DataFrame(data)
    
    # Ensure we have both classes
    if df['target'].nunique() < 2:
        print("⚠️ Fixing target distribution...")
        # Force some high-risk cases to be positive
        high_risk_indices = df.nlargest(50, ['age'])['index'] if 'index' in df.columns else df.nlargest(50, 'age').index
        df.loc[high_risk_indices[:25], 'target'] = 1
        
        # Force some low-risk cases to be negative
        low_risk_indices = df.nsmallest(50, ['age'])['index'] if 'index' in df.columns else df.nsmallest(50, 'age').index
        df.loc[low_risk_indices[:25], 'target'] = 0
    
    print(f"✅ Created dataset with {len(df)} samples")
    print(f"📊 Target distribution: {df['target'].value_counts().to_dict()}")
    print(f"📈 Target mean: {df['target'].mean():.3f}")
    
    return df

def load_heart_disease_data():
    """Load heart disease dataset from CSV file or create sample data if not found"""
    
    # First, try to load real UCI Heart Disease dataset
    if os.path.exists('data/heart.csv'):
        print("📂 Loading existing heart.csv...")
        df = pd.read_csv('data/heart.csv')
        print(f"✅ Loaded dataset with {len(df)} samples")
        
        # Check if target distribution is reasonable
        target_dist = df['target'].value_counts()
        if len(target_dist) >= 2 and target_dist.min() > 10:
            print(f"📊 Good target distribution: {target_dist.to_dict()}")
            return df
        else:
            print("⚠️ Poor target distribution, creating new dataset...")
    
    # Create new realistic dataset
    print("🔄 Creating new realistic heart disease dataset...")
    df = create_realistic_heart_data()
    
    # Save for future use
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/heart.csv', index=False)
    print("💾 Saved as data/heart.csv")
    
    return df

def preprocess_data():
    """Complete data preprocessing pipeline"""
    print("=" * 50)
    print("HEART DISEASE DATA PREPROCESSING")
    print("=" * 50)
    
    # Load data
    df = load_heart_disease_data()
    
    # Basic info
    print(f"\nDataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    
    # Feature Engineering
    print("\n🔧 Creating engineered features...")
    
    # Age groups
    df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 40, 55, 70, 100], 
                            labels=[0, 1, 2, 3]).astype(int)
    
    # Risk factors
    df['chol_risk'] = (df['chol'] > 240).astype(int)
    df['bp_risk'] = (df['trestbps'] > 140).astype(int)
    df['hr_risk'] = (df['thalach'] < 120).astype(int)
    
    # Combined risk score
    df['risk_score'] = (
        (df['age'] > 55).astype(int) + 
        df['sex'] + 
        (df['cp'] <= 1).astype(int) + 
        df['chol_risk'] + 
        df['bp_risk'] + 
        df['exang'] + 
        (df['oldpeak'] > 1).astype(int)
    )
    
    # Feature selection
    feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
                   'age_group', 'chol_risk', 'bp_risk', 'hr_risk', 'risk_score']
    
    X = df[feature_cols]
    y = df['target']
    
    print(f"✅ Feature columns: {len(feature_cols)}")
    print(f"📊 Feature variance check:")
    for col in feature_cols:
        variance = X[col].var()
        unique_vals = X[col].nunique()
        print(f"   {col}: var={variance:.4f}, unique={unique_vals}")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)
    
    # Reset indices
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    print(f"\n✅ Training set: {X_train_scaled.shape}")
    print(f"✅ Test set: {X_test_scaled.shape}")
    print(f"📊 Training target distribution: {y_train.value_counts().to_dict()}")
    print(f"📊 Test target distribution: {y_test.value_counts().to_dict()}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols, df

def create_eda_plots(df):
    """Create comprehensive EDA visualizations"""
    print("\n📊 Creating EDA visualizations...")
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Heart Disease Dataset - Key Analysis', fontsize=16, fontweight='bold')
    
    # 1. Target distribution
    target_counts = df['target'].value_counts()
    axes[0, 0].bar(['No Disease', 'Disease'], target_counts.values, 
                   color=['lightgreen', 'lightcoral'], alpha=0.7)
    axes[0, 0].set_title('Heart Disease Distribution')
    axes[0, 0].set_ylabel('Count')
    for i, v in enumerate(target_counts.values):
        axes[0, 0].text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    # 2. Age vs Target
    sns.boxplot(data=df, x='target', y='age', ax=axes[0, 1])
    axes[0, 1].set_title('Age Distribution by Heart Disease')
    axes[0, 1].set_xticklabels(['No Disease', 'Disease'])
    
    # 3. Key risk factors
    risk_data = pd.DataFrame({
        'High BP': (df['trestbps'] > 140).groupby(df['target']).mean(),
        'High Cholesterol': (df['chol'] > 240).groupby(df['target']).mean(),
        'Exercise Angina': df['exang'].groupby(df['target']).mean(),
        'Male': df['sex'].groupby(df['target']).mean()
    })
    
    risk_data.plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Risk Factors by Heart Disease')
    axes[1, 0].set_xlabel('Heart Disease (0=No, 1=Yes)')
    axes[1, 0].set_ylabel('Proportion')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].tick_params(axis='x', rotation=0)
    
    # 4. Correlation with target
    correlations = df.corr()['target'].abs().sort_values(ascending=True)
    correlations = correlations[correlations.index != 'target'][-10:]  # Top 10
    
    correlations.plot(kind='barh', ax=axes[1, 1])
    axes[1, 1].set_title('Top 10 Feature Correlations with Target')
    axes[1, 1].set_xlabel('Absolute Correlation')
    
    plt.tight_layout()
    plt.savefig('eda_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ EDA plots saved as 'eda_plots.png'")

def save_processed_data(X_train, X_test, y_train, y_test):
    """Save processed data to CSV files"""
    print("\n💾 Saving processed data...")
    
    os.makedirs('data', exist_ok=True)
    
    # Combine features and target
    train_data = X_train.copy()
    train_data['target'] = y_train
    
    test_data = X_test.copy()
    test_data['target'] = y_test
    
    # Save to CSV
    train_data.to_csv('data/processed_train.csv', index=False)
    test_data.to_csv('data/processed_test.csv', index=False)
    
    print(f"✅ Saved training data: {train_data.shape}")
    print(f"✅ Saved test data: {test_data.shape}")

if __name__ == "__main__":
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Run preprocessing pipeline
    X_train, X_test, y_train, y_test, scaler, features, df = preprocess_data()
    
    # Create visualizations
    create_eda_plots(df)
    
    # Save processed data
    save_processed_data(X_train, X_test, y_train, y_test)
    
    print("\n" + "=" * 50)
    print("✅ DATA PREPROCESSING COMPLETED!")
    print("=" * 50)
    print(f"📁 Files created:")
    print("   - data/heart.csv")
    print("   - data/processed_train.csv")
    print("   - data/processed_test.csv")
    print("   - eda_plots.png")
    print("\n🚀 Next step: Run 'python src/train_models.py'")