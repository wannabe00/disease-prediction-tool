# test_predictions.py - Automated testing script
import requests
import json
import time
from datetime import datetime

# Make sure your Flask app is running on localhost:5000
BASE_URL = "http://localhost:5000"

def test_prediction(patient_data, description):
    """Test a single patient prediction"""
    print(f"\n{'='*80}")
    print(f"🧪 TESTING: {description}")
    print(f"{'='*80}")
    
    try:
        # Make prediction request
        print("📡 Making prediction request...")
        pred_response = requests.post(
            f"{BASE_URL}/predict",
            json=patient_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if pred_response.status_code != 200:
            print(f"❌ Prediction failed: {pred_response.status_code}")
            print(f"Error: {pred_response.text}")
            return None
        
        prediction_result = pred_response.json()
        
        # Display prediction results
        print(f"🎯 PREDICTION RESULTS:")
        print(f"   Risk Level: {prediction_result.get('risk_level', 'N/A')}")
        print(f"   Probability: {prediction_result.get('probability', 0):.1%}")
        print(f"   Best Model: {prediction_result.get('best_model', 'N/A')}")
        print(f"   Confidence: {prediction_result.get('confidence', 0):.3f}")
        
        # Get AI recommendations
        print(f"\n🤖 Getting AI recommendations...")
        rec_response = requests.post(
            f"{BASE_URL}/recommendations",
            json={
                "prediction_result": prediction_result,
                "patient_data": patient_data
            },
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        
        if rec_response.status_code != 200:
            print(f"❌ Recommendations failed: {rec_response.status_code}")
            recommendations = {"recommendations": "Failed to get recommendations"}
        else:
            recommendations = rec_response.json()
        
        # Display recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   Source: {recommendations.get('source', 'Unknown')}")
        print(f"\n{recommendations.get('recommendations', 'No recommendations available')}")
        
        # Return summary for analysis
        return {
            "description": description,
            "patient_data": patient_data,
            "risk_level": prediction_result.get('risk_level'),
            "probability": prediction_result.get('probability'),
            "recommendations": recommendations.get('recommendations', ''),
            "source": recommendations.get('source', '')
        }
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return None
    except Exception as e:
        print(f"❌ Test error: {e}")
        return None

def run_comprehensive_tests():
    """Run tests on different patient profiles"""
    
    print("🏥 HEART DISEASE PREDICTION - COMPREHENSIVE TESTING")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define test patients
    test_patients = [
        {
            "data": {
                "age": "25",
                "sex": "female", 
                "chest_pain": "3",
                "blood_pressure": "105",
                "cholesterol": "175",
                "fasting_sugar": "no",
                "rest_ecg": "0",
                "max_heart_rate": "185",
                "exercise_angina": "no",
                "st_depression": "0.0",
                "slope": "0",
                "vessels": "0",
                "thalassemia": "0"
            },
            "description": "Young Healthy Female (Expected: Low Risk ~5-15%)"
        },
        {
            "data": {
                "age": "65",
                "sex": "male",
                "chest_pain": "0", 
                "blood_pressure": "160",
                "cholesterol": "280",
                "fasting_sugar": "yes",
                "rest_ecg": "1",
                "max_heart_rate": "120",
                "exercise_angina": "yes",
                "st_depression": "2.0",
                "slope": "2",
                "vessels": "2",
                "thalassemia": "2"
            },
            "description": "High Risk Elderly Male (Expected: High Risk ~70-85%)"
        },
        {
            "data": {
                "age": "45",
                "sex": "male",
                "chest_pain": "1",
                "blood_pressure": "140",
                "cholesterol": "220",
                "fasting_sugar": "no",
                "rest_ecg": "0",
                "max_heart_rate": "150",
                "exercise_angina": "no",
                "st_depression": "1.0",
                "slope": "1",
                "vessels": "1",
                "thalassemia": "1"
            },
            "description": "Middle-Aged Male Moderate Risk (Expected: Moderate Risk ~40-60%)"
        },
        {
            "data": {
                "age": "55",
                "sex": "female",
                "chest_pain": "2",
                "blood_pressure": "135",
                "cholesterol": "240",
                "fasting_sugar": "no",
                "rest_ecg": "0",
                "max_heart_rate": "140",
                "exercise_angina": "yes",
                "st_depression": "1.5",
                "slope": "1",
                "vessels": "1",
                "thalassemia": "1"
            },
            "description": "Post-Menopausal Female (Expected: Moderate-High Risk ~50-70%)"
        },
        {
            "data": {
                "age": "35",
                "sex": "male",
                "chest_pain": "3",
                "blood_pressure": "125",
                "cholesterol": "190",
                "fasting_sugar": "no",
                "rest_ecg": "0",
                "max_heart_rate": "175",
                "exercise_angina": "no",
                "st_depression": "0.5",
                "slope": "0",
                "vessels": "0",
                "thalassemia": "0"
            },
            "description": "Young Male Good Health (Expected: Low-Moderate Risk ~15-35%)"
        }
    ]
    
    # Run all tests
    results = []
    for i, patient in enumerate(test_patients, 1):
        print(f"\n🔄 Test {i}/{len(test_patients)}")
        result = test_prediction(patient["data"], patient["description"])
        if result:
            results.append(result)
        time.sleep(1)  # Brief pause between tests
    
    # Summary analysis
    print(f"\n{'='*80}")
    print("📊 TEST SUMMARY & ANALYSIS")
    print(f"{'='*80}")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['description']}")
        print(f"   Risk: {result['risk_level']} ({result['probability']:.1%})")
        print(f"   Age: {result['patient_data']['age']}, Sex: {result['patient_data']['sex']}")
        print(f"   BP: {result['patient_data']['blood_pressure']}, Chol: {result['patient_data']['cholesterol']}")
        
        # Check if recommendations mention specific values
        rec = result['recommendations']
        mentions_age = result['patient_data']['age'] in rec
        mentions_bp = result['patient_data']['blood_pressure'] in rec
        mentions_chol = result['patient_data']['cholesterol'] in rec
        
        print(f"   Personalization: Age✓ BP✓ Chol✓" if mentions_age and mentions_bp and mentions_chol 
              else f"   Personalization: Age{'✓' if mentions_age else '✗'} BP{'✓' if mentions_bp else '✗'} Chol{'✓' if mentions_chol else '✗'}")
    
    print(f"\n🎯 COPY THIS SUMMARY AND SEND TO ME:")
    print(f"{'='*50}")
    for result in results:
        print(f"{result['description']}: {result['probability']:.1%} risk")
        # Show first 100 chars of recommendations for quick analysis
        rec_preview = result['recommendations'][:100].replace('\n', ' ') + "..."
        print(f"   Rec: {rec_preview}")
    print(f"{'='*50}")
    
    return results

def test_specific_patient(age, sex, bp, chol, max_hr, chest_pain="3", exercise_angina="no"):
    """Quick test for specific patient parameters"""
    patient_data = {
        "age": str(age),
        "sex": sex,
        "chest_pain": chest_pain,
        "blood_pressure": str(bp),
        "cholesterol": str(chol),
        "fasting_sugar": "no",
        "rest_ecg": "0",
        "max_heart_rate": str(max_hr),
        "exercise_angina": exercise_angina,
        "st_depression": "0.5",
        "slope": "1",
        "vessels": "0",
        "thalassemia": "1"
    }
    
    description = f"{age}yr {sex}, BP:{bp}, Chol:{chol}, HR:{max_hr}"
    return test_prediction(patient_data, description)

if __name__ == "__main__":
    print("🚀 Starting automated tests...")
    print("⚠️  Make sure your Flask app is running: python src/app.py")
    
    try:
        # Test if server is running
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running, starting tests...")
            results = run_comprehensive_tests()
        else:
            print("❌ Server not responding properly")
    except requests.exceptions.RequestException:
        print("❌ Server not running! Start it with: python src/app.py")
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")