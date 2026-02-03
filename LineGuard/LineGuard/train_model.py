"""
Train the Fixed Fire Risk Assessment Model

This script demonstrates the CORRECTED approach:
1. ✅ Temporal splitting (no data leakage)
2. ✅ Unified model (no classifier/regressor disagreement)
3. ✅ Proper train/validation/test splits
4. ✅ Conservative metrics on held-out test set
"""

from risk_model import FireRiskModel
import pandas as pd
import numpy as np
import os


def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def print_sample_predictions(model):
    """Test the model with sample scenarios."""
    print_header("🔮 SAMPLE PREDICTIONS")
    
    scenarios = [
        {
            'name': 'Low Risk - Short vegetation, good clearance',
            'data': {
                'veg_height_m': 0.8,
                'clearance_m': 7.2,
                'temperature_c': 20,
                'humidity_pct': 60,
                'wind_speed_ms': 4,
                'days_since_rain': 3,
                'latitude': 37.5,
                'longitude': -121.8
            }
        },
        {
            'name': 'Moderate Risk - Medium vegetation, adequate clearance',
            'data': {
                'veg_height_m': 2.0,
                'clearance_m': 4.5,
                'temperature_c': 28,
                'humidity_pct': 45,
                'wind_speed_ms': 8,
                'days_since_rain': 10,
                'latitude': 36.7,
                'longitude': -119.8
            }
        },
        {
            'name': 'High Risk - Tall vegetation, low clearance',
            'data': {
                'veg_height_m': 4.5,
                'clearance_m': 2.8,
                'temperature_c': 33,
                'humidity_pct': 30,
                'wind_speed_ms': 14,
                'days_since_rain': 18,
                'latitude': 38.5,
                'longitude': -120.5
            }
        },
        {
            'name': 'Critical Risk - Very tall vegetation, minimal clearance',
            'data': {
                'veg_height_m': 7.0,
                'clearance_m': 1.2,
                'temperature_c': 38,
                'humidity_pct': 15,
                'wind_speed_ms': 20,
                'days_since_rain': 30,
                'latitude': 37.0,
                'longitude': -121.5
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"{scenario['name']}")
        print('='*60)
        
        prediction = model.predict(scenario['data'])
        
        print(f"\n🎯 Prediction Results:")
        print(f"   Risk Level:    {prediction['risk_level']}")
        print(f"   Risk Score:    {prediction['risk_score']:.3f} ({prediction['risk_percentage']:.1f}%)")
        print(f"   Confidence:    {prediction['confidence']:.2%}")
        
        print(f"\n📊 Risk Probabilities:")
        for level, prob in prediction['probabilities'].items():
            bar_length = int(prob * 30)
            bar = '█' * bar_length + '░' * (30 - bar_length)
            print(f"   {level:10s} {bar} {prob:.2%}")
        
        # Risk meter
        risk_score = prediction['risk_score']
        meter_pos = int(risk_score * 40)
        meter = '█' * meter_pos + '░' * (40 - meter_pos)
        
        # Emoji indicator
        if risk_score < 0.25:
            emoji = '🟢'
        elif risk_score < 0.55:
            emoji = '🟡'
        elif risk_score < 0.80:
            emoji = '🟠'
        else:
            emoji = '🔴'
        
        print(f"\n   Risk Meter: {emoji}")
        print(f"   [0%] {meter} [100%]")


if __name__ == "__main__":
    print_header("🔥 FIRE APP - FIXED ML MODEL TRAINING")
    
    print("\n✅ IMPROVEMENTS IN THIS VERSION:")
    print("   1. Temporal splitting - no future→past data leakage")
    print("   2. Unified regressor - no classifier/regressor disagreement")
    print("   3. Proper train/val/test - realistic performance metrics")
    print("   4. Conservative evaluation - held-out test set")
    
    # Initialize model
    print_header("📌 Step 1: Initializing Model")
    model = FireRiskModel()
    print("✅ Model initialized with unified regressor approach")
    
    # Generate training data with temporal ordering
    print_header("📌 Step 2: Generating Training Data")
    features_df, risk_scores, timestamps = model.generate_training_data(n_samples=2000)
    print("✅ Training data generated with temporal ordering")
    
    print(f"\n📋 Sample features (first 5 rows):")
    print(features_df.head())
    
    # Train models
    print_header("📌 Step 3: Training Model")
    metrics = model.train(features_df, risk_scores, timestamps)
    print("✅ Model training complete")
    
    # Save models
    print_header("📌 Step 4: Saving Model")
    timestamp = model.save_models()
    
    # Test predictions
    print_header("📌 Step 5: Testing Predictions")
    print_sample_predictions(model)
    
    # Final summary
    print_header("✅ TRAINING COMPLETE")
    
    print(f"\n📦 Summary:")
    print(f"   • Trained on 2000 samples with temporal ordering")
    print(f"   • Features: {len(model.feature_names)}")
    print(f"   • Model: {metrics['model']}")
    print(f"   • Validation R²: {metrics['validation']['r2']:.4f}")
    print(f"   • Validation MAE: {metrics['validation']['mae']:.4f}")
    print(f"   • Test R² (held-out): {metrics['test']['r2']:.4f}")
    print(f"   • Test MAE (held-out): {metrics['test']['mae']:.4f}")
    print(f"   • Risk Level Accuracy: {metrics['test']['risk_accuracy']:.2%}")
    print(f"   • Model saved with timestamp: {timestamp}")
    
    print(f"\n💡 Key Improvements:")
    print(f"   ✅ NO data leakage (temporal splitting)")
    print(f"   ✅ NO model disagreement (unified regressor)")
    print(f"   ✅ REALISTIC metrics (proper validation)")
    print(f"   ✅ CONSERVATIVE evaluation (held-out test set)")
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Models are saved in the 'models/' directory")
    print(f"   2. Run 'python3 app.py' to start the Flask server")
    print(f"   3. The ML model will be automatically loaded")
    print(f"   4. Access /api/ml_predict endpoint for predictions")
    
    print(f"\n🎉 Your Fire App now has a PROPERLY VALIDATED ML model!")
    print("="*60)
