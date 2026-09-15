"""
CUSTOM DATA TESTING UTILITY
============================
Test the trained breast cancer pCR prediction model on custom data.

Usage Examples:

1. Test with built-in example profiles:
   python predict_custom.py --example

2. Test a single patient via command-line arguments:
   python predict_custom.py --her2 1 --hr 0 --treatment "Paclitaxel + MK-2206"
   python predict_custom.py --her2 0 --hr 1 --treatment "Paclitaxel (Control)"

3. Test a batch CSV file of custom patients:
   python predict_custom.py --csv path/to/my_patients.csv --output predictions.csv

4. Interactive Mode (prompts you for values):
   python predict_custom.py --interactive
"""
import sys
import os
import argparse
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from breast_cancer_ai.src.models.predictor import pCRPredictor

MODEL_DIR = "experiments/latest"

def get_loaded_predictor():
    if not os.path.exists(f"{MODEL_DIR}/predictor_model.joblib"):
        print("Error: No trained model found in experiments/latest.")
        print("Please run 'python train.py' or 'python quick_train.py' first.")
        sys.exit(1)
    
    predictor = pCRPredictor(MODEL_DIR)
    predictor.load()
    return predictor

def test_single_patient(predictor, her2=1, hr=0, treatment="Paclitaxel + MK-2206", gene_expressions=None, patient_id="CUSTOM_PATIENT_01"):
    features = {}
    
    # Gene / Protein overrides if provided
    if gene_expressions:
        features.update(gene_expressions)
    
    # Clinical markers
    features["her2"] = float(her2)
    features["hr"] = float(hr)
    
    # Treatment one-hot encoding matching trained model feature names
    treatment_arm_clean = treatment.replace(" ", "_").replace("+", "_").replace("-", "_").replace("/", "_")
    for feat in predictor.feature_names:
        if feat.startswith("arm_") or feat.startswith("treatment_"):
            if treatment.lower() in feat.lower() or any(part.lower() in feat.lower() for part in treatment.split() if len(part) > 2):
                features[feat] = 1.0
            else:
                features[feat] = 0.0

    result = predictor.predict_patient(features)
    
    print("\n" + "="*50)
    print(f"  PATIENT PREDICTION: {patient_id}")
    print("="*50)
    print(f"  Clinical Profile:")
    print(f"    HER2 Status    : {'Positive (+)' if her2 else 'Negative (-)'}")
    print(f"    HR Status      : {'Positive (+)' if hr else 'Negative (-)'}")
    print(f"    Treatment Arm  : {treatment}")
    if gene_expressions:
        print(f"    Custom Genes   : {', '.join(f'{k}={v}' for k, v in list(gene_expressions.items())[:5])}")
    print("\n  Prediction Result:")
    print(f"    pCR Probability: {result['probability']*100:.1f}%")
    print(f"    Decision       : {result['prediction_label'].upper()}")
    print(f"    Model Used     : {result['model_type']} (threshold={result['threshold']})")
    print("="*50)
    return result

def run_examples(predictor):
    print("\nRunning test on 3 distinct clinical profiles:")
    
    # Profile 1: HER2+ / HR- (historically high pCR with targeted therapy)
    test_single_patient(
        predictor,
        her2=1, hr=0,
        treatment="Paclitaxel + MK-2206",
        gene_expressions={"TCAP": 1.8, "MXD1": 1.2, "her2": 1.0, "MCM6": -0.5},
        patient_id="PATIENT_HER2_POS_HIGH_RESPONDER"
    )

    # Profile 2: Triple Negative (HER2- / HR- with Paclitaxel control)
    test_single_patient(
        predictor,
        her2=0, hr=0,
        treatment="Paclitaxel (Control)",
        gene_expressions={"TCAP": -0.2, "MCM6": 0.8, "KIF2A": 1.1},
        patient_id="PATIENT_TNBC_CHEMO_RESISTANT"
    )

    # Profile 3: HR+ / HER2- (Luminal, typically lower chemotherapy pCR response)
    test_single_patient(
        predictor,
        her2=0, hr=1,
        treatment="Paclitaxel + AMG-386",
        gene_expressions={"TCAP": -1.0, "MCM6": 1.5, "SERPINA3": 1.2},
        patient_id="PATIENT_HR_POS_LUMINAL"
    )

def test_batch_csv(predictor, csv_path, output_path="predictions.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found.")
        return
    
    print(f"Loading custom data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns.")
    
    results = predictor.predict_batch(df)
    results.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    print(results.head())

def interactive_mode(predictor):
    print("\n--- INTERACTIVE CUSTOM PATIENT PREDICTOR ---")
    pid = input("Enter Patient ID (default: TEST_PATIENT_01): ").strip() or "TEST_PATIENT_01"
    
    her2_in = input("HER2 Status (1 = Positive, 0 = Negative) [default: 1]: ").strip() or "1"
    her2 = int(her2_in)
    
    hr_in = input("HR Status (1 = Positive, 0 = Negative) [default: 0]: ").strip() or "0"
    hr = int(hr_in)
    
    print("\nAvailable Treatment Arms:")
    arms = [
        "Paclitaxel + MK-2206",
        "Paclitaxel + AMG-386",
        "Paclitaxel + ABT 888 + Carboplatin",
        "Paclitaxel + Neratinib",
        "Paclitaxel + Pembrolizumab",
        "Paclitaxel + Pertuzumab",
        "Paclitaxel (Control)",
        "Paclitaxel + T-DM1/Pertuzumab",
    ]
    for idx, arm in enumerate(arms, 1):
        print(f"  {idx}. {arm}")
    
    choice = input("Select arm number (1-8) [default: 1]: ").strip() or "1"
    try:
        treatment = arms[int(choice) - 1]
    except Exception:
        treatment = arms[0]
        
    test_single_patient(predictor, her2=her2, hr=hr, treatment=treatment, patient_id=pid)

def main():
    parser = argparse.ArgumentParser(description="Test Breast Cancer AI model on custom data.")
    parser.add_argument("--example", action="store_true", help="Run 3 example clinical patient profiles")
    parser.add_argument("--interactive", action="store_true", help="Prompt interactively for patient features")
    parser.add_argument("--her2", type=int, default=None, help="HER2 status (0 or 1)")
    parser.add_argument("--hr", type=int, default=None, help="HR status (0 or 1)")
    parser.add_argument("--treatment", type=str, default="Paclitaxel + MK-2206", help="Treatment arm string")
    parser.add_argument("--csv", type=str, default=None, help="Path to custom CSV file for batch inference")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output path for batch predictions CSV")
    
    args = parser.parse_args()
    predictor = get_loaded_predictor()
    
    if args.csv:
        test_batch_csv(predictor, args.csv, args.output)
    elif args.interactive:
        interactive_mode(predictor)
    elif args.her2 is not None and args.hr is not None:
        test_single_patient(predictor, her2=args.her2, hr=args.hr, treatment=args.treatment)
    else:
        run_examples(predictor)

if __name__ == "__main__":
    main()
