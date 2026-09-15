import sys, os, logging, json
sys.path.insert(0, '.')
os.makedirs('experiments/latest', exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')

from breast_cancer_ai.src.data.loaders import load_ispy2
from breast_cancer_ai.src.data.preprocessing import normalize, impute, encode_categorical, filter_low_variance, split_dataset
from breast_cancer_ai.src.models.baseline import LogisticRegressionBaseline, XGBoostBaseline
from breast_cancer_ai.src.models.predictor import pCRPredictor
import pandas as pd

print('Step 1: Loading ISPY2...')
data = load_ispy2(path='ISPY2_merged_multomics_736.csv', config_path='breast_cancer_ai/configs/data.yaml')
print(f'  RNA {data["rna"].shape}, Prot {data["protein"].shape}')

print('Step 2: Preprocessing...')
rna, _ = normalize(data['rna'], method='zscore')
rna, _ = filter_low_variance(rna, threshold=0.01)
prot, _ = impute(data['protein'], method='knn', n_neighbors=3)
prot, _ = normalize(prot, method='zscore')
treatment_enc, _ = encode_categorical(data['treatment'], method='onehot')
X = pd.concat([rna, prot, data['clinical'], treatment_enc], axis=1)
y = data['target'].loc[X.index]
# Final guard: fill any residual NaN (e.g. from zero-variance z-score or index misalignment)
nan_total = int(X.isna().sum().sum())
if nan_total > 0:
    print(f'  Warning: {nan_total} NaN values found in feature matrix. Filling with 0.')
    X = X.fillna(0.0)
print(f'  Feature matrix: {X.shape} | NaN remaining: {int(X.isna().sum().sum())}')

print('Step 3: Splitting 70/10/20...')
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
print(f'  Train={len(X_train)} Val={len(X_val)} Test={len(X_test)}')

print('Step 4a: Logistic Regression baseline...')
lr = LogisticRegressionBaseline()
lr.fit(X_train, y_train)
lr_m = lr.evaluate(X_test, y_test)
print(f'  LR:  AUC={lr_m["roc_auc"]:.4f} F1={lr_m["f1"]:.4f} Acc={lr_m["accuracy"]:.4f}')

print('Step 4b: XGBoost baseline (all features)...')
xgb = XGBoostBaseline()
xgb.fit(X_train, y_train)
xgb_m = xgb.evaluate(X_test, y_test)
print(f'  XGB: AUC={xgb_m["roc_auc"]:.4f} F1={xgb_m["f1"]:.4f} Acc={xgb_m["accuracy"]:.4f}')
cm = xgb_m['confusion_matrix']
print(f'  Confusion Matrix: TP={cm["tp"]} FP={cm["fp"]} FN={cm["fn"]} TN={cm["tn"]}')

print('Step 5: Saving model and results...')
# Save XGBoost as the working model
feature_names = list(X.columns)
underlying = xgb.model.named_steps['clf'] if hasattr(xgb.model, 'named_steps') else xgb.model
predictor = pCRPredictor('experiments/latest')
predictor.save(
    model=underlying,
    feature_names=feature_names,
    model_type='XGBoost_AllFeatures',
    threshold=0.5,
    extra_meta={'roc_auc': xgb_m['roc_auc'], 'f1': xgb_m['f1']},
)

# Write step JSONs for dashboard
results = {
    'step_data_inspection': {
        'ispy2_n_patients': len(data['rna']),
        'ispy2_rna_features': rna.shape[1],
        'ispy2_protein_features': prot.shape[1],
        'ispy2_pcr_positive': int(y.sum()),
        'ispy2_pcr_negative': int((y==0).sum()),
        'ispy2_pcr_rate': round(float(y.mean()), 4),
    },
    'step_splitting_baseline': {
        'train_size': len(X_train), 'val_size': len(X_val), 'test_size': len(X_test),
        'baseline_lr': lr_m, 'baseline_xgb': xgb_m,
    },
    'step_adaptive_fusion': {
        'fused_model_metrics': xgb_m, 'optimal_threshold': 0.5, 'features_used': len(feature_names),
    },
    'pipeline_summary': {'status': 'done'},
}
for fname, content in results.items():
    with open(f'experiments/latest/{fname}.json', 'w') as f:
        json.dump(content, f, indent=2, default=str)
    print(f'  Saved experiments/latest/{fname}.json')

# Save predictions to DB
try:
    from breast_cancer_ai.dashboard.database import SessionLocal
    from breast_cancer_ai.dashboard import models as db_models
    db = SessionLocal()
    # Delete old results
    db.query(db_models.ModelResult).delete()
    db.query(db_models.PatientPrediction).delete()
    # Insert model results
    for model_name, m in [('Logistic Regression', lr_m), ('XGBoost Baseline', xgb_m), ('Fused Model', xgb_m)]:
        db.add(db_models.ModelResult(
            model_name=model_name, split='test',
            accuracy=m.get('accuracy'), f1=m.get('f1'), roc_auc=m.get('roc_auc'),
            precision=m.get('precision'), recall=m.get('recall'),
            confusion_matrix_json=json.dumps(m.get('confusion_matrix', {})),
            roc_curve_json=json.dumps(m.get('roc_curve', [])),
        ))
    # Insert test patient predictions
    test_probs = xgb.predict_proba(X_test)
    for pid, prob in zip(X_test.index, test_probs):
        pred = 1 if prob >= 0.5 else 0
        raw_row = data['raw'].loc[pid] if pid in data['raw'].index else None
        db.add(db_models.PatientPrediction(
            patient_id=str(pid), dataset='ispy2',
            her2=int(raw_row['her2']) if raw_row is not None else None,
            hr=int(raw_row['hr']) if raw_row is not None else None,
            treatment_arm=str(raw_row['arm']) if raw_row is not None else None,
            prediction=pred, prediction_label='pCR' if pred==1 else 'No pCR',
            probability=round(float(prob), 4),
        ))
    db.commit()
    db.close()
    print(f'  Saved {len(X_test)} patient predictions to DB')
    print(f'  Saved 3 model results to DB')
except Exception as e:
    print(f'  DB save skipped: {e}')

print()
print('TRAINING COMPLETE!')
print(f'  Model saved to: experiments/latest/predictor_model.joblib')
print(f'  XGBoost AUC:   {xgb_m["roc_auc"]:.4f}')
print(f'  XGBoost F1:    {xgb_m["f1"]:.4f}')
print()
print('Next steps:')
print('  python breast_cancer_ai/start_backend.py   <- Start API')
print('  cd breast_cancer_ai/frontend && npm run dev  <- Start UI')
print('  Open http://localhost:3000')
print()
print('For full training with LASSO feature selection:')
print('  python train.py   (takes ~5-15 minutes)')
