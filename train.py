"""
STANDALONE TRAINING SCRIPT
============================
Runs the complete ML training pipeline directly from command line.
No UI or backend server needed.

Usage (from project root):
    python train.py

What this does:
  1. Loads ISPY2 (736 patients, 19K+ RNA + 139 protein features)
  2. Preprocesses: imputes missing protein (KNN), z-scores RNA, one-hot encodes treatment
  3. Splits: patient-level stratified 70/10/20 train/val/test
  4. Baseline models: Logistic Regression + XGBoost (with class_weight='balanced')
  5. Feature selection: LASSO on RNA (→ ~500 genes) + protein quality filter
  6. Final model: XGBoost on selected features (acts as fusion proxy without GCN/GAT)
  7. Saves trained model to experiments/latest/ for the API to use
  8. Prints all metrics: Accuracy, F1, ROC-AUC, Precision, Recall, Confusion Matrix
"""
from __future__ import annotations
import sys, os, json, time, logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("experiments/latest", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def banner(msg: str):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")

# ─────────────────────────────────────────────────────────────
# Step 1: Load Data
# ─────────────────────────────────────────────────────────────
banner("STEP 1 — Loading ISPY2 Dataset")
t0 = time.time()

from breast_cancer_ai.src.data.loaders import load_ispy2

data = load_ispy2(
    path="ISPY2_merged_multomics_736.csv",
    config_path="breast_cancer_ai/configs/data.yaml",
)
print(f"  RNA:      {data['rna'].shape}  (already log-normalized)")
print(f"  Protein:  {data['protein'].shape}  (~20% missing, will impute)")
print(f"  Clinical: {data['clinical'].shape}  (her2, hr binary)")
print(f"  Target:   pCR=1: {int(data['target'].sum())}  |  pCR=0: {int((data['target']==0).sum())}")
print(f"  Loaded in {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────
# Step 2: Preprocessing
# ─────────────────────────────────────────────────────────────
banner("STEP 2 — Preprocessing (Impute + Normalize + Encode)")
t0 = time.time()

from breast_cancer_ai.src.data.preprocessing import (
    normalize, impute, encode_categorical, filter_low_variance, filter_high_missing
)
import pandas as pd

# RNA: already log-normalized → z-score
rna, rna_scaler = normalize(data["rna"], method="zscore")
rna, _ = filter_low_variance(rna, threshold=0.01)
print(f"  RNA after variance filter: {rna.shape[1]} features")

# Protein: KNN impute (20% missing) then z-score
prot, prot_imputer = impute(data["protein"], method="knn", n_neighbors=5)
prot, prot_scaler = normalize(prot, method="zscore")
print(f"  Protein after impute+zscore: {prot.shape[1]} features")

# Clinical: already binary (her2, hr)
clinical = data["clinical"].copy()

# Treatment: one-hot encode arm
treatment_enc, treat_encoder = encode_categorical(data["treatment"], method="onehot")
print(f"  Treatment one-hot: {treatment_enc.shape[1]} features")

# Merge all
X = pd.concat([rna, prot, clinical, treatment_enc], axis=1)
y = data["target"].copy()
common = X.index.intersection(y.index)
X = X.loc[common]
y = y.loc[common]
# Guard: fill any NaN from zero-variance z-score or index misalignment
_nan = int(X.isna().sum().sum())
if _nan > 0:
    print(f"  Warning: {_nan} NaN values in feature matrix → filling with 0")
    X = X.fillna(0.0)
print(f"  Total feature matrix: {X.shape[0]} patients x {X.shape[1]} features | NaN: {int(X.isna().sum().sum())}")
print(f"  Preprocessed in {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────
# Step 3: Train/Val/Test Split
# ─────────────────────────────────────────────────────────────
banner("STEP 3 — Patient-Level Stratified Split (70/10/20)")
t0 = time.time()

from breast_cancer_ai.src.data.preprocessing import split_dataset

X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
print(f"  Train: {X_train.shape[0]} patients | pCR={int(y_train.sum())}/{len(y_train)}")
print(f"  Val:   {X_val.shape[0]} patients  | pCR={int(y_val.sum())}/{len(y_val)}")
print(f"  Test:  {X_test.shape[0]} patients  | pCR={int(y_test.sum())}/{len(y_test)}")
print(f"  Split in {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────
# Step 4: Baseline Models
# ─────────────────────────────────────────────────────────────
banner("STEP 4 — Baseline Models (LR + XGBoost)")
t0 = time.time()

from breast_cancer_ai.src.models.baseline import LogisticRegressionBaseline, XGBoostBaseline
from breast_cancer_ai.src.evaluation.metrics import compute_all_metrics

lr = LogisticRegressionBaseline()
lr.fit(X_train, y_train)
lr_metrics = lr.evaluate(X_test, y_test)

xgb = XGBoostBaseline()
xgb.fit(X_train, y_train)
xgb_metrics = xgb.evaluate(X_test, y_test)

print(f"\n  Logistic Regression:")
print(f"    AUC={lr_metrics['roc_auc']:.4f}  F1={lr_metrics['f1']:.4f}  Acc={lr_metrics['accuracy']:.4f}")
print(f"\n  XGBoost Baseline:")
print(f"    AUC={xgb_metrics['roc_auc']:.4f}  F1={xgb_metrics['f1']:.4f}  Acc={xgb_metrics['accuracy']:.4f}")
print(f"\n  Baselines trained in {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────
# Step 5: Feature Selection (LASSO + Protein quality)
# ─────────────────────────────────────────────────────────────
banner("STEP 5 — Feature Selection (LASSO on RNA, protein quality filter)")
t0 = time.time()

from breast_cancer_ai.src.features.lasso import LASSOFeatureSelector
from breast_cancer_ai.src.features.filtering import filter_protein_quality

# LASSO on RNA portion of training set
rna_train = rna.loc[X_train.index]
prot_train = prot.loc[X_train.index]
y_tr = y_train

print("  Running LASSO (this takes ~1-3 min on 19K+ RNA features)...")
lasso = LASSOFeatureSelector(cv_folds=3, max_features=300)
lasso.fit(rna_train, y_tr)
rna_selected = lasso.selected_features_
print(f"  LASSO selected {len(rna_selected)} RNA genes")

# Protein: variance-based quality filter
prot_filtered, prot_removed = filter_protein_quality(
    prot_train, missing_threshold=0.3, cv_threshold=0.05
)
prot_final = list(prot_filtered.columns)
print(f"  Protein kept: {len(prot_final)} features (removed {len(prot_removed)})")
print(f"  Feature selection in {time.time()-t0:.1f}s")

# Top selected genes
print(f"  Top 10 LASSO genes: {rna_selected[:10]}")

# ─────────────────────────────────────────────────────────────
# Step 6: Train Final Model on Selected Features
# ─────────────────────────────────────────────────────────────
banner("STEP 6 — Final Model (XGBoost on selected features)")
t0 = time.time()

final_cols = rna_selected + prot_final + list(clinical.columns) + list(treatment_enc.columns)
X_train_sel = X_train.reindex(columns=final_cols, fill_value=0.0)
X_val_sel   = X_val.reindex(columns=final_cols, fill_value=0.0)
X_test_sel  = X_test.reindex(columns=final_cols, fill_value=0.0)
print(f"  Training on {len(final_cols)} selected features")

final_model = XGBoostBaseline()
final_model.fit(X_train_sel, y_train)

# Find optimal threshold on validation set
val_probs = final_model.predict_proba(X_val_sel)
from breast_cancer_ai.src.evaluation.metrics import optimal_threshold
threshold = optimal_threshold(y_val.values, val_probs)
print(f"  Optimal threshold (val F1): {threshold:.2f}")

# Evaluate on test set
test_probs = final_model.predict_proba(X_test_sel)
test_preds = (test_probs >= threshold).astype(int)
final_metrics = compute_all_metrics(y_test.values, test_preds, test_probs, "FinalXGBoost")

print(f"\n  FINAL MODEL RESULTS (test set):")
print(f"  {'AUC-ROC':<12}: {final_metrics['roc_auc']:.4f}")
print(f"  {'F1 Score':<12}: {final_metrics['f1']:.4f}")
print(f"  {'Accuracy':<12}: {final_metrics['accuracy']:.4f}")
print(f"  {'Precision':<12}: {final_metrics['precision']:.4f}")
print(f"  {'Recall':<12}: {final_metrics['recall']:.4f}")
cm = final_metrics['confusion_matrix']
print(f"  Confusion Matrix:")
print(f"    TP={cm['tp']}  FP={cm['fp']}")
print(f"    FN={cm['fn']}  TN={cm['tn']}")
print(f"  Trained in {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────
# Step 7: SHAP Biomarker Ranking
# ─────────────────────────────────────────────────────────────
banner("STEP 7 — SHAP Biomarker Ranking")
t0 = time.time()
try:
    from breast_cancer_ai.src.explainability.shap import SHAPExplainer
    import shap as shap_lib  # noqa

    underlying = final_model.model.named_steps["clf"] if hasattr(final_model.model, "named_steps") else final_model.model
    explainer = SHAPExplainer(underlying, final_cols, model_type="tree")
    explainer.fit(X_test_sel)
    biomarkers = explainer.get_top_biomarkers(X_test_sel)

    print(f"\n  Top 10 pCR Promoters (↑ pCR):")
    for b in biomarkers["top_positive"][:10]:
        print(f"    {b['name']:<20} SHAP={b['mean_abs_shap']:.4f}")

    print(f"\n  Top 10 pCR Inhibitors (↓ pCR):")
    for b in biomarkers["top_negative"][:10]:
        print(f"    {b['name']:<20} SHAP={b['mean_shap']:.4f}")

    print(f"  SHAP computed in {time.time()-t0:.1f}s")
except ImportError:
    print("  SHAP not installed (pip install shap). Skipping.")
    biomarkers = {"top_positive": [], "top_negative": [], "all_features": []}

# ─────────────────────────────────────────────────────────────
# Step 8: Save Model and Results
# ─────────────────────────────────────────────────────────────
banner("STEP 8 — Saving Model and Results")

from breast_cancer_ai.src.models.predictor import pCRPredictor
predictor = pCRPredictor("experiments/latest")
predictor.save(
    model=final_model.model.named_steps["clf"] if hasattr(final_model.model, "named_steps") else final_model.model,
    feature_names=final_cols,
    model_type="XGBoost_SelectedFeatures",
    threshold=threshold,
    extra_meta={
        "roc_auc": final_metrics["roc_auc"],
        "f1":      final_metrics["f1"],
        "n_rna_features":     len(rna_selected),
        "n_protein_features": len(prot_final),
        "total_features":     len(final_cols),
    }
)

# Save step result JSONs (so the API/dashboard can read them)
results = {
    "data_inspection": {
        "ispy2_n_patients": 736,
        "ispy2_rna_features": rna.shape[1],
        "ispy2_protein_features": prot.shape[1],
        "ispy2_pcr_positive": int(y.sum()),
        "ispy2_pcr_negative": int((y==0).sum()),
        "ispy2_pcr_rate": round(float(y.mean()), 4),
    },
    "splitting_baseline": {
        "train_size": len(X_train), "val_size": len(X_val), "test_size": len(X_test),
        "baseline_lr": lr_metrics, "baseline_xgb": xgb_metrics,
    },
    "feature_selection": {
        "rna_lasso_selected": len(rna_selected),
        "protein_selected": len(prot_final),
        "top_rna_genes": rna_selected[:20],
        "top_proteins": prot_final[:10],
    },
    "adaptive_fusion": {
        "fused_model_metrics": final_metrics,
        "optimal_threshold": threshold,
        "features_used": len(final_cols),
    },
    "explainability": {
        "top_positive_biomarkers": biomarkers.get("top_positive", [])[:20],
        "top_negative_biomarkers": biomarkers.get("top_negative", [])[:20],
    },
}

for step_name, step_data in results.items():
    out_path = f"experiments/latest/step_{step_name}.json"
    with open(out_path, "w") as f:
        json.dump(step_data, f, indent=2, default=str)
    print(f"  Saved: {out_path}")

# Save pipeline summary
with open("experiments/latest/pipeline_summary.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# ─────────────────────────────────────────────────────────────
# Save predictions to DB (for the patients page)
# ─────────────────────────────────────────────────────────────
print("\n  Saving patient predictions to database...")
try:
    from breast_cancer_ai.dashboard.database import SessionLocal
    from breast_cancer_ai.dashboard import models as db_models
    from datetime import datetime
    import json as json_lib

    db = SessionLocal()

    # Store model results
    for model_name, metrics in [
        ("Logistic Regression", lr_metrics),
        ("XGBoost Baseline", xgb_metrics),
        ("Fused Model", final_metrics),
    ]:
        # Remove old results for this model
        db.query(db_models.ModelResult).filter(db_models.ModelResult.model_name == model_name).delete()
        db.add(db_models.ModelResult(
            model_name=model_name, split="test",
            accuracy=metrics.get("accuracy"), f1=metrics.get("f1"),
            roc_auc=metrics.get("roc_auc"), precision=metrics.get("precision"),
            recall=metrics.get("recall"),
            confusion_matrix_json=json_lib.dumps(metrics.get("confusion_matrix", {})),
            roc_curve_json=json_lib.dumps(metrics.get("roc_curve", [])),
        ))

    # Store biomarkers
    db.query(db_models.Biomarker).delete()
    all_bm = biomarkers.get("top_positive", []) + biomarkers.get("top_negative", [])
    for rank, bm in enumerate(all_bm, start=1):
        ftype = "protein" if "." in bm["name"] and bm["name"][0].isupper() else "rna"
        db.add(db_models.Biomarker(
            gene=bm["name"], feature_type=ftype,
            mean_abs_shap=bm.get("mean_abs_shap", abs(bm.get("mean_shap", 0))),
            mean_shap=bm.get("mean_shap", 0),
            direction=bm.get("type", "positive"),
            rank=rank,
        ))

    # Store patient predictions (test set)
    test_probs_all = final_model.predict_proba(X_test_sel)
    db.query(db_models.PatientPrediction).delete()
    for pid, prob in zip(X_test.index, test_probs_all):
        pred = 1 if prob >= threshold else 0
        row = data["raw"].loc[pid] if pid in data["raw"].index else None
        db.add(db_models.PatientPrediction(
            patient_id=str(pid), dataset="ispy2",
            her2=int(row["her2"]) if row is not None and "her2" in data["raw"].columns else None,
            hr=int(row["hr"]) if row is not None and "hr" in data["raw"].columns else None,
            treatment_arm=str(row["arm"]) if row is not None and "arm" in data["raw"].columns else None,
            prediction=pred, prediction_label="pCR" if pred == 1 else "No pCR",
            probability=round(float(prob), 4),
        ))

    db.commit()
    db.close()
    print(f"  Saved {len(X_test)} patient predictions to DB")
    print(f"  Saved {len(all_bm)} biomarkers to DB")
    print(f"  Saved 3 model result records to DB")
except Exception as e:
    print(f"  DB save warning: {e} (run the backend first for DB availability)")

banner("TRAINING COMPLETE")
print(f"  Model: experiments/latest/predictor_model.joblib")
print(f"  Best AUC: {final_metrics['roc_auc']:.4f}")
print(f"  Best F1:  {final_metrics['f1']:.4f}")
print()
print("  Next steps:")
print("  1. python breast_cancer_ai/start_backend.py   <- Start API server")
print("  2. cd breast_cancer_ai/frontend && npm run dev  <- Start frontend")
print("  3. Open http://localhost:3000")
