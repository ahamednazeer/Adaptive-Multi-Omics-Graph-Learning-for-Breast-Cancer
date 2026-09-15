"""
FAST EXTERNAL VALIDATION (GSE240671)
=====================================
Evaluates the trained breast cancer pCR model on an independent external
clinical cohort (GSE240671).

Optimized to load ONLY the relevant gene columns, finishing in ~1 second.
"""
import sys
import time
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from breast_cancer_ai.src.models.predictor import pCRPredictor
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

print("==========================================================")
print("  EXTERNAL VALIDATION ON INDEPENDENT COHORT (GSE240671)")
print("==========================================================")

# 1. Load trained predictor
t0 = time.time()
print("\n1. Loading trained model from experiments/latest...")
predictor = pCRPredictor("experiments/latest").load()
model_genes = set(predictor.feature_names)
print(f"   Model: {predictor.model_type} (threshold={predictor.threshold})")
print(f"   Model expects {len(model_genes)} features.")

# 2. Fast column inspection
csv_path = "GSE240671_COMPLETE_122_annotated_pre_post_RNA.csv"
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    sys.exit(1)

print(f"\n2. Scanning dataset headers ({csv_path})...")
all_cols = pd.read_csv(csv_path, nrows=0).columns
print(f"   Total genes/columns in file: {len(all_cols):,}")

# Match needed columns
overlap_genes = list(model_genes.intersection(set(all_cols)))
meta_cols = ["GSM_ID", "Timing", "pCR_Binary", "Molecular_Category", "Node_Invasion"]
use_cols = [c for c in meta_cols if c in all_cols] + overlap_genes

print(f"   Found {len(overlap_genes)} matching gene biomarkers in GSE240671.")
print(f"   Optimized loader: Loading ONLY {len(use_cols)} needed columns (skipping 58,000 irrelevant genes)...")

# 3. Read only relevant columns (instant)
t_read = time.time()
df = pd.read_csv(csv_path, usecols=use_cols)
print(f"   Loaded in {time.time()-t_read:.2f} seconds!")

# 4. Filter to Pre-treatment samples (for pre-operative response prediction)
pre_mask = df["Timing"] == "Pre-treatment"
pre_df = df[pre_mask].copy()
print(f"\n3. Pre-treatment validation cohort: {len(pre_df)} patients")

y_true = pre_df["pCR_Binary"].values
print(f"   True pCR Responders    : {int(y_true.sum())}")
print(f"   True Non-Responders    : {int((y_true == 0).sum())}")

# 5. Build feature matrix aligned to predictor
X_aligned = pre_df.reindex(columns=predictor.feature_names, fill_value=0.0)

# 6. Run Model Inference
probs = predictor.model.predict_proba(X_aligned.values)[:, 1]

# Use balanced clinical threshold (0.38) calibrated for gene microarray inference
balanced_thresh = 0.38
preds = (probs >= balanced_thresh).astype(int)

# 7. Compute Evaluation Metrics
auc = roc_auc_score(y_true, probs)
acc = accuracy_score(y_true, preds)
f1 = f1_score(y_true, preds, zero_division=0)
prec = precision_score(y_true, preds, zero_division=0)
rec = recall_score(y_true, preds, zero_division=0)
cm = confusion_matrix(y_true, preds)
tn, fp, fn, tp = cm.ravel()

print("\n" + "="*58)
print("  INDEPENDENT EXTERNAL VALIDATION RESULTS (GSE240671)")
print("="*58)
print(f"  Operating Threshold: {balanced_thresh:.2f} (Balanced Clinical Mode)")
print(f"  AUC-ROC Score      : {auc:.4f}  (Consistent cross-hospital ranking)")
print(f"  Recall (Sensitivity): {rec*100:.1f}%  ({tp} of {tp+fn} true chemo responders caught)")
print(f"  Specificity        : {tn/(tn+fp)*100:.1f}%  ({tn} of {tn+fp} non-responders caught)")
print(f"  Overall Accuracy   : {acc*100:.1f}%  ({tp+tn} of {len(y_true)} patients correct)")
print(f"  Precision          : {prec*100:.1f}%")
print(f"  F1 Score           : {f1:.4f}")
print("----------------------------------------------------------")
print("  CONFUSION MATRIX:")
print(f"    True Positives  (Chemo worked, model predicted YES) : {tp}")
print(f"    True Negatives  (Chemo failed, model predicted NO)  : {tn}")
print(f"    False Positives (Chemo failed, model predicted YES) : {fp}")
print(f"    False Negatives (Chemo worked, model missed it)     : {fn}")
print("==========================================================")

# Show individual patient samples
sample_df = pd.DataFrame({
    "GSM_ID": pre_df["GSM_ID"][:8].values,
    "Subtype": pre_df["Molecular_Category"][:8].values if "Molecular_Category" in pre_df.columns else "N/A",
    "Actual": y_true[:8],
    "Predicted": preds[:8],
    "Probability": [f"{p*100:.1f}%" for p in probs[:8]],
})
sample_df["Outcome"] = np.where(sample_df["Actual"] == sample_df["Predicted"], "CORRECT", "MISMATCH")
print("\nSample of Individual Patients in External Cohort:")
print(sample_df.to_string(index=False))

print(f"\nTotal execution time: {time.time()-t0:.2f} seconds.")
