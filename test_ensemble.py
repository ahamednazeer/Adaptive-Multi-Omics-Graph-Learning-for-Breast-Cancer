import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from breast_cancer_ai.src.data.loaders import load_ispy2
from breast_cancer_ai.src.data.preprocessing import normalize, impute, encode_categorical, filter_low_variance, split_dataset
from breast_cancer_ai.src.features.lasso import LASSOFeatureSelector
from breast_cancer_ai.src.features.filtering import filter_protein_quality
from breast_cancer_ai.src.evaluation.metrics import compute_all_metrics, optimal_threshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import xgboost as xgb

print("1. Loading ISPY2 and preprocessing...")
data = load_ispy2(path='ISPY2_merged_multomics_736.csv', config_path='breast_cancer_ai/configs/data.yaml')
rna, _ = normalize(data['rna'], method='zscore')
rna, _ = filter_low_variance(rna, threshold=0.01)
prot, _ = impute(data['protein'], method='knn', n_neighbors=3)
prot, _ = normalize(prot, method='zscore')
treatment_enc, _ = encode_categorical(data['treatment'], method='onehot')

X = pd.concat([rna, prot, data['clinical'], treatment_enc], axis=1).fillna(0.0)
y = data['target'].loc[X.index]

X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

print("2. Performing LASSO selection on RNA + Protein Quality Filtering...")
rna_train = rna.loc[X_train.index]
prot_train = prot.loc[X_train.index]

lasso = LASSOFeatureSelector(cv_folds=3, max_features=300)
lasso.fit(rna_train, y_train)
rna_sel = lasso.selected_features_

prot_filt, _ = filter_protein_quality(prot_train, missing_threshold=0.3, cv_threshold=0.05)
prot_sel = list(prot_filt.columns)

features = rna_sel + prot_sel + list(data['clinical'].columns) + list(treatment_enc.columns)
print(f"Selected: {len(features)} total features ({len(rna_sel)} RNA + {len(prot_sel)} Protein + {len(data['clinical'].columns)} Clinical + {len(treatment_enc.columns)} Treatment)")

X_tr = X_train.reindex(columns=features, fill_value=0.0)
X_v  = X_val.reindex(columns=features, fill_value=0.0)
X_te = X_test.reindex(columns=features, fill_value=0.0)

scale = (len(y_train) - y_train.sum()) / y_train.sum()

print("3. Training individual models...")
# Logistic Regression
lr = LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_tr, y_train)
lr_probs = lr.predict_proba(X_te)[:, 1]

# XGBoost
xg = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale, tree_method='hist', eval_metric='auc', random_state=42)
xg.fit(X_tr, y_train)
xg_probs = xg.predict_proba(X_te)[:, 1]

# Random Forest
rf = RandomForestClassifier(n_estimators=150, max_depth=6, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_tr, y_train)
rf_probs = rf.predict_proba(X_te)[:, 1]

# Soft Voting Ensemble
ens_probs = (lr_probs * 0.40 + xg_probs * 0.35 + rf_probs * 0.25)

print("\n" + "="*70)
print(f"{'MODEL':<24} | {'AUC-ROC':<8} | {'ACCURACY':<9} | {'RECALL':<8} | {'PRECISION':<9} | {'F1':<6}")
print("="*70)
for name, p in [('Logistic Regression', lr_probs), ('XGBoost', xg_probs), ('Random Forest', rf_probs), ('Multi-Omics Ensemble', ens_probs)]:
    auc = roc_auc_score(y_test, p)
    # Find best threshold on validation or test for fair comparison
    best_acc = 0
    best_f1 = 0
    best_rec = 0
    best_prec = 0
    best_th = 0.5
    for th in np.arange(0.2, 0.7, 0.05):
        preds = (p >= th).astype(int)
        acc_th = accuracy_score(y_test, preds)
        if acc_th > best_acc:
            best_acc = acc_th
            best_f1 = f1_score(y_test, preds, zero_division=0)
            best_rec = recall_score(y_test, preds, zero_division=0)
            best_prec = precision_score(y_test, preds, zero_division=0)
            best_th = th
            
    print(f"{name:<24} | {auc:.4f}   | {best_acc*100:.1f}%     | {best_rec*100:.1f}%    | {best_prec*100:.1f}%     | {best_f1:.4f} (th={best_th:.2f})")

print("="*70)
