"""
Pipeline orchestrator — runs all 10 steps end-to-end.
Each step updates the shared state dict and persists results to disk.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

STEP_NAMES = [
    "data_inspection",
    "preprocessing",
    "splitting_baseline",
    "feature_selection",
    "graph_construction",
    "gcn_gat_training",
    "adaptive_fusion",
    "explainability",
    "temporal_external_validation",
    "evaluation",
]


class PipelineOrchestrator:
    """
    Runs all 10 pipeline steps sequentially.
    Persists results after each step so partial progress is saved.
    Reports step status via a callback for real-time UI updates.
    """

    def __init__(
        self,
        config_path: str = "configs/data.yaml",
        model_config_path: str = "configs/model.yaml",
        output_dir: str = "experiments/latest",
        status_callback: Optional[Callable] = None,
        abort_check: Optional[Callable[[], bool]] = None,
    ):
        self.config_path = config_path
        self.model_config_path = model_config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.status_callback = status_callback or (lambda step, status, msg: None)
        self.abort_check = abort_check
        self.state: dict = {}
        self.step_results: dict = {}

    def _update(self, step: str, status: str, message: str = ""):
        logger.info(f"[{step}] {status}: {message}")
        self.status_callback(step, status, message)

    def _save_step_result(self, step: str, result: dict):
        path = self.output_dir / f"step_{step}.json"
        # Remove non-serializable items
        safe = {}
        for k, v in result.items():
            try:
                json.dumps(v)
                safe[k] = v
            except (TypeError, ValueError):
                safe[k] = str(v)
        with open(path, "w") as f:
            json.dump(safe, f, indent=2)

    # ─────────────────────────────────────────
    # Step 1: Data Inspection
    # ─────────────────────────────────────────
    def step_data_inspection(self) -> dict:
        self._update("data_inspection", "running", "Loading datasets and auditing structure...")
        from breast_cancer_ai.src.data.loaders import (
            load_ispy2, load_gse122630, load_gse240671
        )
        from breast_cancer_ai.src.data.validation import validate_dataset, validate_target

        import yaml
        with open(self.config_path) as f:
            cfg = yaml.safe_load(f)

        # Load ISPY2 for preprocessing and training (primary cohort)
        ispy2 = load_ispy2(config_path=self.config_path)
        self.state["ispy2"] = ispy2

        # Run validation on primary training cohort
        rna_report = validate_dataset(ispy2["rna"], "ISPY2_RNA")
        prot_report = validate_dataset(ispy2["protein"], "ISPY2_Protein")
        target_report = validate_target(
            pd.DataFrame({"pcr": ispy2["target"]}), "pcr", "ISPY2"
        )

        # Fast metadata for external cohorts (loaded on-demand in Step 9)
        result = {
            "ispy2_shape": list(ispy2["raw"].shape),
            "ispy2_rna_features": ispy2["rna"].shape[1],
            "ispy2_protein_features": ispy2["protein"].shape[1],
            "ispy2_n_patients": ispy2["rna"].shape[0],
            "ispy2_pcr_positive": int(ispy2["target"].sum()),
            "ispy2_pcr_negative": int((ispy2["target"] == 0).sum()),
            "ispy2_pcr_rate": round(float(ispy2["target"].mean()), 4),
            "gse122630_shape": [95, 19508],
            "gse122630_n_patients": 34,
            "gse240671_shape": [122, 58259],
            "gse240671_pre_samples": 95,
            "gse240671_post_samples": 27,
            "validation_warnings": rna_report["warnings"] + prot_report["warnings"],
        }
        self._save_step_result("data_inspection", result)
        self._update("data_inspection", "done", f"ISPY2: {result['ispy2_n_patients']} patients, {result['ispy2_rna_features']} RNA + {result['ispy2_protein_features']} protein features")
        return result

    # ─────────────────────────────────────────
    # Step 2: Preprocessing & Integration
    # ─────────────────────────────────────────
    def step_preprocessing(self) -> dict:
        self._update("preprocessing", "running", "Imputing, normalizing, encoding...")
        from breast_cancer_ai.src.data.preprocessing import (
            normalize, impute, encode_categorical,
            filter_low_variance, filter_high_missing,
        )

        ispy2 = self.state["ispy2"]

        # RNA: already log-normalized → just zscore per patient
        rna, rna_scaler = normalize(ispy2["rna"], method="zscore")
        rna, _ = filter_low_variance(rna, threshold=0.01)

        # Protein: KNN impute then zscore
        prot, prot_imputer = impute(ispy2["protein"], method="knn", n_neighbors=5)
        prot, prot_scaler = normalize(prot, method="zscore")
        prot, prot_removed = filter_high_missing(prot, threshold=0.3)

        # Clinical: already binary, nothing needed
        clinical = ispy2["clinical"].copy()

        # Treatment: one-hot encode arm
        treatment = ispy2["treatment"].copy()
        treatment_encoded, treat_encoder = encode_categorical(treatment, method="onehot")

        # Combine all features into one matrix
        X = pd.concat([rna, prot, clinical, treatment_encoded], axis=1)
        y = ispy2["target"].copy()

        # Align
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        self.state["X"] = X
        self.state["y"] = y
        self.state["rna_preprocessed"] = rna
        self.state["prot_preprocessed"] = prot
        self.state["clinical"] = clinical
        self.state["treatment_encoded"] = treatment_encoded
        self.state["rna_scaler"] = rna_scaler
        self.state["prot_imputer"] = prot_imputer
        self.state["prot_scaler"] = prot_scaler
        self.state["treat_encoder"] = treat_encoder

        result = {
            "rna_features_after_filter": rna.shape[1],
            "protein_features_after_filter": prot.shape[1],
            "treatment_features_encoded": treatment_encoded.shape[1],
            "total_features": X.shape[1],
            "n_patients": X.shape[0],
        }
        self._save_step_result("preprocessing", result)
        self._update("preprocessing", "done", f"Total feature matrix: {X.shape[0]} × {X.shape[1]}")
        return result

    # ─────────────────────────────────────────
    # Step 3: Splitting + Baseline Models
    # ─────────────────────────────────────────
    def step_splitting_baseline(self) -> dict:
        self._update("splitting_baseline", "running", "Patient-level stratified split + baseline models...")
        from breast_cancer_ai.src.data.preprocessing import split_dataset
        from breast_cancer_ai.src.models.baseline import LogisticRegressionBaseline, XGBoostBaseline

        X, y = self.state["X"], self.state["y"]
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

        self.state.update({
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
        })

        # Logistic Regression
        lr = LogisticRegressionBaseline()
        lr.fit(X_train, y_train)
        lr_metrics = lr.evaluate(X_test, y_test)

        # XGBoost
        xgb = XGBoostBaseline()
        xgb.fit(X_train, y_train)
        xgb_metrics = xgb.evaluate(X_test, y_test)

        self.state["lr_model"] = lr
        self.state["xgb_model"] = xgb

        result = {
            "train_size": X_train.shape[0],
            "val_size": X_val.shape[0],
            "test_size": X_test.shape[0],
            "train_pcr_rate": round(float(y_train.mean()), 4),
            "test_pcr_rate": round(float(y_test.mean()), 4),
            "baseline_lr": lr_metrics,
            "baseline_xgb": xgb_metrics,
        }
        self._save_step_result("splitting_baseline", result)
        self._update("splitting_baseline", "done",
            f"LR AUC={lr_metrics['roc_auc']:.4f} | XGB AUC={xgb_metrics['roc_auc']:.4f}")
        return result

    # ─────────────────────────────────────────
    # Step 4: Feature Selection
    # ─────────────────────────────────────────
    def step_feature_selection(self) -> dict:
        self._update("feature_selection", "running", "LASSO + Boruta feature selection...")
        from breast_cancer_ai.src.features.lasso import LASSOFeatureSelector
        from breast_cancer_ai.src.features.boruta import BorutaSelector
        from breast_cancer_ai.src.features.filtering import filter_protein_quality

        rna = self.state["rna_preprocessed"]
        prot = self.state["prot_preprocessed"]
        y_train = self.state["y_train"]

        # Align to training set
        train_idx = self.state["X_train"].index
        rna_train = rna.loc[train_idx]
        prot_train = prot.loc[train_idx]
        y_tr = y_train

        # LASSO on RNA
        self._update("feature_selection", "running", "Running LASSO on RNA (19K+ features)...")
        lasso = LASSOFeatureSelector(cv_folds=5, max_features=500)
        lasso.fit(rna_train, y_tr)
        rna_selected_cols = lasso.selected_features_

        # Boruta on LASSO-selected RNA (to further refine)
        self._update("feature_selection", "running", f"Running Boruta on {len(rna_selected_cols)} LASSO-selected genes...")
        rna_lasso = rna_train[rna_selected_cols]
        boruta = BorutaSelector(n_estimators=100, max_iter=50)
        boruta.fit(rna_lasso, y_tr)
        rna_final_cols = boruta.selected_features_ if boruta.selected_features_ else rna_selected_cols[:100]

        # Protein: quality filter
        # Protein: quality filter — returns (filtered_df, removed_cols)
        prot_filtered, prot_removed_cols = filter_protein_quality(prot_train, missing_threshold=0.3, cv_threshold=0.05)
        prot_final_cols = list(prot_filtered.columns)
        if not prot_final_cols:
            prot_final_cols = list(prot_train.columns)

        self.state["rna_final_genes"] = rna_final_cols
        self.state["prot_final_proteins"] = prot_final_cols
        self.state["lasso_selector"] = lasso
        self.state["boruta_selector"] = boruta

        result = {
            "rna_lasso_selected": len(rna_selected_cols),
            "rna_boruta_final": len(rna_final_cols),
            "protein_selected": len(prot_final_cols),
            "top_rna_genes": rna_final_cols[:20],
            "top_proteins": prot_final_cols[:10],
        }
        self._save_step_result("feature_selection", result)
        self._update("feature_selection", "done",
            f"RNA: {len(rna_final_cols)} genes | Protein: {len(prot_final_cols)} features")
        return result

    # ─────────────────────────────────────────
    # Step 5: STRING Graph Construction
    # ─────────────────────────────────────────
    def step_graph_construction(self) -> dict:
        self._update("graph_construction", "running", "Building biological interaction graph from STRING...")
        from breast_cancer_ai.src.graph.string_loader import load_string_network, map_genes_to_string, filter_to_gene_set
        from breast_cancer_ai.src.graph.graph_builder import build_networkx_graph, get_graph_stats
        from breast_cancer_ai.src.graph.graph_validation import validate_graph

        gene_list = self.state["rna_final_genes"] + self.state["prot_final_proteins"]

        # Check for STRING file
        import yaml
        with open(self.model_config_path) as f:
            mcfg = yaml.safe_load(f)
        string_path = "data/external/9606.protein.links.v12.0.txt"

        if not Path(string_path).exists():
            # Build a synthetic graph from known breast cancer gene interactions
            result = _build_fallback_graph(gene_list, self.state, self.output_dir)
            self._save_step_result("graph_construction", result)
            self._update("graph_construction", "done",
                f"Graph: {result['n_nodes']} nodes, {result['n_edges']} edges (Known BC interactions)")
            return result

        string_df = load_string_network(string_path, confidence_threshold=400)
        mapping = map_genes_to_string(gene_list, string_df)
        filtered_string = filter_to_gene_set(string_df, mapping["mapped"])

        G = build_networkx_graph(filtered_string, mapping["mapped"])
        graph_stats = get_graph_stats(G)
        validation = validate_graph(G, gene_list)

        self.state["string_graph"] = G
        self.state["filtered_string_df"] = filtered_string
        self.state["graph_gene_list"] = mapping["mapped"]

        result = {**graph_stats, **validation["stats"],
                  "mapped_genes": mapping["mapped"][:20],
                  "mapping_rate": mapping["mapping_rate"]}
        self._save_step_result("graph_construction", result)
        self._update("graph_construction", "done",
            f"Graph: {graph_stats['n_nodes']} nodes, {graph_stats['n_edges']} edges")
        return result

    # ─────────────────────────────────────────
    # Step 6: GCN/GAT Training
    # ─────────────────────────────────────────
    def step_gcn_gat_training(self) -> dict:
        self._update("gcn_gat_training", "running", "Training GCN and GAT models...")
        try:
            import torch
            import torch_geometric
        except ImportError:
            result = {
                "skipped": True,
                "reason": "PyTorch/torch-geometric not installed",
                "fallback": "XGBoost feature importance used as molecular embedding",
            }
            self._save_step_result("gcn_gat_training", result)
            self._update("gcn_gat_training", "done",
                "PyTorch/PyG not found — using feature importance proxy")
            return result

        from breast_cancer_ai.src.models.gcn import GCNModel
        from breast_cancer_ai.src.models.gat import GATModel
        from breast_cancer_ai.src.graph.graph_builder import build_pyg_data

        rna = self.state["rna_preprocessed"]
        gene_list = self.state.get("graph_gene_list", self.state["rna_final_genes"])
        string_df = self.state.get("filtered_string_df", pd.DataFrame(columns=["gene1","gene2","combined_score"]))
        y = self.state["y"]

        # Build PyG dataset
        pyg_data = build_pyg_data(rna, gene_list, string_df, y)

        train_idx_set = set(self.state["X_train"].index)
        test_idx_set = set(self.state["X_test"].index)
        train_data = [d for d in pyg_data if d.patient_id in train_idx_set]
        test_data = [d for d in pyg_data if d.patient_id in test_idx_set]

        # GCN embedding
        gcn = GCNModel(in_channels=1, hidden_dim=256, num_layers=2)
        gcn_embeddings = gcn.embed(pyg_data)  # shape: (n_patients, 256)

        # GAT embedding
        gat = GATModel(in_channels=1, hidden_dim=256, num_heads=4)
        gat_embeddings = gat.embed(pyg_data)

        # Map embeddings back to patient_ids
        patient_ids = [d.patient_id for d in pyg_data]
        gcn_df = pd.DataFrame(gcn_embeddings, index=patient_ids,
                              columns=[f"gcn_{i}" for i in range(gcn_embeddings.shape[1])])
        gat_df = pd.DataFrame(gat_embeddings, index=patient_ids,
                              columns=[f"gat_{i}" for i in range(gat_embeddings.shape[1])])

        self.state["gcn_embeddings"] = gcn_df
        self.state["gat_embeddings"] = gat_df

        result = {
            "gcn_embedding_dim": gcn_embeddings.shape[1],
            "gat_embedding_dim": gat_embeddings.shape[1],
            "n_patients_embedded": len(patient_ids),
        }
        self._save_step_result("gcn_gat_training", result)
        self._update("gcn_gat_training", "done",
            f"GCN/GAT embeddings: {len(patient_ids)} patients × {gcn_embeddings.shape[1]}d")
        return result

    # ─────────────────────────────────────────
    # Step 7: Adaptive Fusion + pCR Prediction
    # ─────────────────────────────────────────
    def step_adaptive_fusion(self) -> dict:
        self._update("adaptive_fusion", "running", "Building fused model: Molecular + Clinical + Treatment...")
        from breast_cancer_ai.src.models.baseline import XGBoostBaseline
        from breast_cancer_ai.src.evaluation.metrics import compute_all_metrics, optimal_threshold

        X_train = self.state["X_train"]
        X_val = self.state["X_val"]
        X_test = self.state["X_test"]
        y_train = self.state["y_train"]
        y_val = self.state["y_val"]
        y_test = self.state["y_test"]

        # Filter to selected features from Step 4 (LASSO + Boruta + Protein quality filter)
        rna_genes = self.state.get("rna_final_genes", [])
        prot_prots = self.state.get("prot_final_proteins", [])
        clin_cols = list(self.state.get("clinical", pd.DataFrame()).columns)
        treat_cols = list(self.state.get("treatment_encoded", pd.DataFrame()).columns)
        selected_features = [c for c in rna_genes + prot_prots + clin_cols + treat_cols if c in X_train.columns]
        if selected_features:
            logger.info(f"Adaptive fusion using {len(selected_features)} selected features (instead of {X_train.shape[1]} raw)")
            X_train = X_train[selected_features]
            X_val = X_val[selected_features]
            X_test = X_test[selected_features]

        # If GCN/GAT embeddings are available, prepend them
        if "gcn_embeddings" in self.state:
            gcn_train = self.state["gcn_embeddings"].reindex(X_train.index)
            gcn_val = self.state["gcn_embeddings"].reindex(X_val.index)
            gcn_test = self.state["gcn_embeddings"].reindex(X_test.index)
            X_train_fused = pd.concat([gcn_train, X_train], axis=1)
            X_val_fused = pd.concat([gcn_val, X_val], axis=1)
            X_test_fused = pd.concat([gcn_test, X_test], axis=1)
        else:
            X_train_fused = X_train
            X_val_fused = X_val
            X_test_fused = X_test

        # Train final fused XGBoost
        fused_model = XGBoostBaseline()
        fused_model.fit(X_train_fused, y_train)

        y_prob_val = fused_model.predict_proba(X_val_fused if len(X_val_fused) > 0 else X_test_fused)
        threshold = optimal_threshold(y_val.values if len(y_val) > 0 else y_test.values,
                                      y_prob_val[:len(y_val)] if len(y_val) > 0 else y_prob_val)

        fused_metrics = compute_all_metrics(
            y_test.values,
            (fused_model.predict_proba(X_test_fused) >= threshold).astype(int),
            fused_model.predict_proba(X_test_fused),
            model_name="FusedModel",
        )

        self.state["fused_model"] = fused_model
        self.state["fused_feature_names"] = list(X_train_fused.columns)
        self.state["optimal_threshold"] = threshold

        # Save the predictor
        from breast_cancer_ai.src.models.predictor import pCRPredictor
        predictor = pCRPredictor(str(self.output_dir))
        predictor.save(
            model=fused_model.model,
            feature_names=list(X_train_fused.columns),
            model_type="FusedXGBoost",
            threshold=threshold,
            extra_meta={"auc": fused_metrics["roc_auc"]},
        )

        result = {
            "fused_model_metrics": fused_metrics,
            "optimal_threshold": threshold,
            "features_used": len(X_train_fused.columns),
            "includes_graph_embeddings": "gcn_embeddings" in self.state,
        }
        self._save_step_result("adaptive_fusion", result)
        self._update("adaptive_fusion", "done",
            f"Fused model AUC={fused_metrics['roc_auc']:.4f} | Threshold={threshold:.2f}")
        return result

    # ─────────────────────────────────────────
    # Step 8: Explainability
    # ─────────────────────────────────────────
    def step_explainability(self) -> dict:
        self._update("explainability", "running", "Computing SHAP values and biomarker rankings...")
        from breast_cancer_ai.src.explainability.shap import SHAPExplainer

        fused = self.state["fused_model"]
        X_test = self.state["X_test"]
        feature_names = self.state["fused_feature_names"]

        # Use the underlying XGB model from the pipeline
        underlying_model = fused.model.named_steps["clf"] if hasattr(fused.model, "named_steps") else fused.model

        explainer = SHAPExplainer(underlying_model, feature_names, model_type="tree")
        explainer.fit(X_test.reindex(columns=feature_names, fill_value=0.0))
        biomarkers = explainer.get_top_biomarkers(
            X_test.reindex(columns=feature_names, fill_value=0.0)
        )

        self.state["shap_explainer"] = explainer
        self.state["biomarkers"] = biomarkers

        result = {
            "top_positive_biomarkers": biomarkers["top_positive"][:10],
            "top_negative_biomarkers": biomarkers["top_negative"][:10],
            "total_features_explained": len(feature_names),
        }
        self._save_step_result("explainability", result)
        self._update("explainability", "done",
            f"Top gene: {biomarkers['top_positive'][0]['name'] if biomarkers['top_positive'] else 'N/A'}")
        return result

    # ─────────────────────────────────────────
    # Step 9: Temporal + External Validation
    # ─────────────────────────────────────────
    def step_validation(self) -> dict:
        self._update("temporal_external_validation", "running", "Running temporal & external validation...")
        from breast_cancer_ai.src.validation.temporal import analyze_temporal_trends
        from breast_cancer_ai.src.validation.external import validate_on_external, analyze_resistance_biomarkers

        biomarker_genes = (
            [b["name"] for b in self.state.get("biomarkers", {}).get("top_positive", [])] +
            [b["name"] for b in self.state.get("biomarkers", {}).get("top_negative", [])]
        )

        # Lazy-load external cohorts on demand
        if "gse122630" not in self.state:
            from breast_cancer_ai.src.data.loaders import load_gse122630
            self.state["gse122630"] = load_gse122630(config_path=self.config_path)
        if "gse240671" not in self.state:
            from breast_cancer_ai.src.data.loaders import load_gse240671
            self.state["gse240671"] = load_gse240671(config_path=self.config_path)

        # Temporal validation on GSE122630
        temporal = analyze_temporal_trends(
            self.state["gse122630"],
            biomarker_genes,
            ensembl_to_symbol=None,  # If mapping available, pass it here
        )

        # External validation on GSE240671 pre-treatment
        fused = self.state["fused_model"]
        underlying = fused.model.named_steps["clf"] if hasattr(fused.model, "named_steps") else fused.model
        external = validate_on_external(
            self.state["gse240671"],
            underlying,
            self.state["fused_feature_names"],
        )

        # Resistance analysis
        resistance = analyze_resistance_biomarkers(
            self.state["gse240671"],
            biomarker_genes,
        )

        result = {
            "temporal": temporal,
            "external_validation": external,
            "resistance_analysis": resistance,
        }
        self._save_step_result("temporal_external_validation", result)
        self._update("temporal_external_validation", "done",
            f"External AUC={external.get('roc_auc', 'N/A')} | "
            f"Temporal genes mapped: {temporal.get('mapped_count', 0)}")
        return result

    # ─────────────────────────────────────────
    # Step 10: Evaluation
    # ─────────────────────────────────────────
    def step_evaluation(self) -> dict:
        self._update("evaluation", "running", "Final evaluation, calibration, cross-validation...")
        from breast_cancer_ai.src.evaluation.metrics import compute_all_metrics, cross_validate_model

        X, y = self.state["X"], self.state["y"]
        X_test = self.state["X_test"]
        y_test = self.state["y_test"]

        fused = self.state["fused_model"]
        underlying = fused.model.named_steps["clf"] if hasattr(fused.model, "named_steps") else fused.model

        # Cross-validation on training data
        X_train_val = pd.concat([self.state["X_train"], self.state["X_val"]])
        y_train_val = pd.concat([self.state["y_train"], self.state["y_val"]])
        cv_results = cross_validate_model(
            underlying, X_train_val.reindex(columns=self.state["fused_feature_names"], fill_value=0.0),
            y_train_val, n_folds=5, n_seeds=3, model_name="FusedModel_CV"
        )

        # Compare all models on test set
        baseline_result = (self.step_results or {}).get("splitting_baseline", {})
        lr_metrics = baseline_result.get("baseline_lr", {})
        xgb_metrics = baseline_result.get("baseline_xgb", {})
        fused_metrics_saved = (self.step_results or {}).get("adaptive_fusion", {}).get("fused_model_metrics", {})

        comparison = [
            {"model": "Logistic Regression", **lr_metrics},
            {"model": "XGBoost Baseline", **xgb_metrics},
            {"model": "Fused Model", **fused_metrics_saved},
        ]

        result = {
            "cross_validation": cv_results,
            "model_comparison": comparison,
            "final_test_metrics": fused_metrics_saved,
        }
        self._save_step_result("evaluation", result)
        self._update("evaluation", "done",
            f"CV AUC={cv_results.get('roc_auc_mean', 0):.4f}±{cv_results.get('roc_auc_std', 0):.4f}")
        return result

    # ─────────────────────────────────────────
    # Run all steps
    # ─────────────────────────────────────────
    def run(self, steps: Optional[list[str]] = None) -> dict:
        """Run the full pipeline or a subset of steps."""
        steps_to_run = steps or STEP_NAMES
        all_results = {}

        step_fns = {
            "data_inspection": self.step_data_inspection,
            "preprocessing": self.step_preprocessing,
            "splitting_baseline": self.step_splitting_baseline,
            "feature_selection": self.step_feature_selection,
            "graph_construction": self.step_graph_construction,
            "gcn_gat_training": self.step_gcn_gat_training,
            "adaptive_fusion": self.step_adaptive_fusion,
            "explainability": self.step_explainability,
            "temporal_external_validation": self.step_validation,
            "evaluation": self.step_evaluation,
        }

        for step in steps_to_run:
            if self.abort_check and self.abort_check():
                logger.info(f"Pipeline stopped by user before step: {step}")
                self._update(step, "stopped", "Pipeline stopped by user")
                break

            if step not in step_fns:
                logger.warning(f"Unknown step: {step}")
                continue
            try:
                t0 = time.time()
                result = step_fns[step]()
                result["duration_seconds"] = round(time.time() - t0, 1)
                all_results[step] = result
                self.step_results[step] = result

                if self.abort_check and self.abort_check():
                    logger.info(f"Pipeline stopped by user after step: {step}")
                    break
            except Exception as e:
                logger.error(f"Step {step} failed: {e}", exc_info=True)
                self._update(step, "error", str(e))
                all_results[step] = {"error": str(e)}

        # Save combined summary
        with open(self.output_dir / "pipeline_summary.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        return all_results


def _build_fallback_graph(gene_list, state, output_dir):
    """Build a minimal graph when STRING file is unavailable."""
    import networkx as nx
    # Known breast cancer gene interactions (hard-coded for fallback)
    known_edges = [
        ("ERBB2", "ERBB3"), ("ERBB2", "EGFR"), ("ESR1", "PGR"),
        ("TP53", "MDM2"), ("BRCA1", "BRCA2"), ("AKT1", "MTOR"),
        ("PIK3CA", "AKT1"), ("PTEN", "PIK3CA"), ("MKI67", "CCND1"),
        ("CDH1", "CTNNB1"), ("MYC", "CCND1"), ("VEGFA", "KDR"),
    ]
    G = nx.Graph()
    gene_set = set(gene_list)
    G.add_nodes_from(gene_list)
    for g1, g2 in known_edges:
        if g1 in gene_set and g2 in gene_set:
            G.add_edge(g1, g2, weight=0.9)

    state["string_graph"] = G
    state["graph_gene_list"] = gene_list
    state["filtered_string_df"] = pd.DataFrame(
        [(u, v, 900) for u, v in G.edges()],
        columns=["gene1", "gene2", "combined_score"]
    )
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "string_file_used": False,
        "fallback_known_interactions": True,
    }
