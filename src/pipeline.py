import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from .dataset import load_and_validate_dataset, split_dataset
from .features import build_feature_matrix
from .selection import FeatureSelector
from .models import QSARModelSuite
from .evaluate import compare_models, calculate_metrics
from .applicability_domain import ApplicabilityDomain
from .visualize import plot_parity, plot_williams, plot_feature_importance


def run_qsar_pipeline(
    input_csv: str | Path,
    output_dir: str | Path = "results",
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Executes complete QSAR modelling, cross-validation, and applicability domain workflow."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Loading
    df = load_and_validate_dataset(input_csv)

    # 2. Molecular Feature Extraction
    X_raw, feat_names_raw = build_feature_matrix(df, smiles_col="canonical_smiles", n_bits=128)
    y = df["pIC50"].values

    # 3. Train-Test Partitioning (Avoid Data Leakage)
    indices = np.arange(len(df))
    np.random.seed(random_state)
    np.random.shuffle(indices)
    split_idx = int(len(df) * (1.0 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 4. Feature Selection fitted strictly on training data
    selector = FeatureSelector(variance_threshold=0.01, correlation_threshold=0.95)
    X_train, feat_names = selector.fit_transform(X_train_raw, feat_names_raw)
    X_test = selector.transform(X_test_raw)

    # 5. Model Suite Training & Tuning
    suite = QSARModelSuite(random_state=random_state)
    suite.tune_hyperparameters(X_train, y_train)
    suite.train_all(X_train, y_train)

    # 6. Evaluation & Model Comparison
    comparison_df = compare_models(suite.models, X_train, y_train, X_test, y_test)
    comparison_df.to_csv(out_dir / "model_comparison.csv", index=False)

    # 7. Select Best Model (XGBoost / Random Forest)
    best_model_name = "XGBoost" if "XGBoost" in suite.models else "Random Forest"
    best_model = suite.models[best_model_name]

    train_preds = best_model.predict(X_train)
    test_preds = best_model.predict(X_test)

    # 8. Applicability Domain & Williams Plot
    ad = ApplicabilityDomain()
    ad.fit(X_train, y_train, train_preds)
    ad_df = ad.evaluate_test_set(X_test, y_test, test_preds)
    ad_df.to_csv(out_dir / "applicability_domain_test.csv", index=False)

    # 9. Publication Visualizations
    plot_parity(y_train, train_preds, y_test, test_preds, fig_dir / "parity_plot.png", model_name=best_model_name)
    plot_williams(ad_df, ad.warning_leverage_, fig_dir / "williams_plot.png")

    if hasattr(best_model, "feature_importances_"):
        plot_feature_importance(feat_names, best_model.feature_importances_, fig_dir / "feature_importance.png")

    return comparison_df, ad_df


def main():
    parser = argparse.ArgumentParser(description="AI-Based QSAR and Bioactivity Prediction")
    parser.add_argument("--input", "-i", type=str, default="data/bace1_activity.csv", help="Input bioactivity CSV")
    parser.add_argument("--output", "-o", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    comp_df, ad_df = run_qsar_pipeline(args.input, output_dir=args.output)
    print("\nQSAR Pipeline Execution Complete! Model Comparison Table:")
    print(comp_df.to_string(index=False))
    print(f"\nApplicability Domain summary (Test Set n={len(ad_df)}):")
    print(ad_df["domain_classification"].value_counts().to_string())


if __name__ == "__main__":
    main()
