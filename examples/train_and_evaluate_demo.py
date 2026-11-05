import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import run_qsar_pipeline

if __name__ == "__main__":
    csv_path = project_root / "data" / "bace1_activity.csv"
    out_dir = project_root / "results"

    print("Running AI-Based QSAR Training & Applicability Domain Demo...")
    comp_df, ad_df = run_qsar_pipeline(csv_path, output_dir=out_dir)

    print("\nModel Benchmark Summary:")
    print(comp_df.to_string(index=False))

    print("\nTest Compounds in Applicability Domain:")
    print(ad_df.head(6).to_string(index=False))
