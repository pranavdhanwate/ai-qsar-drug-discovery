import pytest
import pandas as pd
from src.dataset import load_and_validate_dataset, split_dataset


def test_load_and_validate_dataset(tmp_path):
    sample_csv = tmp_path / "sample.csv"
    sample_csv.write_text(
        "molecule_id,smiles,pIC50\n"
        "M1,CC(=O)Oc1ccccc1C(=O)O,6.5\n"
        "M2,INVALID_SMILES_STRING,4.2\n"
        "M3,c1ccccc1,5.0\n"
    )
    df = load_and_validate_dataset(sample_csv)

    assert len(df) == 2  # Invalid row dropped
    assert "canonical_smiles" in df.columns
    assert "scaffold" in df.columns
    assert df.iloc[0]["molecule_id"] == "M1"


def test_split_dataset_no_leakage():
    data = {
        "molecule_id": [f"M_{i}" for i in range(20)],
        "canonical_smiles": ["c1ccccc1" for _ in range(20)],
        "pIC50": [5.0 + (i * 0.1) for i in range(20)],
    }
    df = pd.DataFrame(data)
    train_df, test_df = split_dataset(df, test_size=0.25, random_state=42)

    assert len(train_df) == 15
    assert len(test_df) == 5
    # Ensure zero index overlap
    train_ids = set(train_df["molecule_id"])
    test_ids = set(test_df["molecule_id"])
    assert len(train_ids.intersection(test_ids)) == 0
