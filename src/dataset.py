from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split


def load_and_validate_dataset(
    filepath: str | Path,
    smiles_col: str = "smiles",
    activity_col: str = "pIC50",
    id_col: str = "molecule_id",
) -> pd.DataFrame:
    """Loads CSV, validates chemical structures, and removes invalid SMILES."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    clean_records = []

    for idx, row in df.iterrows():
        smi = str(row[smiles_col]).strip()
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        # Compute Bemis-Murcko core scaffold
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=canonical_smiles)

        record = row.to_dict()
        record["canonical_smiles"] = canonical_smiles
        record["scaffold"] = scaffold
        clean_records.append(record)

    clean_df = pd.DataFrame(clean_records)
    clean_df[activity_col] = pd.to_numeric(clean_df[activity_col], errors="coerce")
    return clean_df.dropna(subset=[activity_col]).reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    stratify_by_scaffold: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits dataset into independent training and test sets to prevent data leakage."""
    if stratify_by_scaffold and "scaffold" in df.columns:
        # Group by scaffold to ensure test set contains distinct chemical series
        scaffolds = df["scaffold"].unique()
        train_scaffolds, test_scaffolds = train_test_split(
            scaffolds, test_size=test_size, random_state=random_state
        )
        train_df = df[df["scaffold"].isin(train_scaffolds)].copy()
        test_df = df[df["scaffold"].isin(test_scaffolds)].copy()
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
