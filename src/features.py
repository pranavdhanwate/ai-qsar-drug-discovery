from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


def extract_molecular_descriptors(mol: Chem.Mol) -> dict[str, float]:
    """Calculates 10 fundamental 2D physicochemical descriptors."""
    return {
        "MW": float(Descriptors.MolWt(mol)),
        "LogP": float(Crippen.MolLogP(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "HBD": float(Lipinski.NumHDonors(mol)),
        "HBA": float(Lipinski.NumHAcceptors(mol)),
        "RotBonds": float(Lipinski.NumRotatableBonds(mol)),
        "AromaticRings": float(Descriptors.NumAromaticRings(mol)),
        "HeavyAtoms": float(mol.GetNumHeavyAtoms()),
        "FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "MolMR": float(Crippen.MolMR(mol)),
    }


def generate_morgan_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 512) -> np.ndarray:
    """Generates Morgan/ECFP4 bit vector as a 1D numpy array."""
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    for bit in fp.GetOnBits():
        arr[bit] = 1.0
    return arr


def build_feature_matrix(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    include_fingerprints: bool = True,
    n_bits: int = 256,
) -> Tuple[np.ndarray, List[str]]:
    """Constructs numeric feature matrix X and corresponding feature names list."""
    desc_list = []
    fp_list = []

    for smi in df[smiles_col]:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid SMILES encountered: {smi}")

        desc_dict = extract_molecular_descriptors(mol)
        desc_list.append(list(desc_dict.values()))

        if include_fingerprints:
            fp = generate_morgan_fingerprint(mol, radius=2, n_bits=n_bits)
            fp_list.append(fp)

    desc_arr = np.array(desc_list, dtype=np.float32)
    feature_names = list(extract_molecular_descriptors(Chem.MolFromSmiles("CC")).keys())

    if include_fingerprints:
        fp_arr = np.array(fp_list, dtype=np.float32)
        X = np.hstack([desc_arr, fp_arr])
        feature_names += [f"ECFP4_bit_{i}" for i in range(n_bits)]
    else:
        X = desc_arr

    return X, feature_names
