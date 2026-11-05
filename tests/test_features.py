import pytest
from rdkit import Chem
from src.features import extract_molecular_descriptors, generate_morgan_fingerprint


def test_extract_molecular_descriptors():
    # Ethanol: C2H5OH, MW ~ 46.07
    mol = Chem.MolFromSmiles("CCO")
    desc = extract_molecular_descriptors(mol)

    assert 45.0 < desc["MW"] < 47.0
    assert desc["HBD"] == 1
    assert desc["HBA"] == 1
    assert desc["AromaticRings"] == 0
    assert desc["HeavyAtoms"] == 3


def test_generate_morgan_fingerprint_shape():
    mol = Chem.MolFromSmiles("c1ccccc1")
    fp = generate_morgan_fingerprint(mol, radius=2, n_bits=128)

    assert fp.shape == (128,)
    assert fp.sum() > 0  # Should have set bits
