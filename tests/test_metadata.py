from pathlib import Path

import pandas as pd

from src.data.build_metadata import EXPECTED_VARIANTS

# Percorso del file metadata centralizzato
METADATA_PATH = Path("data/processed/metadata.csv")

def load_metadata() -> pd.DataFrame:
    # Tutti i test partono dallo stesso file.
    # Se il file manca, il blocco precedente non è stato eseguito correttamente.
    assert METADATA_PATH.exists(), "Metadata non trovato."
    return pd.read_csv(METADATA_PATH)

def test_metadata_has_expected_number_of_examples() -> None:
    # Il Cartoon Set 10k deve contenere esattamente 10.000 esempi.
    metadata = load_metadata()
    assert len(metadata) == 10_000

def test_metadata_has_unique_image_ids() -> None:
    # Ogni immagine deve comparire una sola volta nel metadata.
    metadata = load_metadata()
    assert metadata["image_id"].nunique() == 10_000
    assert not metadata["image_id"].duplicated().any()

def test_all_expected_columns_are_present() -> None:
    # Le colonne attese includono gli identificativi base
    # più tutti i 18 attributi discreti del dataset.
    metadata = load_metadata()

    expected_columns = {
        "image_id",
        "image_path",
        "csv_path",
        *EXPECTED_VARIANTS.keys(),
    }

    assert set(metadata.columns) == expected_columns

def test_all_attribute_values_are_in_valid_range() -> None:
    # Ogni attributo discreto deve avere valori interi
    # compresi tra 0 e num_variants - 1.
    metadata = load_metadata()

    for attribute, num_variants in EXPECTED_VARIANTS.items():
        assert metadata[attribute].min() >= 0
        assert metadata[attribute].max() == num_variants - 1

def test_all_attributes_cover_all_expected_variants() -> None:
    # Verifichiamo che per ogni attributo siano presenti
    # tutte le varianti dichiarate nella specifica del dataset.
    metadata = load_metadata()

    for attribute, num_variants in EXPECTED_VARIANTS.items():
        observed_values = sorted(metadata[attribute].unique().tolist())
        expected_values = list(range(num_variants))
        assert observed_values == expected_values

def test_all_referenced_files_exist() -> None:
    # I path salvati nel metadata devono puntare a file reali
    # dentro la directory raw del dataset.
    metadata = load_metadata()
    dataset_root = Path("data/raw/cartoonset10k")

    for row in metadata.itertuples(index=False):
        assert (dataset_root / row.image_path).is_file()
        assert (dataset_root / row.csv_path).is_file()
