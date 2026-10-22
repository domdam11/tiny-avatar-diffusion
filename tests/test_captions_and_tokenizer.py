from pathlib import Path

import pandas as pd
import re
from tokenizers import Tokenizer

from src.data.build_captions import (
    generate_caption,
    load_text_config,
)
from src.data.build_metadata import EXPECTED_VARIANTS
from src.data.build_splits import VALID_SPLITS


CONFIG_PATH = Path("configs/text_conditioning.yaml")
CAPTIONS_PATH = Path(
    "data/processed/captions/metadata_with_captions.csv"
)
TOKENIZER_PATH = Path(
    "data/processed/tokenizer/tokenizer.json"
)


def load_inputs() -> tuple[dict, pd.DataFrame, Tokenizer]:
    # Carica gli artefatti creati nel blocco captioning
    assert CONFIG_PATH.exists()
    assert CAPTIONS_PATH.exists()
    assert TOKENIZER_PATH.exists()

    config = load_text_config(CONFIG_PATH)
    captioned = pd.read_csv(CAPTIONS_PATH)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    return config, captioned, tokenizer


def test_caption_dataset_has_expected_size() -> None:
    # Le caption devono coprire tutti gli esempi del dataset
    _, captioned, _ = load_inputs()

    assert len(captioned) == 10_000
    assert captioned["image_id"].nunique() == 10_000


def test_caption_dataset_preserves_splits() -> None:
    # Le assegnazioni split devono restare invariate
    _, captioned, _ = load_inputs()

    assert set(captioned["split"].unique()) == set(
        VALID_SPLITS
    )


def test_captions_are_not_empty() -> None:
    # Ogni esempio deve avere una caption testuale non vuota
    _, captioned, _ = load_inputs()

    assert captioned["caption"].notna().all()
    assert captioned["caption"].str.strip().ne("").all()


def test_caption_generation_is_deterministic() -> None:
    # La stessa riga di metadata deve generare la stessa caption
    config, captioned, _ = load_inputs()

    row = captioned.iloc[0].to_dict()

    first_caption = generate_caption(
        row=row,
        caption_config=config["caption"],
    )
    second_caption = generate_caption(
        row=row,
        caption_config=config["caption"],
    )

    assert first_caption == second_caption
    assert first_caption == row["caption"]


def test_caption_values_are_in_expected_ranges() -> None:
    # Le caption si basano su valori discreti validi
    _, captioned, _ = load_inputs()

    for attribute, num_variants in EXPECTED_VARIANTS.items():
        assert captioned[attribute].min() >= 0
        assert captioned[attribute].max() < num_variants


def test_tokenizer_contains_special_tokens() -> None:
    # Tutti i token speciali devono avere un ID assegnato
    config, _, tokenizer = load_inputs()

    for token in config["tokenizer"]["special_tokens"].values():
        assert tokenizer.token_to_id(token) is not None


def test_train_captions_roundtrip_without_unknown_tokens() -> None:
    # Le caption train devono essere rappresentabili senza [UNK].
    # Il tokenizer restituisce anche [SOS], [EOS] e padding:
    # per confrontare il testo originale consideriamo solo i token della caption.
    config, captioned, tokenizer = load_inputs()

    special = config["tokenizer"]["special_tokens"]
    special_tokens = {
        special["sos"],
        special["eos"],
        special["pad"],
    }
    unk_token = special["unk"]

    train_captions = captioned.loc[
        captioned["split"] == "train",
        "caption",
    ].tolist()

    for caption in train_captions[:100]:
        encoding = tokenizer.encode(caption)

        # Il vocabolario è addestrato sulle sole caption di training:
        # nessuna caption train deve produrre token sconosciuti.
        assert unk_token not in encoding.tokens

        # Rimuoviamo i token tecnici aggiunti dal post-processing
        # prima del confronto con la tokenizzazione attesa della caption.
        caption_tokens = [
            token
            for token in encoding.tokens
            if token not in special_tokens
        ]

        assert caption_tokens == re.findall(r"\w+|[^\w\s]", caption)
