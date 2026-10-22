from pathlib import Path

import torch
import yaml
from torch import Tensor, nn


def load_text_encoder_config(config_path: Path) -> dict:
    # Carica il file YAML della configurazione.
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Verifica la presenza della sezione principale.
    if "text_encoder" not in config:
        raise ValueError("Sezione 'text_encoder' mancante.")

    encoder_config = config["text_encoder"]

    # Chiavi minime richieste dal modello.
    required_keys = {
        "vocab_size",
        "max_sequence_length",
        "pad_token_id",
        "d_model",
        "num_heads",
        "num_layers",
        "feedforward_dim",
        "dropout",
        "positional_encoding",
        "normalization",
        "activation",
    }

    missing_keys = required_keys - set(encoder_config)
    if missing_keys:
        raise ValueError(
            "Parametri text encoder mancanti: "
            f"{sorted(missing_keys)}"
        )

    # Controlli di validità di base.
    if int(encoder_config["vocab_size"]) <= 0:
        raise ValueError("vocab_size deve essere positivo.")

    if int(encoder_config["max_sequence_length"]) <= 0:
        raise ValueError("max_sequence_length deve essere positivo.")

    if int(encoder_config["pad_token_id"]) < 0:
        raise ValueError("pad_token_id non può essere negativo.")

    if int(encoder_config["d_model"]) <= 0:
        raise ValueError("d_model deve essere positivo.")

    if int(encoder_config["num_heads"]) <= 0:
        raise ValueError("num_heads deve essere positivo.")

    if int(encoder_config["num_layers"]) <= 0:
        raise ValueError("num_layers deve essere positivo.")

    if int(encoder_config["feedforward_dim"]) <= 0:
        raise ValueError("feedforward_dim deve essere positivo.")

    if float(encoder_config["dropout"]) < 0.0 or float(encoder_config["dropout"]) >= 1.0:
        raise ValueError("dropout deve stare in [0, 1).")

    if int(encoder_config["d_model"]) % int(encoder_config["num_heads"]) != 0:
        raise ValueError("d_model deve essere divisibile per num_heads.")

    if str(encoder_config["positional_encoding"]) != "sinusoidal":
        raise ValueError("È supportato solo positional_encoding='sinusoidal'.")

    if str(encoder_config["normalization"]) != "pre_norm":
        raise ValueError("È supportato solo normalization='pre_norm'.")

    if str(encoder_config["activation"]) != "gelu":
        raise ValueError("È supportato solo activation='gelu'.")

    return config


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_sequence_length: int) -> None:
        super().__init__()

        if int(d_model) <= 0:
            raise ValueError("d_model deve essere positivo.")
        if int(max_sequence_length) <= 0:
            raise ValueError("max_sequence_length deve essere positivo.")

        # Costruisce una matrice fissa di positional encoding sinusoidale.
        position = torch.arange(max_sequence_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0, dtype=torch.float32)) / d_model)
        )

        positional_encoding = torch.zeros(
            max_sequence_length,
            d_model,
            dtype=torch.float32,
        )
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term[: positional_encoding[:, 1::2].shape[1]])

        # Buffer: si sposta con il modello ma non è un parametro addestrabile.
        self.register_buffer(
            "positional_encoding",
            positional_encoding.unsqueeze(0),
            persistent=False,
        )

    def forward(self, token_embeddings: Tensor) -> Tensor:
        if token_embeddings.ndim != 3:
            raise ValueError(
                "token_embeddings deve avere shape [batch, sequence_length, d_model]."
            )

        sequence_length = token_embeddings.shape[1]
        if sequence_length > self.positional_encoding.shape[1]:
            raise ValueError("sequence_length supera il massimo supportato.")

        return token_embeddings + self.positional_encoding[:, :sequence_length, :]


class TextTransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        pad_token_id: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
        positional_encoding: str,
        normalization: str,
        activation: str,
    ) -> None:
        super().__init__()

        # Attributi esposti anche ad altri componenti della pipeline.
        self.vocab_size = int(vocab_size)
        self.max_sequence_length = int(max_sequence_length)
        self.pad_token_id = int(pad_token_id)
        self.d_model = int(d_model)

        # Embedding dei token.
        self.token_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_model,
            padding_idx=self.pad_token_id,
        )

        self.position_encoding = SinusoidalPositionalEncoding(
            d_model=self.d_model,
            max_sequence_length=self.max_sequence_length,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=int(feedforward_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=int(num_layers),
            enable_nested_tensor=False,
        )

        self.final_dropout = nn.Dropout(float(dropout))

    @classmethod
    def from_config(cls, config: dict) -> "TextTransformerEncoder":
        encoder_config = config["text_encoder"]
        return cls(
            vocab_size=int(encoder_config["vocab_size"]),
            max_sequence_length=int(encoder_config["max_sequence_length"]),
            pad_token_id=int(encoder_config["pad_token_id"]),
            d_model=int(encoder_config["d_model"]),
            num_heads=int(encoder_config["num_heads"]),
            num_layers=int(encoder_config["num_layers"]),
            feedforward_dim=int(encoder_config["feedforward_dim"]),
            dropout=float(encoder_config["dropout"]),
            positional_encoding=str(encoder_config["positional_encoding"]),
            normalization=str(encoder_config["normalization"]),
            activation=str(encoder_config["activation"]),
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        # Controlli dimensionali e di tipo.
        if input_ids.ndim != 2:
            raise ValueError("input_ids deve avere shape [batch, sequence_length].")

        if attention_mask.ndim != 2:
            raise ValueError("attention_mask deve avere shape [batch, sequence_length].")

        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids e attention_mask devono avere la stessa shape.")

        if input_ids.dtype != torch.long:
            raise TypeError("input_ids deve avere dtype torch.long.")

        if attention_mask.dtype not in (torch.long, torch.bool, torch.int32, torch.int64):
            raise TypeError("attention_mask deve avere dtype intero o booleano.")

        if input_ids.shape[1] > self.max_sequence_length:
            raise ValueError(
                "La sequenza supera max_sequence_length: "
                f"{input_ids.shape[1]} > {self.max_sequence_length}"
            )

        # Forziamo la mask a formato intero 0/1.
        attention_mask = attention_mask.long()

        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("attention_mask deve contenere solo 0 e 1.")

        # Embedding + positional encoding.
        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.position_encoding(hidden_states)
        hidden_states = self.final_dropout(hidden_states)

        # In PyTorch il padding mask usa True sulle posizioni da ignorare.
        key_padding_mask = attention_mask == 0

        # Output token-level per la cross-attention della U-Net.
        encoded = self.encoder(
            hidden_states,
            src_key_padding_mask=key_padding_mask,
        )

        # Azzera esplicitamente le posizioni di padding nell'output finale.
        encoded = encoded * attention_mask.unsqueeze(-1).to(encoded.dtype)

        return encoded
