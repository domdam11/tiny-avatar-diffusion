
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from torch import Tensor, nn


def load_text_encoder_config(
    config_path: Path,
) -> dict:
    with Path(config_path).open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if "text_encoder" not in config:
        raise ValueError(
            "Sezione 'text_encoder' mancante."
        )

    encoder_config = config["text_encoder"]

    required_keys = {
        "vocab_size",
        "max_sequence_length",
        "pad_token_id",
        "d_model",
        "num_heads",
        "num_layers",
        "feed_forward_dim",
        "dropout",
        "positional_encoding",
        "normalization",
        "activation",
    }

    missing_keys = (
        required_keys - set(encoder_config)
    )

    if missing_keys:
        raise ValueError(
            "Parametri text encoder mancanti: "
            f"{sorted(missing_keys)}"
        )

    vocab_size = int(
        encoder_config["vocab_size"]
    )

    max_sequence_length = int(
        encoder_config["max_sequence_length"]
    )

    pad_token_id = int(
        encoder_config["pad_token_id"]
    )

    d_model = int(
        encoder_config["d_model"]
    )

    num_heads = int(
        encoder_config["num_heads"]
    )

    num_layers = int(
        encoder_config["num_layers"]
    )

    feed_forward_dim = int(
        encoder_config["feed_forward_dim"]
    )

    dropout = float(
        encoder_config["dropout"]
    )

    if vocab_size < 1:
        raise ValueError(
            "vocab_size deve essere positivo."
        )

    if max_sequence_length < 1:
        raise ValueError(
            "max_sequence_length deve essere positivo."
        )

    if not 0 <= pad_token_id < vocab_size:
        raise ValueError(
            "pad_token_id fuori intervallo."
        )

    if d_model < 1:
        raise ValueError(
            "d_model deve essere positivo."
        )

    if num_heads < 1:
        raise ValueError(
            "num_heads deve essere positivo."
        )

    if d_model % num_heads != 0:
        raise ValueError(
            "d_model deve essere divisibile "
            "per num_heads."
        )

    if num_layers < 1:
        raise ValueError(
            "num_layers deve essere positivo."
        )

    if feed_forward_dim < d_model:
        raise ValueError(
            "feed_forward_dim deve essere almeno "
            "pari a d_model."
        )

    if not 0.0 <= dropout < 1.0:
        raise ValueError(
            "dropout deve appartenere a [0, 1)."
        )

    if (
        encoder_config["positional_encoding"]
        != "sinusoidal"
    ):
        raise ValueError(
            "È supportato positional encoding "
            "sinusoidale."
        )

    if (
        encoder_config["normalization"]
        != "pre_norm"
    ):
        raise ValueError(
            "L'architettura usa pre-normalization."
        )

    if encoder_config["activation"] != "gelu":
        raise ValueError(
            "L'attivazione prevista è GELU."
        )

    return config


class SinusoidalPositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model: int,
        max_sequence_length: int,
    ) -> None:
        super().__init__()

        positions = torch.arange(
            max_sequence_length,
            dtype=torch.float32,
        ).unsqueeze(1)

        frequencies = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32,
            )
            * (
                -math.log(10_000.0)
                / d_model
            )
        )

        encoding = torch.zeros(
            max_sequence_length,
            d_model,
            dtype=torch.float32,
        )

        encoding[:, 0::2] = torch.sin(
            positions * frequencies
        )

        if d_model > 1:
            encoding[:, 1::2] = torch.cos(
                positions
                * frequencies[
                    :encoding[:, 1::2].shape[1]
                ]
            )

        self.register_buffer(
            "encoding",
            encoding.unsqueeze(0),
            persistent=True,
        )

    def forward(
        self,
        sequence_length: int,
        dtype: torch.dtype,
    ) -> Tensor:
        if sequence_length > self.encoding.shape[1]:
            raise ValueError(
                "Sequenza più lunga del massimo "
                "positional encoding."
            )

        return self.encoding[
            :, :sequence_length
        ].to(dtype=dtype)


class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model deve essere divisibile "
                "per num_heads."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = (
            d_model // num_heads
        )
        self.scale = (
            self.head_dim ** -0.5
        )

        self.query_key_value = nn.Linear(
            d_model,
            3 * d_model,
            bias=True,
        )

        self.output_projection = nn.Linear(
            d_model,
            d_model,
            bias=True,
        )

        self.attention_dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape
        )

        query_key_value = (
            self.query_key_value(
                hidden_states
            )
        )

        query_key_value = (
            query_key_value
            .reshape(
                batch_size,
                sequence_length,
                3,
                self.num_heads,
                self.head_dim,
            )
            .permute(2, 0, 3, 1, 4)
        )

        query, key, value = (
            query_key_value.unbind(dim=0)
        )

        attention_scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        ) * self.scale

        key_is_valid = (
            attention_mask
            .to(dtype=torch.bool)
            .unsqueeze(1)
            .unsqueeze(2)
        )

        attention_scores = (
            attention_scores.masked_fill(
                ~key_is_valid,
                torch.finfo(
                    attention_scores.dtype
                ).min,
            )
        )

        attention_probabilities = F.softmax(
            attention_scores,
            dim=-1,
            dtype=torch.float32,
        ).to(dtype=value.dtype)

        attention_probabilities = (
            self.attention_dropout(
                attention_probabilities
            )
        )

        context = torch.matmul(
            attention_probabilities,
            value,
        )

        context = (
            context
            .transpose(1, 2)
            .contiguous()
            .reshape(
                batch_size,
                sequence_length,
                self.d_model,
            )
        )

        return self.output_projection(
            context
        )


class PositionWiseFeedForward(nn.Module):

    def __init__(
        self,
        d_model: int,
        feed_forward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Linear(
            d_model,
            feed_forward_dim,
        )

        self.output_projection = nn.Linear(
            feed_forward_dim,
            d_model,
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        hidden_states = (
            self.input_projection(
                hidden_states
            )
        )

        hidden_states = F.gelu(
            hidden_states
        )

        hidden_states = self.dropout(
            hidden_states
        )

        return self.output_projection(
            hidden_states
        )


class TransformerEncoderBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        feed_forward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.attention_normalization = (
            nn.LayerNorm(d_model)
        )

        self.self_attention = (
            MultiHeadSelfAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
            )
        )

        self.attention_residual_dropout = (
            nn.Dropout(dropout)
        )

        self.feed_forward_normalization = (
            nn.LayerNorm(d_model)
        )

        self.feed_forward = (
            PositionWiseFeedForward(
                d_model=d_model,
                feed_forward_dim=(
                    feed_forward_dim
                ),
                dropout=dropout,
            )
        )

        self.feed_forward_residual_dropout = (
            nn.Dropout(dropout)
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        normalized_states = (
            self.attention_normalization(
                hidden_states
            )
        )

        attention_output = (
            self.self_attention(
                hidden_states=normalized_states,
                attention_mask=attention_mask,
            )
        )

        hidden_states = (
            hidden_states
            + self.attention_residual_dropout(
                attention_output
            )
        )

        normalized_states = (
            self.feed_forward_normalization(
                hidden_states
            )
        )

        feed_forward_output = (
            self.feed_forward(
                normalized_states
            )
        )

        hidden_states = (
            hidden_states
            + self.feed_forward_residual_dropout(
                feed_forward_output
            )
        )

        valid_positions = (
            attention_mask
            .unsqueeze(-1)
            .to(dtype=hidden_states.dtype)
        )

        return (
            hidden_states
            * valid_positions
        )


class TextTransformerEncoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        pad_token_id: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        feed_forward_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model deve essere divisibile "
                "per num_heads."
            )

        self.vocab_size = int(
            vocab_size
        )
        self.max_sequence_length = int(
            max_sequence_length
        )
        self.pad_token_id = int(
            pad_token_id
        )
        self.d_model = int(
            d_model
        )

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_token_id,
        )

        self.positional_encoding = (
            SinusoidalPositionalEncoding(
                d_model=d_model,
                max_sequence_length=(
                    max_sequence_length
                ),
            )
        )

        self.embedding_dropout = nn.Dropout(
            dropout
        )

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    feed_forward_dim=(
                        feed_forward_dim
                    ),
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_normalization = (
            nn.LayerNorm(d_model)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.token_embedding.weight,
            mean=0.0,
            std=0.02,
        )

        with torch.no_grad():
            self.token_embedding.weight[
                self.pad_token_id
            ].zero_()

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(
                module,
                nn.LayerNorm,
            ):
                nn.init.ones_(
                    module.weight
                )
                nn.init.zeros_(
                    module.bias
                )

    @classmethod
    def from_config(
        cls,
        config: dict,
    ) -> "TextTransformerEncoder":
        encoder_config = config[
            "text_encoder"
        ]

        return cls(
            vocab_size=int(
                encoder_config["vocab_size"]
            ),
            max_sequence_length=int(
                encoder_config[
                    "max_sequence_length"
                ]
            ),
            pad_token_id=int(
                encoder_config["pad_token_id"]
            ),
            d_model=int(
                encoder_config["d_model"]
            ),
            num_heads=int(
                encoder_config["num_heads"]
            ),
            num_layers=int(
                encoder_config["num_layers"]
            ),
            feed_forward_dim=int(
                encoder_config[
                    "feed_forward_dim"
                ]
            ),
            dropout=float(
                encoder_config["dropout"]
            ),
        )

    def validate_inputs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> None:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids deve avere shape "
                "[batch, sequence]."
            )

        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask e input_ids devono "
                "avere la stessa shape."
            )

        if input_ids.dtype != torch.long:
            raise TypeError(
                "input_ids deve essere torch.long."
            )

        if attention_mask.dtype not in {
            torch.bool,
            torch.int32,
            torch.int64,
        }:
            raise TypeError(
                "attention_mask deve essere "
                "booleana o intera."
            )

        if input_ids.shape[1] > (
            self.max_sequence_length
        ):
            raise ValueError(
                "La sequenza supera la lunghezza "
                "massima configurata."
            )

        if input_ids.numel() > 0:
            minimum_id = int(
                input_ids.min().item()
            )

            maximum_id = int(
                input_ids.max().item()
            )

            if minimum_id < 0:
                raise IndexError(
                    "Token ID negativo."
                )

            if maximum_id >= self.vocab_size:
                raise IndexError(
                    f"Token ID {maximum_id} fuori "
                    f"dal vocabolario di dimensione "
                    f"{self.vocab_size}."
                )

        if not torch.all(
            (attention_mask == 0)
            | (attention_mask == 1)
        ):
            raise ValueError(
                "attention_mask deve contenere "
                "solo 0 e 1."
            )

        if torch.any(
            attention_mask.sum(dim=1) == 0
        ):
            raise ValueError(
                "Ogni sequenza deve contenere "
                "almeno un token valido."
            )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        self.validate_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        sequence_length = (
            input_ids.shape[1]
        )

        hidden_states = (
            self.token_embedding(
                input_ids
            )
            * math.sqrt(self.d_model)
        )

        hidden_states = (
            hidden_states
            + self.positional_encoding(
                sequence_length=sequence_length,
                dtype=hidden_states.dtype,
            )
        )

        valid_positions = (
            attention_mask
            .unsqueeze(-1)
            .to(dtype=hidden_states.dtype)
        )

        hidden_states = (
            self.embedding_dropout(
                hidden_states
            )
            * valid_positions
        )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )

        hidden_states = (
            self.output_normalization(
                hidden_states
            )
        )

        return (
            hidden_states
            * valid_positions
        )

    @staticmethod
    def masked_mean_pool(
        hidden_states: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states deve avere shape "
                "[batch, sequence, d_model]."
            )

        if (
            attention_mask.shape
            != hidden_states.shape[:2]
        ):
            raise ValueError(
                "Shape attention_mask non compatibile."
            )

        weights = (
            attention_mask
            .unsqueeze(-1)
            .to(dtype=hidden_states.dtype)
        )

        denominator = weights.sum(
            dim=1
        ).clamp(min=1.0)

        return (
            hidden_states * weights
        ).sum(dim=1) / denominator
