
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn


def load_unet_config(config_path: Path) -> dict:
    # Carica e valida la configurazione della U-Net

    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if "unet" not in config:
        raise ValueError("Sezione 'unet' mancante.")

    unet_config = config["unet"]

    required_keys = {
        "image_channels",
        "output_channels",
        "base_channels",
        "channel_multipliers",
        "num_res_blocks",
        "normalization_groups",
        "dropout",
        "time_embedding_dim",
        "conditional",
        "context_dim",
        "attention_levels",
        "attention_heads",
    }

    missing_keys = required_keys - set(unet_config)

    if missing_keys:
        raise ValueError(
            "Parametri U-Net mancanti: "
            f"{sorted(missing_keys)}"
        )

    image_channels = int(unet_config["image_channels"])
    output_channels = int(unet_config["output_channels"])
    base_channels = int(unet_config["base_channels"])

    channel_multipliers = [
        int(value)
        for value in unet_config["channel_multipliers"]
    ]

    num_res_blocks = int(unet_config["num_res_blocks"])
    normalization_groups = int(
        unet_config["normalization_groups"]
    )
    time_embedding_dim = int(
        unet_config["time_embedding_dim"]
    )
    context_dim = int(unet_config["context_dim"])
    attention_heads = int(unet_config["attention_heads"])

    attention_levels = [
        int(value)
        for value in unet_config["attention_levels"]
    ]

    if image_channels < 1:
        raise ValueError("image_channels deve essere positivo.")

    if output_channels < 1:
        raise ValueError("output_channels deve essere positivo.")

    if base_channels < 1:
        raise ValueError("base_channels deve essere positivo.")

    if not channel_multipliers:
        raise ValueError(
            "channel_multipliers non può essere vuoto."
        )

    if any(value < 1 for value in channel_multipliers):
        raise ValueError(
            "channel_multipliers deve contenere valori positivi."
        )

    if num_res_blocks < 1:
        raise ValueError(
            "num_res_blocks deve essere positivo."
        )

    if normalization_groups < 1:
        raise ValueError(
            "normalization_groups deve essere positivo."
        )

    if time_embedding_dim < 1:
        raise ValueError(
            "time_embedding_dim deve essere positivo."
        )

    if context_dim < 1:
        raise ValueError("context_dim deve essere positivo.")

    if attention_heads < 1:
        raise ValueError(
            "attention_heads deve essere positivo."
        )

    if not 0.0 <= float(unet_config["dropout"]) < 1.0:
        raise ValueError("dropout deve appartenere a [0, 1).")

    channels = [
        base_channels * multiplier
        for multiplier in channel_multipliers
    ]

    for channel_count in channels:
        if channel_count % normalization_groups != 0:
            raise ValueError(
                f"{channel_count} canali non divisibili per "
                f"{normalization_groups} gruppi."
            )

    invalid_attention_levels = [
        level
        for level in attention_levels
        if level < 0 or level >= len(channels)
    ]

    if invalid_attention_levels:
        raise ValueError(
            "Livelli attention non validi: "
            f"{invalid_attention_levels}"
        )

    for level in attention_levels:
        if channels[level] % attention_heads != 0:
            raise ValueError(
                f"I canali del livello {level} non sono "
                "divisibili per attention_heads."
            )

    return config


class SinusoidalTimeEmbedding(nn.Module):
    # Embedding sinusoidale dei timestep DDPM

    def __init__(
        self,
        embedding_dim: int,
        max_period: float = 10_000.0,
    ) -> None:
        super().__init__()

        self.embedding_dim = int(embedding_dim)
        self.max_period = float(max_period)

    def forward(self, timesteps: Tensor) -> Tensor:
        if timesteps.ndim != 1:
            raise ValueError(
                "timesteps deve avere shape [batch]."
            )

        half_dim = self.embedding_dim // 2

        if half_dim == 0:
            return timesteps.float().unsqueeze(1)

        frequencies = torch.exp(
            -math.log(self.max_period)
            * torch.arange(
                half_dim,
                device=timesteps.device,
                dtype=torch.float32,
            )
            / max(half_dim - 1, 1)
        )

        arguments = (
            timesteps.float().unsqueeze(1)
            * frequencies.unsqueeze(0)
        )

        embedding = torch.cat(
            [
                torch.sin(arguments),
                torch.cos(arguments),
            ],
            dim=1,
        )

        if embedding.shape[1] < self.embedding_dim:
            embedding = F.pad(embedding, (0, 1))

        return embedding


class ResidualBlock(nn.Module):
    # Blocco residuale con iniezione dell'embedding temporale.

    # Il timestep viene proiettato sui canali e sommato come bias
    # spazialmente condiviso dopo la prima convoluzione.
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        time_embedding_dim: int,
        normalization_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.input_normalization = nn.GroupNorm(
            normalization_groups,
            input_channels,
        )

        self.input_convolution = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            padding=1,
        )

        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                time_embedding_dim,
                output_channels,
            ),
        )

        self.output_normalization = nn.GroupNorm(
            normalization_groups,
            output_channels,
        )

        self.dropout = nn.Dropout(dropout)

        self.output_convolution = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
        )

        if input_channels == output_channels:
            self.residual_projection = nn.Identity()
        else:
            self.residual_projection = nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=1,
            )

    def forward(
        self,
        hidden_states: Tensor,
        time_embedding: Tensor,
    ) -> Tensor:
        residual = self.residual_projection(hidden_states)

        hidden_states = self.input_normalization(
            hidden_states
        )
        hidden_states = F.silu(hidden_states)
        hidden_states = self.input_convolution(
            hidden_states
        )

        time_bias = self.time_projection(
            time_embedding
        ).unsqueeze(-1).unsqueeze(-1)

        hidden_states = hidden_states + time_bias

        hidden_states = self.output_normalization(
            hidden_states
        )
        hidden_states = F.silu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.output_convolution(
            hidden_states
        )

        return hidden_states + residual


class CrossAttention2D(nn.Module):
    # Cross-attention tra token spaziali dell'immagine e token testuali.

    # Il context è l'output token-level del Transformer, non un vettore
    # globale: ogni posizione spaziale può quindi pesare diversamente gli
    # attributi testuali della caption.
    def __init__(
        self,
        channels: int,
        context_dim: int,
        attention_heads: int,
        normalization_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if channels % attention_heads != 0:
            raise ValueError(
                "channels deve essere divisibile per attention_heads."
            )

        self.channels = int(channels)
        self.context_dim = int(context_dim)
        self.attention_heads = int(attention_heads)
        self.head_dim = self.channels // self.attention_heads
        self.scale = self.head_dim ** -0.5

        self.normalization = nn.GroupNorm(
            normalization_groups,
            self.channels,
        )

        self.query_projection = nn.Linear(
            self.channels,
            self.channels,
        )

        self.key_projection = nn.Linear(
            self.context_dim,
            self.channels,
        )

        self.value_projection = nn.Linear(
            self.context_dim,
            self.channels,
        )

        self.output_projection = nn.Linear(
            self.channels,
            self.channels,
        )

        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: Tensor,
        context: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        if hidden_states.ndim != 4:
            raise ValueError(
                "hidden_states deve avere shape "
                "[batch, channels, height, width]."
            )

        if context.ndim != 3:
            raise ValueError(
                "context deve avere shape "
                "[batch, sequence, context_dim]."
            )

        if attention_mask.ndim != 2:
            raise ValueError(
                "attention_mask deve avere shape "
                "[batch, sequence]."
            )

        batch_size, channels, height, width = (
            hidden_states.shape
        )

        if context.shape[0] != batch_size:
            raise ValueError(
                "Batch incoerente tra immagini e context."
            )

        if context.shape[2] != self.context_dim:
            raise ValueError("context_dim non compatibile.")

        if attention_mask.shape != context.shape[:2]:
            raise ValueError(
                "attention_mask non compatibile con context."
            )

        valid_tokens = attention_mask.to(torch.bool)

        if not valid_tokens.any(dim=1).all():
            raise ValueError(
                "Ogni caption deve contenere almeno un token valido."
            )

        residual = hidden_states

        normalized_states = self.normalization(
            hidden_states
        )

        image_tokens = normalized_states.flatten(
            start_dim=2
        ).transpose(1, 2)

        query = self.query_projection(image_tokens)
        key = self.key_projection(context)
        value = self.value_projection(context)

        query = query.reshape(
            batch_size,
            height * width,
            self.attention_heads,
            self.head_dim,
        ).transpose(1, 2)

        key = key.reshape(
            batch_size,
            context.shape[1],
            self.attention_heads,
            self.head_dim,
        ).transpose(1, 2)

        value = value.reshape(
            batch_size,
            context.shape[1],
            self.attention_heads,
            self.head_dim,
        ).transpose(1, 2)

        attention_scores = torch.matmul(
            query.float(),
            key.float().transpose(-2, -1),
        ) * self.scale

        token_mask = valid_tokens.unsqueeze(1).unsqueeze(2)

        attention_scores = attention_scores.masked_fill(
            ~token_mask,
            torch.finfo(attention_scores.dtype).min,
        )

        attention_probabilities = torch.softmax(
            attention_scores,
            dim=-1,
        ).to(dtype=value.dtype)

        attended = torch.matmul(
            attention_probabilities,
            value,
        )

        attended = attended.transpose(1, 2).contiguous()
        attended = attended.reshape(
            batch_size,
            height * width,
            channels,
        )

        attended = self.output_projection(attended)
        attended = self.output_dropout(attended)

        attended = attended.transpose(1, 2).reshape(
            batch_size,
            channels,
            height,
            width,
        )

        return residual + attended


class Downsample2D(nn.Module):
    # Riduce altezza e larghezza di un fattore due

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.convolution = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.convolution(hidden_states)


class Upsample2D(nn.Module):
    # Aumenta altezza e larghezza di un fattore due

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()

        self.convolution = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = F.interpolate(
            hidden_states,
            scale_factor=2.0,
            mode="nearest",
        )

        return self.convolution(hidden_states)


class DownBlock(nn.Module):
    # Livello encoder della U-Net

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        time_embedding_dim: int,
        num_res_blocks: int,
        normalization_groups: int,
        dropout: float,
        use_attention: bool,
        context_dim: int,
        attention_heads: int,
        add_downsample: bool,
    ) -> None:
        super().__init__()

        blocks = []

        for block_index in range(num_res_blocks):
            blocks.append(
                ResidualBlock(
                    input_channels=(
                        input_channels
                        if block_index == 0
                        else output_channels
                    ),
                    output_channels=output_channels,
                    time_embedding_dim=time_embedding_dim,
                    normalization_groups=normalization_groups,
                    dropout=dropout,
                )
            )

        self.residual_blocks = nn.ModuleList(blocks)

        self.attention = (
            CrossAttention2D(
                channels=output_channels,
                context_dim=context_dim,
                attention_heads=attention_heads,
                normalization_groups=normalization_groups,
                dropout=dropout,
            )
            if use_attention
            else None
        )

        self.downsample = (
            Downsample2D(output_channels)
            if add_downsample
            else None
        )

    def forward(
        self,
        hidden_states: Tensor,
        time_embedding: Tensor,
        context: Optional[Tensor],
        attention_mask: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        for block in self.residual_blocks:
            hidden_states = block(
                hidden_states,
                time_embedding,
            )

        if self.attention is not None:
            if context is None or attention_mask is None:
                raise ValueError(
                    "Context e attention_mask richiesti "
                    "nei livelli condizionati."
                )

            hidden_states = self.attention(
                hidden_states,
                context,
                attention_mask,
            )

        skip_connection = hidden_states

        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)

        return hidden_states, skip_connection


class UpBlock(nn.Module):
    # Livello decoder della U-Net con concatenazione skip

    def __init__(
        self,
        input_channels: int,
        skip_channels: int,
        output_channels: int,
        time_embedding_dim: int,
        num_res_blocks: int,
        normalization_groups: int,
        dropout: float,
        use_attention: bool,
        context_dim: int,
        attention_heads: int,
        add_upsample: bool,
        upsample_channels: Optional[int],
    ) -> None:
        super().__init__()

        blocks = []

        for block_index in range(num_res_blocks):
            blocks.append(
                ResidualBlock(
                    input_channels=(
                        input_channels + skip_channels
                        if block_index == 0
                        else output_channels
                    ),
                    output_channels=output_channels,
                    time_embedding_dim=time_embedding_dim,
                    normalization_groups=normalization_groups,
                    dropout=dropout,
                )
            )

        self.residual_blocks = nn.ModuleList(blocks)

        self.attention = (
            CrossAttention2D(
                channels=output_channels,
                context_dim=context_dim,
                attention_heads=attention_heads,
                normalization_groups=normalization_groups,
                dropout=dropout,
            )
            if use_attention
            else None
        )

        self.upsample = (
            Upsample2D(
                input_channels=output_channels,
                output_channels=int(upsample_channels),
            )
            if add_upsample
            else None
        )

    def forward(
        self,
        hidden_states: Tensor,
        skip_connection: Tensor,
        time_embedding: Tensor,
        context: Optional[Tensor],
        attention_mask: Optional[Tensor],
    ) -> Tensor:
        if hidden_states.shape[-2:] != skip_connection.shape[-2:]:
            raise ValueError(
                "Risoluzioni incompatibili tra decoder e skip connection."
            )

        hidden_states = torch.cat(
            [hidden_states, skip_connection],
            dim=1,
        )

        for block in self.residual_blocks:
            hidden_states = block(
                hidden_states,
                time_embedding,
            )

        if self.attention is not None:
            if context is None or attention_mask is None:
                raise ValueError(
                    "Context e attention_mask richiesti "
                    "nei livelli condizionati."
                )

            hidden_states = self.attention(
                hidden_states,
                context,
                attention_mask,
            )

        if self.upsample is not None:
            hidden_states = self.upsample(hidden_states)

        return hidden_states


class MiddleBlock(nn.Module):
    # Blocco centrale alla minima risoluzione spaziale

    def __init__(
        self,
        channels: int,
        time_embedding_dim: int,
        normalization_groups: int,
        dropout: float,
        use_attention: bool,
        context_dim: int,
        attention_heads: int,
    ) -> None:
        super().__init__()

        self.block1 = ResidualBlock(
            input_channels=channels,
            output_channels=channels,
            time_embedding_dim=time_embedding_dim,
            normalization_groups=normalization_groups,
            dropout=dropout,
        )

        self.attention = (
            CrossAttention2D(
                channels=channels,
                context_dim=context_dim,
                attention_heads=attention_heads,
                normalization_groups=normalization_groups,
                dropout=dropout,
            )
            if use_attention
            else None
        )

        self.block2 = ResidualBlock(
            input_channels=channels,
            output_channels=channels,
            time_embedding_dim=time_embedding_dim,
            normalization_groups=normalization_groups,
            dropout=dropout,
        )

    def forward(
        self,
        hidden_states: Tensor,
        time_embedding: Tensor,
        context: Optional[Tensor],
        attention_mask: Optional[Tensor],
    ) -> Tensor:
        hidden_states = self.block1(
            hidden_states,
            time_embedding,
        )

        if self.attention is not None:
            if context is None or attention_mask is None:
                raise ValueError(
                    "Context e attention_mask richiesti "
                    "nel middle block condizionato."
                )

            hidden_states = self.attention(
                hidden_states,
                context,
                attention_mask,
            )

        return self.block2(
            hidden_states,
            time_embedding,
        )


class ConditionalUNet(nn.Module):
    # U-Net DDPM compatta, condizionata opzionalmente dal testo.

    # La struttura usa una piramide a più risoluzioni, collegamenti skip
    # encoder-decoder e cross-attention nei livelli indicati dal file YAML.
    def __init__(
        self,
        image_channels: int,
        output_channels: int,
        base_channels: int,
        channel_multipliers: list[int],
        num_res_blocks: int,
        normalization_groups: int,
        dropout: float,
        time_embedding_dim: int,
        conditional: bool,
        context_dim: Optional[int],
        attention_levels: list[int],
        attention_heads: int,
    ) -> None:
        super().__init__()

        self.image_channels = int(image_channels)
        self.output_channels = int(output_channels)
        self.base_channels = int(base_channels)

        self.channel_multipliers = [
            int(value)
            for value in channel_multipliers
        ]

        self.num_res_blocks = int(num_res_blocks)
        self.normalization_groups = int(
            normalization_groups
        )
        self.time_embedding_dim = int(
            time_embedding_dim
        )
        self.conditional = bool(conditional)

        self.context_dim = (
            None
            if context_dim is None
            else int(context_dim)
        )

        self.attention_levels = [
            int(value)
            for value in attention_levels
        ]

        self.attention_heads = int(attention_heads)

        if self.conditional and self.context_dim is None:
            raise ValueError(
                "context_dim richiesto se conditional=True."
            )

        self.channels = [
            self.base_channels * multiplier
            for multiplier in self.channel_multipliers
        ]

        self.input_convolution = nn.Conv2d(
            self.image_channels,
            self.channels[0],
            kernel_size=3,
            padding=1,
        )

        self.time_embedding = SinusoidalTimeEmbedding(
            self.time_embedding_dim
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(
                self.time_embedding_dim,
                self.time_embedding_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                self.time_embedding_dim,
                self.time_embedding_dim,
            ),
        )

        down_blocks = []

        for level, output_channels_at_level in enumerate(
            self.channels
        ):
            input_channels_at_level = (
                self.channels[level - 1]
                if level > 0
                else self.channels[0]
            )

            down_blocks.append(
                DownBlock(
                    input_channels=input_channels_at_level,
                    output_channels=output_channels_at_level,
                    time_embedding_dim=self.time_embedding_dim,
                    num_res_blocks=self.num_res_blocks,
                    normalization_groups=self.normalization_groups,
                    dropout=float(dropout),
                    use_attention=(
                        self.conditional
                        and level in self.attention_levels
                    ),
                    context_dim=(
                        int(self.context_dim)
                        if self.conditional
                        else 1
                    ),
                    attention_heads=self.attention_heads,
                    add_downsample=(
                        level < len(self.channels) - 1
                    ),
                )
            )

        self.down_blocks = nn.ModuleList(down_blocks)

        deepest_level = len(self.channels) - 1

        self.middle_block = MiddleBlock(
            channels=self.channels[-1],
            time_embedding_dim=self.time_embedding_dim,
            normalization_groups=self.normalization_groups,
            dropout=float(dropout),
            use_attention=(
                self.conditional
                and deepest_level in self.attention_levels
            ),
            context_dim=(
                int(self.context_dim)
                if self.conditional
                else 1
            ),
            attention_heads=self.attention_heads,
        )

        up_blocks = []
        current_channels = self.channels[-1]

        for level in reversed(range(len(self.channels))):
            output_channels_at_level = self.channels[level]

            add_upsample = level > 0

            up_blocks.append(
                UpBlock(
                    input_channels=current_channels,
                    skip_channels=self.channels[level],
                    output_channels=output_channels_at_level,
                    time_embedding_dim=self.time_embedding_dim,
                    num_res_blocks=self.num_res_blocks,
                    normalization_groups=self.normalization_groups,
                    dropout=float(dropout),
                    use_attention=(
                        self.conditional
                        and level in self.attention_levels
                    ),
                    context_dim=(
                        int(self.context_dim)
                        if self.conditional
                        else 1
                    ),
                    attention_heads=self.attention_heads,
                    add_upsample=add_upsample,
                    upsample_channels=(
                        self.channels[level - 1]
                        if add_upsample
                        else None
                    ),
                )
            )

            current_channels = (
                self.channels[level - 1]
                if add_upsample
                else output_channels_at_level
            )

        self.up_blocks = nn.ModuleList(up_blocks)

        self.output_normalization = nn.GroupNorm(
            self.normalization_groups,
            self.channels[0],
        )

        self.output_convolution = nn.Conv2d(
            self.channels[0],
            self.output_channels,
            kernel_size=3,
            padding=1,
        )

    @classmethod
    def from_config(
        cls,
        config: dict,
        conditional: Optional[bool] = None,
    ) -> "ConditionalUNet":
        unet_config = config["unet"]

        effective_conditional = (
            bool(unet_config["conditional"])
            if conditional is None
            else bool(conditional)
        )

        return cls(
            image_channels=int(
                unet_config["image_channels"]
            ),
            output_channels=int(
                unet_config["output_channels"]
            ),
            base_channels=int(
                unet_config["base_channels"]
            ),
            channel_multipliers=[
                int(value)
                for value
                in unet_config["channel_multipliers"]
            ],
            num_res_blocks=int(
                unet_config["num_res_blocks"]
            ),
            normalization_groups=int(
                unet_config["normalization_groups"]
            ),
            dropout=float(unet_config["dropout"]),
            time_embedding_dim=int(
                unet_config["time_embedding_dim"]
            ),
            conditional=effective_conditional,
            context_dim=(
                int(unet_config["context_dim"])
                if effective_conditional
                else None
            ),
            attention_levels=[
                int(value)
                for value
                in unet_config["attention_levels"]
            ],
            attention_heads=int(
                unet_config["attention_heads"]
            ),
        )

    def forward(
        self,
        noisyimages: Tensor,
        timesteps: Tensor,
        context: Optional[Tensor],
        attention_mask: Optional[Tensor],
    ) -> Tensor:
        if noisyimages.ndim != 4:
            raise ValueError(
                "noisyimages deve avere shape "
                "[batch, channels, height, width]."
            )

        if noisyimages.shape[1] != self.image_channels:
            raise ValueError(
                "Numero di canali immagine non compatibile."
            )

        if timesteps.ndim != 1:
            raise ValueError(
                "timesteps deve avere shape [batch]."
            )

        if noisyimages.shape[0] != timesteps.shape[0]:
            raise ValueError(
                "Batch incoerente tra immagini e timestep."
            )

        if self.conditional:
            if context is None or attention_mask is None:
                raise ValueError(
                    "Context e attention_mask richiesti "
                    "in modalità conditional."
                )

            if context.ndim != 3:
                raise ValueError(
                    "context deve avere shape "
                    "[batch, sequence, context_dim]."
                )

            if attention_mask.ndim != 2:
                raise ValueError(
                    "attention_mask deve avere shape "
                    "[batch, sequence]."
                )

            if context.shape[0] != noisyimages.shape[0]:
                raise ValueError(
                    "Batch incoerente tra immagini e testo."
                )

            if context.shape[2] != self.context_dim:
                raise ValueError(
                    "Dimensione context non compatibile."
                )

            if attention_mask.shape != context.shape[:2]:
                raise ValueError(
                    "attention_mask non compatibile con context."
                )

        else:
            context = None
            attention_mask = None

        time_embedding = self.time_embedding(timesteps)
        time_embedding = self.time_mlp(time_embedding)

        hidden_states = self.input_convolution(noisyimages)

        skip_connections = []

        for down_block in self.down_blocks:
            hidden_states, skip_connection = down_block(
                hidden_states=hidden_states,
                time_embedding=time_embedding,
                context=context,
                attention_mask=attention_mask,
            )

            skip_connections.append(skip_connection)

        hidden_states = self.middle_block(
            hidden_states=hidden_states,
            time_embedding=time_embedding,
            context=context,
            attention_mask=attention_mask,
        )

        for up_block, skip_connection in zip(
            self.up_blocks,
            reversed(skip_connections),
        ):
            hidden_states = up_block(
                hidden_states=hidden_states,
                skip_connection=skip_connection,
                time_embedding=time_embedding,
                context=context,
                attention_mask=attention_mask,
            )

        hidden_states = self.output_normalization(
            hidden_states
        )
        hidden_states = F.silu(hidden_states)

        predicted_noise = self.output_convolution(
            hidden_states
        )

        if predicted_noise.shape != noisyimages.shape:
            raise RuntimeError(
                "La U-Net non ha preservato la shape dell'immagine: "
                f"{tuple(predicted_noise.shape)} invece di "
                f"{tuple(noisyimages.shape)}."
            )

        return predicted_noise
