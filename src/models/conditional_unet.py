
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import yaml

from torch import Tensor, nn


def load_unet_config(
    config_path: Path,
) -> dict:
    with Path(config_path).open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if "unet" not in config:
        raise ValueError(
            "Sezione 'unet' mancante."
        )

    model_config = config["unet"]

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

    missing = required_keys - set(model_config)

    if missing:
        raise ValueError(
            "Parametri U-Net mancanti: "
            f"{sorted(missing)}"
        )

    image_channels = int(
        model_config["image_channels"]
    )

    output_channels = int(
        model_config["output_channels"]
    )

    base_channels = int(
        model_config["base_channels"]
    )

    multipliers = [
        int(value)
        for value
        in model_config[
            "channel_multipliers"
        ]
    ]

    num_res_blocks = int(
        model_config["num_res_blocks"]
    )

    normalization_groups = int(
        model_config[
            "normalization_groups"
        ]
    )

    dropout = float(
        model_config["dropout"]
    )

    time_embedding_dim = int(
        model_config["time_embedding_dim"]
    )

    context_dim = int(
        model_config["context_dim"]
    )

    attention_heads = int(
        model_config["attention_heads"]
    )

    attention_levels = [
        int(value)
        for value
        in model_config[
            "attention_levels"
        ]
    ]

    if image_channels < 1:
        raise ValueError(
            "image_channels deve essere positivo."
        )

    if output_channels < 1:
        raise ValueError(
            "output_channels deve essere positivo."
        )

    if base_channels < 1:
        raise ValueError(
            "base_channels deve essere positivo."
        )

    if not multipliers:
        raise ValueError(
            "channel_multipliers non può essere vuoto."
        )

    if any(value < 1 for value in multipliers):
        raise ValueError(
            "I moltiplicatori devono essere positivi."
        )

    channels = [
        base_channels * multiplier
        for multiplier in multipliers
    ]

    if num_res_blocks < 1:
        raise ValueError(
            "num_res_blocks deve essere positivo."
        )

    if normalization_groups < 1:
        raise ValueError(
            "normalization_groups deve essere positivo."
        )

    for channel_count in channels:
        if (
            channel_count
            % normalization_groups
            != 0
        ):
            raise ValueError(
                f"{channel_count} canali non sono "
                f"divisibili per "
                f"{normalization_groups} gruppi."
            )

    if not 0.0 <= dropout < 1.0:
        raise ValueError(
            "dropout deve appartenere a [0, 1)."
        )

    if time_embedding_dim < 1:
        raise ValueError(
            "time_embedding_dim deve essere positivo."
        )

    if context_dim < 1:
        raise ValueError(
            "context_dim deve essere positivo."
        )

    if attention_heads < 1:
        raise ValueError(
            "attention_heads deve essere positivo."
        )

    invalid_levels = [
        level
        for level in attention_levels
        if level < 0 or level >= len(channels)
    ]

    if invalid_levels:
        raise ValueError(
            "Livelli attention non validi: "
            f"{invalid_levels}"
        )

    for level in attention_levels:
        if channels[level] % attention_heads != 0:
            raise ValueError(
                f"I canali del livello {level} "
                "non sono divisibili per il numero "
                "di attention head."
            )

    return config


class SinusoidalTimeEmbedding(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        max_period: float = 10_000.0,
    ) -> None:
        super().__init__()

        if embedding_dim < 1:
            raise ValueError(
                "embedding_dim deve essere positivo."
            )

        self.embedding_dim = embedding_dim
        self.max_period = float(max_period)

    def forward(
        self,
        timesteps: Tensor,
    ) -> Tensor:
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
            / max(half_dim, 1)
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
            embedding = F.pad(
                embedding,
                (0, 1),
            )

        return embedding


class ResidualBlock(nn.Module):

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        time_embedding_dim: int,
        normalization_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if (
            input_channels
            % normalization_groups
            != 0
        ):
            raise ValueError(
                "input_channels non divisibile "
                "per normalization_groups."
            )

        if (
            output_channels
            % normalization_groups
            != 0
        ):
            raise ValueError(
                "output_channels non divisibile "
                "per normalization_groups."
            )

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

        self.time_projection = nn.Linear(
            time_embedding_dim,
            output_channels,
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
            self.residual_projection = (
                nn.Identity()
            )
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
        residual = self.residual_projection(
            hidden_states
        )

        hidden_states = (
            self.input_normalization(
                hidden_states
            )
        )

        hidden_states = F.silu(hidden_states)

        hidden_states = (
            self.input_convolution(
                hidden_states
            )
        )

        projected_time = (
            self.time_projection(
                F.silu(time_embedding)
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        hidden_states = (
            hidden_states + projected_time
        )

        hidden_states = (
            self.output_normalization(
                hidden_states
            )
        )

        hidden_states = F.silu(hidden_states)
        hidden_states = self.dropout(
            hidden_states
        )

        hidden_states = (
            self.output_convolution(
                hidden_states
            )
        )

        return hidden_states + residual


class CrossAttention2D(nn.Module):

    def __init__(
        self,
        channels: int,
        context_dim: int,
        num_heads: int,
        normalization_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(
                "channels deve essere divisibile "
                "per num_heads."
            )

        if (
            channels
            % normalization_groups
            != 0
        ):
            raise ValueError(
                "channels deve essere divisibile "
                "per normalization_groups."
            )

        self.channels = channels
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.normalization = nn.GroupNorm(
            normalization_groups,
            channels,
        )

        self.query_projection = nn.Linear(
            channels,
            channels,
        )

        self.key_projection = nn.Linear(
            context_dim,
            channels,
        )

        self.value_projection = nn.Linear(
            context_dim,
            channels,
        )

        self.output_projection = nn.Linear(
            channels,
            channels,
        )

        self.output_dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        hidden_states: Tensor,
        context: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        if hidden_states.ndim != 4:
            raise ValueError(
                "hidden_states deve avere forma "
                "[batch, channels, height, width]."
            )

        if context.ndim != 3:
            raise ValueError(
                "context deve avere forma "
                "[batch, sequence, context_dim]."
            )

        if attention_mask.ndim != 2:
            raise ValueError(
                "attention_mask deve avere forma "
                "[batch, sequence]."
            )

        batch_size, channels, height, width = (
            hidden_states.shape
        )

        if context.shape[0] != batch_size:
            raise ValueError(
                "Batch di immagini e testo diversi."
            )

        if context.shape[2] != self.context_dim:
            raise ValueError(
                "Dimensione del context errata."
            )

        if attention_mask.shape != (
            batch_size,
            context.shape[1],
        ):
            raise ValueError(
                "Shape della attention mask errata."
            )

        valid_tokens = attention_mask.to(
            dtype=torch.bool
        )

        if not valid_tokens.any(dim=1).all():
            raise ValueError(
                "Ogni esempio deve contenere almeno "
                "un token valido."
            )

        residual = hidden_states

        normalized = self.normalization(
            hidden_states
        )

        image_sequence = (
            normalized
            .flatten(start_dim=2)
            .transpose(1, 2)
        )

        query = self.query_projection(
            image_sequence
        )

        key = self.key_projection(context)
        value = self.value_projection(context)

        query = (
            query.reshape(
                batch_size,
                height * width,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

        key = (
            key.reshape(
                batch_size,
                context.shape[1],
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

        value = (
            value.reshape(
                batch_size,
                context.shape[1],
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

        attention_scores = torch.matmul(
            query.float(),
            key.float().transpose(-2, -1),
        ) * self.scale

        expanded_mask = (
            valid_tokens
            .unsqueeze(1)
            .unsqueeze(2)
        )

        attention_scores = (
            attention_scores.masked_fill(
                ~expanded_mask,
                torch.finfo(
                    attention_scores.dtype
                ).min,
            )
        )

        attention_probabilities = (
            torch.softmax(
                attention_scores,
                dim=-1,
            )
            .to(dtype=value.dtype)
        )

        attended = torch.matmul(
            attention_probabilities,
            value,
        )

        attended = (
            attended
            .transpose(1, 2)
            .contiguous()
            .reshape(
                batch_size,
                height * width,
                channels,
            )
        )

        attended = self.output_projection(
            attended
        )

        attended = self.output_dropout(
            attended
        )

        attended = (
            attended
            .transpose(1, 2)
            .reshape(
                batch_size,
                channels,
                height,
                width,
            )
        )

        return residual + attended


class Downsample2D(nn.Module):

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.convolution = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        return self.convolution(
            hidden_states
        )


class Upsample2D(nn.Module):

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

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        hidden_states = F.interpolate(
            hidden_states,
            scale_factor=2.0,
            mode="nearest",
        )

        return self.convolution(
            hidden_states
        )


class DownBlock(nn.Module):

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

        residual_blocks = []

        for block_index in range(
            num_res_blocks
        ):
            residual_blocks.append(
                ResidualBlock(
                    input_channels=(
                        input_channels
                        if block_index == 0
                        else output_channels
                    ),
                    output_channels=(
                        output_channels
                    ),
                    time_embedding_dim=(
                        time_embedding_dim
                    ),
                    normalization_groups=(
                        normalization_groups
                    ),
                    dropout=dropout,
                )
            )

        self.residual_blocks = nn.ModuleList(
            residual_blocks
        )

        if use_attention:
            self.attention = CrossAttention2D(
                channels=output_channels,
                context_dim=context_dim,
                num_heads=attention_heads,
                normalization_groups=(
                    normalization_groups
                ),
                dropout=dropout,
            )
        else:
            self.attention = None

        if add_downsample:
            self.downsample = Downsample2D(
                output_channels
            )
        else:
            self.downsample = None

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
            if (
                context is None
                or attention_mask is None
            ):
                raise ValueError(
                    "Context e attention mask "
                    "necessari per la cross-attention."
                )

            hidden_states = self.attention(
                hidden_states=hidden_states,
                context=context,
                attention_mask=attention_mask,
            )

        skip_connection = hidden_states

        if self.downsample is not None:
            hidden_states = self.downsample(
                hidden_states
            )

        return hidden_states, skip_connection


class UpBlock(nn.Module):

    def __init__(
        self,
        input_channels: int,
        skip_channels: int,
        output_channels: int,
        next_channels: Optional[int],
        time_embedding_dim: int,
        num_res_blocks: int,
        normalization_groups: int,
        dropout: float,
        use_attention: bool,
        context_dim: int,
        attention_heads: int,
    ) -> None:
        super().__init__()

        residual_blocks = []

        for block_index in range(
            num_res_blocks
        ):
            residual_blocks.append(
                ResidualBlock(
                    input_channels=(
                        input_channels
                        + skip_channels
                        if block_index == 0
                        else output_channels
                    ),
                    output_channels=(
                        output_channels
                    ),
                    time_embedding_dim=(
                        time_embedding_dim
                    ),
                    normalization_groups=(
                        normalization_groups
                    ),
                    dropout=dropout,
                )
            )

        self.residual_blocks = nn.ModuleList(
            residual_blocks
        )

        if use_attention:
            self.attention = CrossAttention2D(
                channels=output_channels,
                context_dim=context_dim,
                num_heads=attention_heads,
                normalization_groups=(
                    normalization_groups
                ),
                dropout=dropout,
            )
        else:
            self.attention = None

        if next_channels is None:
            self.upsample = None
        else:
            self.upsample = Upsample2D(
                input_channels=output_channels,
                output_channels=next_channels,
            )

    def forward(
        self,
        hidden_states: Tensor,
        skip_connection: Tensor,
        time_embedding: Tensor,
        context: Optional[Tensor],
        attention_mask: Optional[Tensor],
    ) -> Tensor:
        if (
            hidden_states.shape[0]
            != skip_connection.shape[0]
        ):
            raise ValueError(
                "Batch incompatibili nella "
                "skip connection."
            )

        if (
            hidden_states.shape[-2:]
            != skip_connection.shape[-2:]
        ):
            raise ValueError(
                "Risoluzioni incompatibili nella "
                "skip connection."
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
            if (
                context is None
                or attention_mask is None
            ):
                raise ValueError(
                    "Context e attention mask "
                    "necessari per la cross-attention."
                )

            hidden_states = self.attention(
                hidden_states=hidden_states,
                context=context,
                attention_mask=attention_mask,
            )

        if self.upsample is not None:
            hidden_states = self.upsample(
                hidden_states
            )

        return hidden_states


class ConditionalUNet(nn.Module):

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
        context_dim: int,
        attention_levels: list[int],
        attention_heads: int,
    ) -> None:
        super().__init__()

        self.image_channels = image_channels
        self.output_channels = output_channels
        self.conditional = conditional
        self.context_dim = context_dim

        channels = [
            base_channels * multiplier
            for multiplier
            in channel_multipliers
        ]

        self.time_encoding = (
            SinusoidalTimeEmbedding(
                base_channels
            )
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(
                base_channels,
                time_embedding_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                time_embedding_dim,
                time_embedding_dim,
            ),
        )

        self.input_convolution = nn.Conv2d(
            image_channels,
            channels[0],
            kernel_size=3,
            padding=1,
        )

        down_blocks = []
        current_channels = channels[0]

        for level, output_channel_count in enumerate(
            channels
        ):
            use_attention = (
                conditional
                and level in attention_levels
            )

            down_blocks.append(
                DownBlock(
                    input_channels=current_channels,
                    output_channels=(
                        output_channel_count
                    ),
                    time_embedding_dim=(
                        time_embedding_dim
                    ),
                    num_res_blocks=num_res_blocks,
                    normalization_groups=(
                        normalization_groups
                    ),
                    dropout=dropout,
                    use_attention=use_attention,
                    context_dim=context_dim,
                    attention_heads=attention_heads,
                    add_downsample=(
                        level < len(channels) - 1
                    ),
                )
            )

            current_channels = (
                output_channel_count
            )

        self.down_blocks = nn.ModuleList(
            down_blocks
        )

        self.middle_block_1 = ResidualBlock(
            input_channels=channels[-1],
            output_channels=channels[-1],
            time_embedding_dim=(
                time_embedding_dim
            ),
            normalization_groups=(
                normalization_groups
            ),
            dropout=dropout,
        )

        if conditional:
            self.middle_attention = (
                CrossAttention2D(
                    channels=channels[-1],
                    context_dim=context_dim,
                    num_heads=attention_heads,
                    normalization_groups=(
                        normalization_groups
                    ),
                    dropout=dropout,
                )
            )
        else:
            self.middle_attention = None

        self.middle_block_2 = ResidualBlock(
            input_channels=channels[-1],
            output_channels=channels[-1],
            time_embedding_dim=(
                time_embedding_dim
            ),
            normalization_groups=(
                normalization_groups
            ),
            dropout=dropout,
        )

        up_blocks = []
        current_channels = channels[-1]

        for level in reversed(
            range(len(channels))
        ):
            output_channel_count = (
                channels[level]
            )

            next_channels = (
                channels[level - 1]
                if level > 0
                else None
            )

            use_attention = (
                conditional
                and level in attention_levels
            )

            up_blocks.append(
                UpBlock(
                    input_channels=current_channels,
                    skip_channels=channels[level],
                    output_channels=(
                        output_channel_count
                    ),
                    next_channels=next_channels,
                    time_embedding_dim=(
                        time_embedding_dim
                    ),
                    num_res_blocks=num_res_blocks,
                    normalization_groups=(
                        normalization_groups
                    ),
                    dropout=dropout,
                    use_attention=use_attention,
                    context_dim=context_dim,
                    attention_heads=attention_heads,
                )
            )

            current_channels = (
                next_channels
                if next_channels is not None
                else output_channel_count
            )

        self.up_blocks = nn.ModuleList(
            up_blocks
        )

        self.output_normalization = nn.GroupNorm(
            normalization_groups,
            channels[0],
        )

        self.output_convolution = nn.Conv2d(
            channels[0],
            output_channels,
            kernel_size=3,
            padding=1,
        )

    @classmethod
    def from_config(
        cls,
        config: dict,
        conditional: Optional[bool] = None,
    ) -> "ConditionalUNet":
        model_config = config["unet"]

        if conditional is None:
            conditional = bool(
                model_config["conditional"]
            )

        return cls(
            image_channels=int(
                model_config["image_channels"]
            ),
            output_channels=int(
                model_config["output_channels"]
            ),
            base_channels=int(
                model_config["base_channels"]
            ),
            channel_multipliers=[
                int(value)
                for value
                in model_config[
                    "channel_multipliers"
                ]
            ],
            num_res_blocks=int(
                model_config["num_res_blocks"]
            ),
            normalization_groups=int(
                model_config[
                    "normalization_groups"
                ]
            ),
            dropout=float(
                model_config["dropout"]
            ),
            time_embedding_dim=int(
                model_config[
                    "time_embedding_dim"
                ]
            ),
            conditional=conditional,
            context_dim=int(
                model_config["context_dim"]
            ),
            attention_levels=[
                int(value)
                for value
                in model_config[
                    "attention_levels"
                ]
            ],
            attention_heads=int(
                model_config["attention_heads"]
            ),
        )

    def forward(
        self,
        noisy_images: Tensor,
        timesteps: Tensor,
        context: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if noisy_images.ndim != 4:
            raise ValueError(
                "noisy_images deve avere forma "
                "[batch, channels, height, width]."
            )

        if (
            noisy_images.shape[1]
            != self.image_channels
        ):
            raise ValueError(
                "Numero di canali immagine errato."
            )

        if timesteps.ndim != 1:
            raise ValueError(
                "timesteps deve avere forma [batch]."
            )

        if timesteps.dtype != torch.long:
            raise TypeError(
                "timesteps deve avere dtype torch.long."
            )

        if (
            timesteps.shape[0]
            != noisy_images.shape[0]
        ):
            raise ValueError(
                "Batch immagini e timestep diversi."
            )

        if (
            noisy_images.shape[-2] % 4 != 0
            or noisy_images.shape[-1] % 4 != 0
        ):
            raise ValueError(
                "Le dimensioni spaziali devono "
                "essere divisibili per 4."
            )

        if self.conditional:
            if context is None:
                raise ValueError(
                    "Il modello condizionato richiede "
                    "il context testuale."
                )

            if attention_mask is None:
                raise ValueError(
                    "Il modello condizionato richiede "
                    "la attention mask."
                )

            if (
                context.shape[0]
                != noisy_images.shape[0]
            ):
                raise ValueError(
                    "Batch immagini e context diversi."
                )

            if (
                context.shape[-1]
                != self.context_dim
            ):
                raise ValueError(
                    "Dimensione del context errata."
                )
        else:
            if (
                context is not None
                or attention_mask is not None
            ):
                raise ValueError(
                    "Il modello unconditional non deve "
                    "ricevere conditioning testuale."
                )

        time_embedding = self.time_encoding(
            timesteps
        )

        time_embedding = self.time_mlp(
            time_embedding
        )

        hidden_states = self.input_convolution(
            noisy_images
        )

        skip_connections = []

        for block in self.down_blocks:
            hidden_states, skip = block(
                hidden_states=hidden_states,
                time_embedding=time_embedding,
                context=context,
                attention_mask=attention_mask,
            )

            skip_connections.append(skip)

        hidden_states = self.middle_block_1(
            hidden_states,
            time_embedding,
        )

        if self.middle_attention is not None:
            hidden_states = (
                self.middle_attention(
                    hidden_states=hidden_states,
                    context=context,
                    attention_mask=attention_mask,
                )
            )

        hidden_states = self.middle_block_2(
            hidden_states,
            time_embedding,
        )

        for block in self.up_blocks:
            skip_connection = (
                skip_connections.pop()
            )

            hidden_states = block(
                hidden_states=hidden_states,
                skip_connection=skip_connection,
                time_embedding=time_embedding,
                context=context,
                attention_mask=attention_mask,
            )

        if skip_connections:
            raise RuntimeError(
                "Sono rimaste skip connection "
                "non utilizzate."
            )

        hidden_states = (
            self.output_normalization(
                hidden_states
            )
        )

        hidden_states = F.silu(
            hidden_states
        )

        return self.output_convolution(
            hidden_states
        )
