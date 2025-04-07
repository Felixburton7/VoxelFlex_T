"""
CNN models for VoxelFlex (Temperature-Aware).

Contains 3D CNN architectures adapted for RMSF prediction, including a
temperature feature input. Includes DenseNet3D, DilatedResNet3D, MultipathRMSFNet.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional, Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# Use centralized logger
logger = logging.getLogger("voxelflex.models") # Use parent logger name

# --- DenseNet Building Blocks ---

class _DenseLayer(nn.Module):
    """Single layer within a DenseBlock."""
    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, memory_efficient: bool = False
    ):
        super().__init__()
        self.norm1: nn.BatchNorm3d
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.relu1: nn.ReLU
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.conv1: nn.Conv3d
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))
        self.norm2: nn.BatchNorm3d
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.relu2: nn.ReLU
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.conv2: nn.Conv3d
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient
        # Checkpointing not implemented here for simplicity, but could be added via torch.utils.checkpoint

    def bn_function(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # Correctly handle list of tensors or single tensor input
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def forward(self, input_features: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        # Ensure input is treated as a list for concatenation in bn_function
        if isinstance(input_features, torch.Tensor):
            prev_features = [input_features]
        else:
            prev_features = input_features

        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    """Dense Convolutional Block."""
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False,
    ):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: torch.Tensor) -> torch.Tensor:
        features = [init_features]
        # Correctly iterate over layers in the ModuleDict
        for layer in self.values():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    """Transition layer between DenseBlocks."""
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


# --- Temperature-Aware Models ---

class DenseNet3DRegression(nn.Module):
    """
    3D DenseNet architecture adapted for temperature-aware RMSF regression.
    """
    def __init__(
        self,
        input_channels: int = 5,
        growth_rate: int = 16,
        block_config: Tuple[int, ...] = (4, 4, 4), # Use Tuple[int, ...] for flexibility
        num_init_features: int = 32,
        bn_size: int = 4,
        dropout_rate: float = 0.3,
        memory_efficient: bool = False # Add memory efficient option if needed later
    ):
        """
        Initialize DenseNet3DRegression.

        Args:
            input_channels: Number of input voxel channels.
            growth_rate: How many features to add per layer (k).
            block_config: Tuple containing number of layers in each dense block.
            num_init_features: Number of features after initial convolution.
            bn_size: Multiplicative factor for bottleneck layers.
            dropout_rate: Dropout rate for dense layers and final FC layer.
            memory_efficient: Use checkpointing to save memory (slower).
        """
        super().__init__()
        logger.info("Initializing DenseNet3DRegression model...")

        # --- Initial Convolution ---
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(input_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))
        logger.info(f"  Initial Conv: {input_channels} -> {num_init_features} features")

        # --- Dense Blocks ---
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=dropout_rate,
                memory_efficient=memory_efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            logger.info(f"  Dense Block {i+1}: {num_layers} layers, Output features: {num_features}")

            # Add transition layer if not the last block
            if i != len(block_config) - 1:
                num_output_features = num_features // 2
                trans = _Transition(num_input_features=num_features, num_output_features=num_output_features)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_output_features
                logger.info(f"  Transition {i+1}: Output features: {num_features}")

        # --- Final Batch Norm ---
        self.features.add_module('norm_final', nn.BatchNorm3d(num_features))

        # --- Global Pooling ---
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        # --- Temperature-Aware Regression Head ---
        # Input size = features from DenseNet + 1 (scaled temperature)
        self.classifier_input_features = num_features + 1
        # Define regression head layers
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1) # Final RMSF prediction
        )

        logger.info(f"  Regression Head Input Features: {self.classifier_input_features}")

        # --- Weight Initialization ---
        self._initialize_weights()
        logger.info("DenseNet3DRegression initialized.")


    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Standard initialization for linear layers in regression head
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, voxel_input: torch.Tensor, scaled_temp: torch.Tensor) -> torch.Tensor:
        """
        Forward pass incorporating voxel data and scaled temperature.

        Args:
            voxel_input: Tensor of shape (Batch, Channels, D, H, W).
            scaled_temp: Tensor of shape (Batch, 1) with scaled temperature values [0, 1].

        Returns:
            Tensor of shape (Batch,) with predicted RMSF values.
        """
        # Process voxel data through DenseNet features
        features = self.features(voxel_input)
        out = F.relu(features, inplace=True) # Final ReLU after last BatchNorm
        out = self.global_avg_pool(out)
        voxel_features = torch.flatten(out, 1) # Shape: (Batch, num_features)

        # Ensure scaled_temp has shape (Batch, 1)
        if scaled_temp.ndim == 1:
            scaled_temp = scaled_temp.unsqueeze(1)
        elif scaled_temp.shape[1] != 1:
             raise ValueError(f"scaled_temp input must have shape (Batch, 1), but got {scaled_temp.shape}")

        # Concatenate voxel features and scaled temperature
        combined_features = torch.cat((voxel_features, scaled_temp), dim=1)

        # Pass through regression head
        predictions = self.classifier(combined_features)

        return predictions.squeeze(1) # Return shape (Batch,)


# --- Residual Block (Used by DilatedResNet3D) ---
class ResidualBlock3D(nn.Module):
    """3D Residual block with optional dilation and dropout."""
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1, dropout_rate: float = 0.0):
        super().__init__()
        padding = dilation # Standard padding for dilated convolution
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity() # Use Identity if no dropout

        # Skip connection: Adjust channels if necessary
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out) # Apply dropout after first activation
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual # Add skip connection BEFORE final activation
        out = F.relu(out, inplace=True)
        return out

class DilatedResNet3D(nn.Module):
    """
    Dilated ResNet 3D architecture adapted for temperature-aware RMSF prediction.
    """
    def __init__(
        self,
        input_channels: int = 5,
        base_filters: int = 32,
        channel_growth_rate: float = 1.5,
        num_residual_blocks: int = 4,
        dropout_rate: float = 0.3
    ):
        """Initialize DilatedResNet3D."""
        super().__init__()
        logger.info("Initializing DilatedResNet3D model...")

        self.conv1 = nn.Conv3d(input_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(base_filters)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        channels = [base_filters]
        for i in range(num_residual_blocks):
            # Ensure channel count increases, minimum of +1 channel
            next_channels = max(channels[-1] + 1, int(channels[-1] * channel_growth_rate))
            channels.append(next_channels)

        self.res_blocks = nn.ModuleList()
        for i in range(num_residual_blocks):
            dilation = 2**(i % 3) # Dilations: 1, 2, 4, 1...
            block = ResidualBlock3D(channels[i], channels[i+1], dilation=dilation, dropout_rate=dropout_rate)
            self.res_blocks.append(block)
            logger.info(f"  Res Block {i+1}: {channels[i]}->{channels[i+1]} filters, Dilation={dilation}")

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        # Temperature-Aware Regression Head
        self.cnn_output_features = channels[-1]
        self.classifier_input_features = self.cnn_output_features + 1 # +1 for temp
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1) # Final RMSF prediction
        )

        logger.info(f"  Regression Head Input Features: {self.classifier_input_features}")
        self._initialize_weights()
        logger.info("DilatedResNet3D initialized.")

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, voxel_input: torch.Tensor, scaled_temp: torch.Tensor) -> torch.Tensor:
        """Forward pass incorporating voxel data and scaled temperature."""
        # Initial convolution
        x = self.conv1(voxel_input)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.pool1(x)

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Global average pooling
        x = self.global_avg_pool(x)
        voxel_features = torch.flatten(x, 1) # Shape: (Batch, cnn_output_features)

        # Prepare and concatenate temperature
        if scaled_temp.ndim == 1: scaled_temp = scaled_temp.unsqueeze(1) # Ensure (Batch, 1)
        elif scaled_temp.shape[1] != 1: raise ValueError(f"scaled_temp input must have shape (Batch, 1), but got {scaled_temp.shape}")
        combined_features = torch.cat((voxel_features, scaled_temp), dim=1)

        # Pass through regression head
        predictions = self.classifier(combined_features)

        return predictions.squeeze(1)


class MultipathRMSFNet(nn.Module):
    """
    Multi-path 3D CNN architecture adapted for temperature-aware RMSF prediction.
    Preferred model based on dissertation context.
    """
    def __init__(
        self,
        input_channels: int = 5,
        base_filters: int = 32,
        channel_growth_rate: float = 1.5,
        num_residual_blocks: int = 3, # Number of blocks *per path* after initial pooling
        dropout_rate: float = 0.3
    ):
        """Initialize MultipathRMSFNet."""
        super().__init__()
        logger.info("Initializing MultipathRMSFNet model...")

        c1 = base_filters
        # Channel size within paths (after first conv in path)
        c2 = max(c1 + 1, int(c1 * channel_growth_rate))
        # Channel size after fusion
        c3 = max(c2*3 + 1, int(c2 * 3 * channel_growth_rate / 2)) # Grow channels after fusion, but maybe less aggressively

        # Initial shared convolution
        self.conv1 = nn.Conv3d(input_channels, c1, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(c1)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Multi-path branches
        # kernel_size defines the main conv kernel for the path
        self.path1 = self._create_path(c1, c2, kernel_size=3, blocks=num_residual_blocks, dropout_rate=dropout_rate)
        self.path2 = self._create_path(c1, c2, kernel_size=5, blocks=num_residual_blocks, dropout_rate=dropout_rate)
        self.path3 = self._create_path(c1, c2, kernel_size=7, blocks=num_residual_blocks, dropout_rate=dropout_rate)
        logger.info(f"  Paths created: {num_residual_blocks} blocks each, output channels={c2}")

        # Fusion layer (adjusts channels after concatenation)
        self.fusion_conv = nn.Conv3d(c2 * 3, c3, kernel_size=1, bias=False) # 1x1 convolution for channel fusion
        self.fusion_bn = nn.BatchNorm3d(c3)
        logger.info(f"  Fusion layer: {c2*3} -> {c3} channels")

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        # Temperature-Aware Regression Head
        self.cnn_output_features = c3
        self.classifier_input_features = self.cnn_output_features + 1 # +1 for temp
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1) # Final RMSF prediction
        )

        logger.info(f"  Regression Head Input Features: {self.classifier_input_features}")
        self._initialize_weights()
        logger.info("MultipathRMSFNet initialized.")

    def _create_path(
        self, in_channels: int, path_channels: int, kernel_size: int, blocks: int, dropout_rate: float
    ) -> nn.Sequential:
        """Create a single path for the Multipath network."""
        layers = []
        padding = kernel_size // 2

        # First convolution in the path
        layers.append(nn.Conv3d(in_channels, path_channels, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.BatchNorm3d(path_channels))
        layers.append(nn.ReLU(inplace=True))
        # No pooling here, pooling happens before paths diverge

        # Additional blocks within the path (using kernel_size=3 for subsequent blocks for consistency?)
        # Or keep using the path's main kernel size? Let's keep it path-specific for now.
        current_channels = path_channels
        for _ in range(blocks - 1): # If blocks=3, adds 2 more conv layers
            layers.append(nn.Conv3d(current_channels, current_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm3d(current_channels))
            layers.append(nn.ReLU(inplace=True))
            if dropout_rate > 0:
                layers.append(nn.Dropout3d(dropout_rate))
            # Note: Could add residual connections within the path if desired

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, voxel_input: torch.Tensor, scaled_temp: torch.Tensor) -> torch.Tensor:
        """Forward pass incorporating voxel data and scaled temperature."""
        # Initial shared convolution & pooling
        x = self.conv1(voxel_input)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x_pooled = self.pool1(x) # Pool once after initial conv

        # Multi-path processing
        out1 = self.path1(x_pooled)
        out2 = self.path2(x_pooled)
        out3 = self.path3(x_pooled)

        # Concatenate path outputs along the channel dimension
        out_cat = torch.cat([out1, out2, out3], dim=1)

        # Fusion layer
        fused = self.fusion_conv(out_cat)
        fused = self.fusion_bn(fused)
        fused = F.relu(fused, inplace=True)

        # Global average pooling
        pooled = self.global_avg_pool(fused)
        voxel_features = torch.flatten(pooled, 1) # Shape: (Batch, cnn_output_features)

        # Prepare and concatenate temperature
        if scaled_temp.ndim == 1: scaled_temp = scaled_temp.unsqueeze(1) # Ensure (Batch, 1)
        elif scaled_temp.shape[1] != 1: raise ValueError(f"scaled_temp input must have shape (Batch, 1), but got {scaled_temp.shape}")
        combined_features = torch.cat((voxel_features, scaled_temp), dim=1)

        # Pass through regression head
        predictions = self.classifier(combined_features)

        return predictions.squeeze(1)


# --- Helper function to get model instance ---

def get_model(config: Dict[str, Any], input_shape: Tuple[int, ...]) -> nn.Module:
    """
    Get a model instance based on the configuration.

    Args:
        config: Model configuration dictionary section (config['model']).
        input_shape: Shape of a single voxel input (Channels, D, H, W). Used for validation.

    Returns:
        Initialized PyTorch model.
    """
    architecture = config.get('architecture')
    input_channels = config.get('input_channels', 5) # Get from config or default
    dropout_rate = config.get('dropout_rate', 0.3)

    # Verify input channels match data shape (first dimension)
    if input_shape[0] != input_channels:
         logger.warning(f"Model 'input_channels' in config ({input_channels}) does not match "
                        f"detected data channels ({input_shape[0]}). Using detected data channels value.")
         input_channels = input_shape[0] # Override config value


    logger.info(f"Creating model architecture: {architecture} with {input_channels} input channels.")

    if architecture == "densenet3d_regression":
        densenet_cfg = config.get('densenet', {})
        if not densenet_cfg: logger.warning("DenseNet config section ('model.densenet') not found or empty in config. Using defaults.")
        return DenseNet3DRegression(
            input_channels=input_channels,
            growth_rate=densenet_cfg.get('growth_rate', 16),
            block_config=tuple(densenet_cfg.get('block_config', [4, 4, 4])),
            num_init_features=densenet_cfg.get('num_init_features', 32),
            bn_size=densenet_cfg.get('bn_size', 4),
            dropout_rate=dropout_rate
        )
    elif architecture == "dilated_resnet3d":
        return DilatedResNet3D(
            input_channels=input_channels,
            base_filters=config.get('base_filters', 32),
            channel_growth_rate=config.get('channel_growth_rate', 1.5),
            num_residual_blocks=config.get('num_residual_blocks', 4),
            dropout_rate=dropout_rate
        )
    elif architecture == "multipath_rmsf_net":
        return MultipathRMSFNet(
            input_channels=input_channels,
            base_filters=config.get('base_filters', 32),
            channel_growth_rate=config.get('channel_growth_rate', 1.5),
            num_residual_blocks=config.get('num_residual_blocks', 3),
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Unknown architecture specified in config: {architecture}. Valid options: 'densenet3d_regression', 'dilated_resnet3d', 'multipath_rmsf_net'")
