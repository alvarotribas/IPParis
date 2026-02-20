"""
PIR - Projet d'Initiation a la recherche @ Telecom Paris
Code 04 - Deep learning model architecture

Author: Alvaro RIBAS
"""

# 0 - Imports ====================================================================================

import torch
import torch.nn as nn
import numpy as np

# 1 - Branches ===================================================================================

# 1.1 - Building Branch
# Used to process the images from the building height maps
class BuildingBranch(nn.Module):
    def __init__(self):
        super(BuildingBranch, self).__init__()
        
        # Layer 1: Conv3x3 + BN + ReLU
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, 
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Layer 2: Conv3x3 + BN + ReLU
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, 
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Layer 3: MP2x2
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 4: Conv + BN + ReLU (for attention mechanism)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, 
                               kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Layer 5: Conv3x3 (for attention mechanism)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=1, 
                               kernel_size=3, stride=1, padding=1)
        
        # Layer 6: Sigmoid (for attention mechanism)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Input: [batch, 1, 256, 256] - Single building height map per sample
        x = self.relu1(self.bn1(self.conv1(x)))  # [batch, 16, 256, 256]
        x = self.relu2(self.bn2(self.conv2(x)))  # [batch, 32, 256, 256]
        pooled_features = self.maxpool(x)        # [batch, 32, 128, 128]
        
        # Spatial attention mechanism
        attention = self.relu3(self.bn3(self.conv3(pooled_features)))  # [batch, 16, 128, 128]
        attention = self.conv4(attention)                               # [batch, 1, 128, 128]
        attention_map = self.sigmoid(attention)                         # [batch, 1, 128, 128]
        
        # Apply attention to pooled features
        weighted_features = pooled_features * attention_map  # [batch, 32, 128, 128]
        
        return weighted_features
    

# 1.2 - Antenna branch
# Used to process the images from the antenna radiation pattern maps
class AntennaBranch(nn.Module):
    def __init__(self):
        super(AntennaBranch, self).__init__()
        
        # Layer 1: Conv3x3 + BN + ReLU
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, 
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Layer 2: MP2x2
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 3: Conv3x3 + BN + ReLU
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, 
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Input: [batch, 1, 256, 256] - Single antenna pattern map per sample
        x = self.relu1(self.bn1(self.conv1(x)))  # [batch, 32, 256, 256]
        x = self.maxpool(x)                      # [batch, 32, 128, 128]
        x = self.relu2(self.bn2(self.conv2(x)))  # [batch, 64, 128, 128]
        
        return x


# 1.3 - Propagation branch
# Used to get the frequency and distance from the BS and return information about the path loss
class PropagationBranch(nn.Module):
    def __init__(self, output_size=64):
        super(PropagationBranch, self).__init__()
        
        # Speed of light (constant used in FSPL)
        self.c = 3e8  # m/s
        
        # Network structure (Layers 1 and 2)
        self.fc = nn.Sequential(
            nn.Linear(3, 128),  # 3 features: log_f, log_d, fspl
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, output_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )
        
    def forward(self, frequency, distance):
        # Input: frequency [batch, 1], distance [batch, 1]
        
        # Compute physics-based features
        log_f = torch.log10(frequency + 1e-10)  # [batch, 1]
        log_d = torch.log10(distance + 1e-10)   # [batch, 1]
        
        # Free Space Path Loss
        fspl = 20 * log_f + 20 * log_d + 20 * np.log10(4 * np.pi / self.c)  # [batch, 1]
        
        # Concatenate features
        features = torch.cat([log_f, log_d, fspl], dim=1)  # [batch, 3]
        
        return self.fc(features)  # [batch, 64]


# 1.4 - Spatial Fusion block
# Used to combine the information from both types of maps into one output
class SpatialFusionBlock(nn.Module):
    def __init__(self):
        super(SpatialFusionBlock, self).__init__()
        
        # Block 1: Conv + BN + ReLU + MaxPool
        self.conv1 = nn.Conv2d(in_channels=96, out_channels=128, 
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2: Conv + BN + ReLU + MaxPool
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, 
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3: Conv + BN + ReLU + MaxPool
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, 
                               kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 4: Conv + BN + ReLU + MaxPool
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, 
                               kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # Input: [batch, 96, 128, 128] (building: 32 + antenna: 64 channels)
        x = self.relu1(self.bn1(self.conv1(x)))  # [batch, 128, 128, 128]
        x = self.maxpool1(x)                      # [batch, 128, 64, 64]
        
        x = self.relu2(self.bn2(self.conv2(x)))  # [batch, 256, 64, 64]
        x = self.maxpool2(x)                      # [batch, 256, 32, 32]
        
        x = self.relu3(self.bn3(self.conv3(x)))  # [batch, 512, 32, 32]
        x = self.maxpool3(x)                      # [batch, 512, 16, 16]
        
        x = self.relu4(self.bn4(self.conv4(x)))  # [batch, 512, 16, 16]
        x = self.maxpool4(x)                      # [batch, 512, 8, 8]
        
        return x


# 1.5 - Frequency block
# Takes the result of the result of Spatial Fusion and extracts features before combining with the propagation branch
class FrequencyBlock(nn.Module):
    def __init__(self):
        super(FrequencyBlock, self).__init__()
        
        # Layer 1: Conv + BN + ReLU
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=256, 
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Layer 2: Conv + BN + ReLU
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, 
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Layer 3: Conv + BN + ReLU
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, 
                               kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Layer 4: Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        # Input: [batch, 512, 8, 8]
        x = self.relu1(self.bn1(self.conv1(x)))  # [batch, 256, 8, 8]
        x = self.relu2(self.bn2(self.conv2(x)))  # [batch, 128, 8, 8]
        x = self.relu3(self.bn3(self.conv3(x)))  # [batch, 64, 8, 8]
        x = self.global_avg_pool(x)              # [batch, 64, 1, 1]
        x = x.view(x.size(0), -1)                # [batch, 64]
        return x


# 1.6 - Reference Signal block
# Uses the frequency features and the propagation model to calculate the predicted RSRP
class ReferenceSignalBlock(nn.Module):
    def __init__(self, hidden_size=256):
        super(ReferenceSignalBlock, self).__init__()
        
        # Input size = propagation_output (64) + 1 value + frequency_output (64) = 129
        input_size = 129
        
        # Layer 1: FC + ReLU + Dropout
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.3)
        
        # Layer 2: FC + ReLU + Dropout
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.3)
        
        # Layer 3: FC + ReLU + Dropout
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(p=0.3)
        
    def forward(self, x):
        # Input: [batch, 129] (propagation: 64 + 1 value + frequency: 64)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        # Output: [batch, 256] - represents predicted RSRP features
        rsrp = x
        return rsrp


# 1.7 - Transmission branch
# Uses the predicted RSRP combined with the application service type to determine the Tx power
class TransmissionBranch(nn.Module):
    def __init__(self, num_categories=3, hidden_size=256, output_size=1):
        super(TransmissionBranch, self).__init__()
        
        # Input size = RSRP output (256) + one-hot encoded categories (4) = 260
        input_size = 256 + num_categories
        
        # Layer 1: FC + ReLU + Dropout
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.3)
        
        # Layer 2: FC + ReLU + Dropout
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.3)
        
        # Layer 3: FC (output layer)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Input: [batch, 260] (RSRP: 256 + one-hot: 4)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        tx_power = x
        return tx_power
    
# 2 - Complete model for the 5G network ============================================================

class PowerModel(nn.Module):
    def __init__(self):
        super(PowerModel, self).__init__()
        
        # Image processing branches
        self.building_branch = BuildingBranch()
        self.antenna_branch = AntennaBranch()
        
        # Propagation branch (takes single frequency and distance)
        self.propagation_branch = PropagationBranch(output_size=64)
        
        # Spatial fusion
        self.spatial_fusion = SpatialFusionBlock()
        
        # Frequency block
        self.frequency_block = FrequencyBlock()
        
        # Reference signal block
        # Input: propagation output (64) + 1 value + frequency output (64) = 129
        self.reference_signal_block = ReferenceSignalBlock(hidden_size=256)
        
        # Transmission branch
        # Input: RSRP output (256) + 1 value = 257
        self.transmission_branch = TransmissionBranch(
            num_categories=3,
            hidden_size=256,
            output_size=1  # Final Tx Power output
        )
        
    def forward(self, building_img, antenna_img, frequency, distance, 
                ref_value, trans_value):

        # Process building image
        building_out = self.building_branch(building_img)  # [batch, 32, 128, 128]
        
        # Process antenna image
        antenna_out = self.antenna_branch(antenna_img)  # [batch, 64, 128, 128]
        
        # Concatenate building and antenna outputs
        spatial_input = torch.cat([building_out, antenna_out], dim=1)  # [batch, 96, 128, 128]
        
        # Spatial fusion
        spatial_out = self.spatial_fusion(spatial_input)  # [batch, 512, 8, 8]
        
        # Frequency block
        freq_out = self.frequency_block(spatial_out)  # [batch, 64]
        
        # Propagation branch (now processes single values)
        prop_out = self.propagation_branch(frequency, distance)  # [batch, 64]
        
        # Concatenate propagation output (64) + ref_value (1) + frequency output (64)
        ref_input = torch.cat([prop_out, ref_value, freq_out], dim=1)  # [batch, 129]
        
        # Reference signal block - produces RSRP
        RSRP = self.reference_signal_block(ref_input)  # [batch, 256]
        
        # Concatenate RSRP (256) + trans_value (4)
        trans_input = torch.cat([RSRP, trans_value], dim=1)  # [batch, 260]
        
        # Transmission branch - final Tx Power output
        tx_power = self.transmission_branch(trans_input)  # [batch, 1]
        
        return tx_power, RSRP