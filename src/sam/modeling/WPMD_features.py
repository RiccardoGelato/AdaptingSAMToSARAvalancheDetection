import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np

class WaveletTransform(nn.Module):
    """
    2D Wavelet Transform implementation for extracting high-frequency components
    """
    def __init__(self, wavelet='db4', mode='symmetric'):
        super(WaveletTransform, self).__init__()
        self.wavelet = wavelet
        self.mode = mode
    
    def forward(self, x):
        """
        Apply 2D wavelet transform
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Dictionary with LL, LH, HL, HH components
        """
        batch_size, channels, height, width = x.shape
        
        # Process each channel separately
        wavelet_components = {'LL': [], 'LH': [], 'HL': [], 'HH': []}
        
        for b in range(batch_size):
            for c in range(channels):
                # Convert to numpy for pywt
                img = x[b, c].detach().cpu().numpy()
                
                # Perform 2D wavelet transform
                coeffs = pywt.dwt2(img, self.wavelet, mode=self.mode)
                cA, (cH, cV, cD) = coeffs
                
                # Convert back to tensors
                wavelet_components['LL'].append(torch.from_numpy(cA).float())
                wavelet_components['LH'].append(torch.from_numpy(cH).float())  # Horizontal high-freq
                wavelet_components['HL'].append(torch.from_numpy(cV).float())  # Vertical high-freq
                wavelet_components['HH'].append(torch.from_numpy(cD).float())  # Diagonal high-freq
        
        # Stack tensors
        device = x.device
        for key in wavelet_components:
            wavelet_components[key] = torch.stack(wavelet_components[key]).view(
                batch_size, channels, wavelet_components[key][0].shape[0], wavelet_components[key][0].shape[1]
            ).to(device)
        
        return wavelet_components

class WPMD_Block(nn.Module):
    """
    Wavelet-based Perona-Malik Diffusion Block
    Implements equation (4) from the paper
    """
    def __init__(self, channels, k=0.1):
        super(WPMD_Block, self).__init__()
        self.k = k  # Diffusion parameter
        self.wavelet_transform = WaveletTransform()
        
        # Convolution to map features to same dimension
        self.feature_conv = nn.Conv2d(channels, channels, 1)
        self.norm = nn.LayerNorm([channels])
        self.relu = nn.ReLU()
        
    def diffusion_coefficient(self, gradient_magnitude):
        """
        Compute g(|∇u|) = 1/(1 + |∇u|²/k²)
        """
        return 1.0 / (1.0 + (gradient_magnitude ** 2) / (self.k ** 2))
    
    def forward(self, x):
        """
        Apply WPMD according to equation (4):
        u_k - u_{k-1} = F_LH(g(√(u_LH² + u_HL²)) · u_LH) + F_HL(g(√(u_LH² + u_HL²)) · u_HL)
        """
        batch_size, channels, height, width = x.shape
        
        # Step 1: Apply wavelet transform to get high-frequency components
        wavelet_components = self.wavelet_transform(x)
        u_LH = wavelet_components['LH']  # Horizontal high-frequency (∂u/∂x equivalent)
        u_HL = wavelet_components['HL']  # Vertical high-frequency (∂u/∂y equivalent)
        
        # Step 2: Compute gradient magnitude using wavelet components
        # |∇u| ≈ √(u_LH² + u_HL²)
        gradient_magnitude = torch.sqrt(u_LH**2 + u_HL**2 + 1e-8)
        
        # Step 3: Compute diffusion coefficient
        g = self.diffusion_coefficient(gradient_magnitude)
        
        # Step 4: Apply diffusion to wavelet components
        # g(|∇u|) · u_LH and g(|∇u|) · u_HL
        diffused_LH = g * u_LH
        diffused_HL = g * u_HL
        
        # Step 5: Reconstruct the diffused signal
        # We need to upsample the wavelet components back to original size
        diffused_LH_upsampled = F.interpolate(diffused_LH, size=(height, width), 
                                            mode='bilinear', align_corners=False)
        diffused_HL_upsampled = F.interpolate(diffused_HL, size=(height, width), 
                                            mode='bilinear', align_corners=False)
        
        # Step 6: Combine the diffused components (this is the innovation from equation 4)
        # u_k = u_{k-1} + diffusion_update
        diffusion_update = diffused_LH_upsampled + diffused_HL_upsampled
        
        # Step 7: Apply the update
        u_k = x + diffusion_update
        
        # Step 8: Map to same dimension as encoder features
        structure_features = self.feature_conv(u_k)
        structure_features = self.norm(structure_features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        structure_features = self.relu(structure_features)
        
        return structure_features

class WPMD_Encoder(nn.Module):
    """
    Encoder with multiple WPMD blocks integrated at different layers
    """
    def __init__(self, input_channels=3, embed_dims=[64, 128, 256, 512]):
        super(WPMD_Encoder, self).__init__()
        
        # Initial convolution
        self.initial_conv = nn.Conv2d(input_channels, embed_dims[0], 3, padding=1)
        
        # WPMD blocks for different layers
        self.wpmd_blocks = nn.ModuleList([
            WPMD_Block(embed_dims[0]),
            WPMD_Block(embed_dims[1]), 
            WPMD_Block(embed_dims[2]),
            WPMD_Block(embed_dims[3])
        ])
        
        # Encoder layers (simplified ViT-like structure)
        self.encoder_layers = nn.ModuleList([
            nn.Conv2d(embed_dims[0], embed_dims[1], 3, stride=2, padding=1),
            nn.Conv2d(embed_dims[1], embed_dims[2], 3, stride=2, padding=1),
            nn.Conv2d(embed_dims[2], embed_dims[3], 3, stride=2, padding=1),
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dims[i]) for i in range(4)
        ])
        
    def forward(self, x):
        """
        Forward pass through encoder with WPMD blocks
        """
        features = []
        
        # Initial feature extraction
        x = self.initial_conv(x)
        
        # Apply WPMD and encoder at each layer
        for i in range(4):
            # Apply WPMD block to preserve structure and reduce noise
            structure_features = self.wpmd_blocks[i](x)
            
            # Add residual connection
            x = x + structure_features
            
            # Normalize
            b, c, h, w = x.shape
            x_norm = self.norms[i](x.permute(0, 2, 3, 1).contiguous().view(-1, c))
            x = x_norm.view(b, h, w, c).permute(0, 3, 1, 2)
            
            features.append(x)
            
            # Downsample for next layer (except last layer)
            if i < 3:
                x = self.encoder_layers[i](x)
        
        return features

# Example usage and testing
if __name__ == "__main__":
    # Test WPMD block
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create sample input (batch_size=2, channels=3, height=64, width=64)
    x = torch.randn(2, 3, 64, 64).to(device)
    
    # Test individual WPMD block
    wpmd_block = WPMD_Block(channels=3).to(device)
    output = wpmd_block(x)
    print(f"WPMD Block - Input shape: {x.shape}, Output shape: {output.shape}")
    
    # Test full encoder
    encoder = WPMD_Encoder().to(device)
    features = encoder(x)
    print(f"Encoder features shapes: {[f.shape for f in features]}")
    
    print("WPMD implementation completed successfully!")