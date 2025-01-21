import torch
import torch.nn as nn
import torch.nn.functional as F

class RiskPredUNet(nn.Module):
    def __init__(self, input_dim=7, output_dim=1, hidden_dim=64, kernel_size=3):
        super(RiskPredUNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(input_dim, hidden_dim, kernel_size)
        self.enc2 = self.conv_block(hidden_dim, hidden_dim * 2, kernel_size)
        self.enc3 = self.conv_block(hidden_dim * 2, hidden_dim * 4, kernel_size)

        # Bottleneck
        self.bottleneck = self.conv_block(hidden_dim * 4, hidden_dim * 8, kernel_size)

        # Decoder
        self.dec3 = self.conv_block(hidden_dim * 8, hidden_dim * 4, kernel_size)
        self.dec2 = self.conv_block(hidden_dim * 4, hidden_dim * 2, kernel_size)
        self.dec1 = self.conv_block(hidden_dim * 2, hidden_dim, kernel_size)

        # Final output layer
        self.final = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)

    def conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc3, 2))

        # Decoder
        dec3 = self.dec3(F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True) + enc3)
        dec2 = self.dec2(F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True) + enc2)
        dec1 = self.dec1(F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True) + enc1)

        # Final output
        out = torch.tanh(self.final(dec1))
        return out

