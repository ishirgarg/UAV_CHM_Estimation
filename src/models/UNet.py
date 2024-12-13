'''
Copyright (c) 2024 Ishir Garg

Defines architecture for basic UNet components with ResNet backbone. Refer to original UNet and ResNet papers for details.
'''

import torch
class BasicBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.relu = torch.nn.Sigmoid()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(channels)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(channels)

    def forward(self, x):
        identity_input = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity_input
        out = self.relu(out)

        return out
    
class DownsamplingBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.relu = torch.nn.Sigmoid()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(in_planes)
        self.conv1 = torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_planes)
        self.conv2 = torch.nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(out_planes)

        self.downsampler = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_planes, out_planes, 1)
        )

    def forward(self, x):
        identity = self.downsampler(x)
        x = self.maxpool(x)
        x = self.bn1(x)

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out
    
class UpsamplingBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.relu = torch.nn.Sigmoid()
        self.conv1 = torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_planes)
        self.conv2 = torch.nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_planes)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class UNet(torch.nn.Module):
    def __init__(self, init_channels):
        super().__init__()
        # Start of Encoder
        self.relu = torch.nn.Sigmoid()
        self.bilinear = torch.nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv1 = torch.nn.Conv2d(init_channels, 64, 3, padding=1)

        # Block 1
        self.layer1 = torch.nn.Sequential(
            BasicBlock(64),
            BasicBlock(64),
            BasicBlock(64)
        )
        self.layer2 = torch.nn.Sequential(
            DownsamplingBlock(64, 128),
            BasicBlock(128),
            BasicBlock(128),
            BasicBlock(128)
        )
        self.layer3 = torch.nn.Sequential(
            DownsamplingBlock(128, 256),
            BasicBlock(256),
            BasicBlock(256),
            BasicBlock(256),
            BasicBlock(256),
            BasicBlock(256),
        )
        self.layer4 = torch.nn.Sequential(
            DownsamplingBlock(256, 512),
            BasicBlock(512),
            BasicBlock(512),
        )
        # End of Encoder

        # Decoder Layers
        self.encoder_conv1 = UpsamplingBlock(768, 256)
        self.encoder_conv2 = UpsamplingBlock(384, 128)
        self.encoder_conv3 = UpsamplingBlock(192, 64)
        self.encoder_conv4 = torch.nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        # START OF ENCODER
        l1_output = self.layer1(x)
        l2_output = self.layer2(l1_output)
        l3_output = self.layer3(l2_output)
        l4_output = self.layer4(l3_output)

        encoder_output = l4_output
        # # END OF ENCODER

        # START OF DECODER
        bilinear1_output = self.bilinear(encoder_output)
        concat1 = torch.concat([bilinear1_output, l3_output], dim=1)
        decoder_conv1_output = self.encoder_conv1(concat1)

        bilinear2_output = self.bilinear(decoder_conv1_output)
        concat2 = torch.concat([bilinear2_output, l2_output], dim=1)
        decoder_conv2_output = self.encoder_conv2(concat2)

        bilinear3_output = self.bilinear(decoder_conv2_output)
        concat3 = torch.concat([bilinear3_output, l1_output], dim=1)
        decoder_conv3_output = self.encoder_conv3(concat3)

        decoder_conv4_output = self.encoder_conv4(decoder_conv3_output)
        # END OF DECODER
        
        return decoder_conv4_output
    

class MiniUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Start of Encoder
        self.relu = torch.nn.Sigmoid()
        self.bilinear = torch.nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)

        # Block 1
        self.layer1 = torch.nn.Sequential(
            BasicBlock(16),
            BasicBlock(16),
        )
        self.layer2 = torch.nn.Sequential(
            DownsamplingBlock(16, 32),
            BasicBlock(32),
            BasicBlock(32),
        )
        self.layer3 = torch.nn.Sequential(
            DownsamplingBlock(32, 64),
            BasicBlock(64),
            BasicBlock(64),
        )
        # End of Encoder

        # Decoder Layers
        self.encoder_conv1 = UpsamplingBlock(96, 32)
        self.encoder_conv2 = UpsamplingBlock(48, 1)
    def forward(self, x):
        x = self.conv1(x)
        # START OF ENCODER
        l1_output = self.layer1(x)
        l2_output = self.layer2(l1_output)
        l3_output = self.layer3(l2_output)

        encoder_output = l3_output
        # # END OF ENCODER

        # START OF DECODER
        bilinear1_output = self.bilinear(encoder_output)
        pad = (l2_output.shape[3] - bilinear1_output.shape[3], 0,
               l2_output.shape[2] - bilinear1_output.shape[2], 0)
        bilinear1_output = torch.nn.functional.pad(bilinear1_output, pad)
        concat1 = torch.concat([bilinear1_output, l2_output], dim=1)
        decoder_conv1_output = self.encoder_conv1(concat1)

        bilinear2_output = self.bilinear(decoder_conv1_output)
        pad = (l1_output.shape[3] - bilinear2_output.shape[3], 0,
               l1_output.shape[2] - bilinear2_output.shape[2], 0)
        bilinear2_output = torch.nn.functional.pad(bilinear2_output, pad)
        concat2 = torch.concat([bilinear2_output, l1_output], dim=1)
        decoder_conv2_output = self.encoder_conv2(concat2)

        # END OF DECODER        
        return decoder_conv2_output
    
    