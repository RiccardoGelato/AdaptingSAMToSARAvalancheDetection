import torch.nn as nn
import torch
import torchvision.models as models


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        self.downsample = None
        # If dimensions change or downsampling is applied, adjust the residual branch.
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out
    
class BlockMaxPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.downsample = None
        # If dimensions change or downsampling is applied, adjust the residual branch.
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        out = self.maxpool(out)
        return out

class AutoEncoder(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=256):
        """
        This encoder assumes an input of shape (B, 3, 1024, 1024) and
        outputs a feature map of shape (B, 256, 64, 64).
        
        The layers list controls the number of blocks in each stage.
        Downsampling is performed via stride-2 convolutions.
        """
        super().__init__()
        self.in_channels = 64

        # Initial convolution and max-pooling
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # (B,64,512,512)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)               # (B,64,256,256)

        # Layer1: no further downsampling (256x256) 
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)                # (B,64,256,256)
        # Layer2: downsample to 128x128, increasing channels to 128
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)               # (B,128,128,128)
        # Layer3: downsample to 64x64, increasing channels to 256
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)               # (B,256,64,64)
        # Optional Layer4: if you want more blocks without spatial change.
        self.layer4 = self._make_layer(block, 256, layers[3], stride=1)               # (B,256,64,64)

        # Optionally, add a final conv to set the desired number of channels (num_classes = 256)
        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1)
    
    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        # The first block may downsample if stride > 1.
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        # Append remaining blocks.
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (B,3,1024,1024)
        x = self.relu(self.bn1(self.conv1(x)))  # (B,64,512,512)
        x = self.maxpool(x)                     # (B,64,256,256)
        skip1 = self.layer1(x)                  # (B,64,256,256) - optional skip connection
        skip2 = self.layer2(skip1)              # (B,128,128,128) - optional skip connection
        x = self.layer3(skip2)                  # (B,256,64,64)
        x = self.layer4(x)                      # (B,256,64,64)
        out = self.final_conv(x)                # (B,256,64,64)
        return out   # You can also return [skip1, skip2, out] if skip connections are needed downstream.
    
class Prefix(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super(Prefix, self).__init__()
        # First convolutional block (medium size)
        self.convMedium1 = BasicBlock(in_channels, 64, stride=1)
        # Second convolutional block (medium size)
        self.convMedium2 = BasicBlock(64, 128, stride=1)

        # First convolutional block (small size)
        self.convSmall1 = BasicBlock(in_channels, 64, stride=1, kernel_size=1)
        # Second convolutional block (small size)
        self.convSmall2 = BasicBlock(64, 128, stride=1, kernel_size=1)

        # First convolutional block (big size)
        self.convBig1 = BasicBlock(in_channels, 64, stride=1, kernel_size=7)
        # Second convolutional block (big size)
        self.convBig2 = BasicBlock(64, 128, stride=1, kernel_size=7)

        # Upsampling block using ConvTranspose2d to double the resolution
        self.upconv = nn.ConvTranspose2d(384, 384, kernel_size=2, stride=2)

        # Last block to reduce channels to 128
        self.lastConv = BasicBlock(384, 128, stride=1)

        # Final convolution to bring channels to out_channels (RGB)
        self.final = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Expect x with shape (B, in_channels, 512, 512)
        # Pass through the first convolutional macro block (medium size)
        xMedium = self.convMedium1(x)
        xMedium = self.convMedium2(xMedium)
        # Pass through the second convolutional macro block (small size)
        xSmall = self.convSmall1(x)
        xSmall = self.convSmall2(xSmall)
        # Pass through the third convolutional macro block (big size)
        xBig = self.convBig1(x)
        xBig = self.convBig2(xBig)
        # Concatenate the outputs of the three macro blocks along the channel dimension
        x = torch.cat((xMedium, xSmall, xBig), dim=1)
        # Upsample the concatenated output to double the resolution
        x = self.upconv(x)
        # Pass through the last convolutional macro block to reduce channels to 128
        x = self.lastConv(x)
        # Pass through the final convolution to bring channels to out_channels (RGB)
        x = self.final(x)
        return x
    
class PrefixSmall(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super(PrefixSmall, self).__init__()
        # First convolutional block (medium size)
        self.convMedium1 = BasicBlock(in_channels, 64, stride=1)

        # Upsampling block using ConvTranspose2d to double the resolution
        self.upconv = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        # Final convolution to bring channels to out_channels (RGB)
        self.final = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Expect x with shape (B, in_channels, 512, 512)
        x = self.convMedium1(x)
        # Upsample the concatenated output to double the resolution
        x = self.upconv(x)
        # Pass through the final convolution to bring channels to out_channels (RGB)
        x = self.final(x)
        return x
    

class AutoEncoderMultiDim(nn.Module):
    def __init__(self, in_channels, num_classes=256):
        super(AutoEncoderMultiDim, self).__init__()
        # First convolutional block (medium size)
        self.convMedium1 = BasicBlock(in_channels, 64, stride=2)
        # Second convolutional block (medium size)
        self.convMedium2 = BasicBlock(64, 128, stride=2)

        # First convolutional block (small size)
        self.convSmall1 = BasicBlock(in_channels, 64, stride=2, kernel_size=1)
        # Second convolutional block (small size)
        self.convSmall2 = BasicBlock(64, 128, stride=2, kernel_size=1)

        # First convolutional block (big size)
        self.convBig1 = BasicBlock(in_channels, 64, stride=2, kernel_size=7)
        # Second convolutional block (big size)
        self.convBig2 = BasicBlock(64, 128, stride=2, kernel_size=7)

        # Adjust `convFinal1` to match the checkpoint dimensions
        self.convFinal1 = BasicBlock(384, 256, stride=2)

        # Adjust `convFinal2` to match the checkpoint dimensions
        self.convFinal2 = BasicBlock(256, 256, stride=2)  # Input channels changed to 256

        # Final convolution to bring channels to `num_classes`
        self.final = nn.Conv2d(256, num_classes, kernel_size=3, padding=1, stride=1)
        

    def forward(self, x):
        # Expect x with shape (B, in_channels, 512, 512)
        # Pass through the first convolutional macro block (medium size)
        xMedium = self.convMedium1(x)
        xMedium = self.convMedium2(xMedium)
        # Pass through the second convolutional macro block (small size)
        xSmall = self.convSmall1(x)
        xSmall = self.convSmall2(xSmall)
        # Pass through the third convolutional macro block (big size)
        xBig = self.convBig1(x)
        xBig = self.convBig2(xBig)
        # Concatenate the outputs of the three macro blocks along the channel dimension
        x = torch.cat((xMedium, xSmall, xBig), dim=1)
        x = self.convFinal1(x)
        x = self.convFinal2(x)
        # Final convolution to bring channels to num_classes
        x = self.final(x)
        return x
    

class AutoEncoderSmall(nn.Module):
    def __init__(self, block=BlockMaxPool, num_classes=256, input_channels=3):
        """
        This encoder assumes an input of shape (B, 3, 1024, 1024) and
        outputs a feature map of shape (B, 256, 64, 64).
        
        The layers list controls the number of blocks in each stage.
        Downsampling is performed via stride-2 convolutions.
        """
        super().__init__()
        self.in_channels = 64

        # Initial convolution and max-pooling
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)  # (B,64,512,512)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        self.block1 = block(in_channels=64, out_channels=64, kernel_size=3)
        self.block2 = block(in_channels=64, out_channels=128, kernel_size=3)
        self.block3 = block(in_channels=128, out_channels=256, kernel_size=3)
        #self.block4 = BasicBlock(in_channels=256, out_channels=256, kernel_size=3)
        #self.block5 = BasicBlock(in_channels=256, out_channels=256, kernel_size=3)
        #self.block6 = BasicBlock(in_channels=256, out_channels=256, kernel_size=3)
        
        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1)
    
    def forward(self, x):
        # x: (B,3,1024,1024)
        x = self.relu(self.bn1(self.conv1(x)))  # (B,64,512,512)
        x = self.block1(x)                     # (B,64,256,256)
        x = self.block2(x)                     # (B,128,128,128)
        x = self.block3(x)                     # (B,256,64,64)
       # x = self.block4(x)                     # (B,256,64,64)
       # x = self.block5(x)                     # (B,256,64,64)
       # x = self.block6(x)                     # (B,256,64,64)
        out = self.final_conv(x)                # (B,256,64,64)
        return out 
    

class PretrainedAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Use a pretrained ResNet50 as the encoder.
        # Remove the fully connected and pooling layers.
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2]) 
        # The output shape will be (B, 2048, H/32, W/32) for standard ResNet50 inputs.
        
        # Design a decoder that upsamples to match the SAM image encoder output.
        # For example, if SAM's image encoder produces (B, 256, 64, 64),
        # you might want the autoencoder reconstruction in that same shape.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 256, kernel_size=2, stride=2),  # upsample factor 2
            nn.ReLU(),
        )
    
    def forward(self, x):
        features = self.encoder(x)
        reconstruction = self.decoder(features)
        return reconstruction