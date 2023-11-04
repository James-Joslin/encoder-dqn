import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        size = x.size()
        x, indices = self.pool(x)
        return x, indices, size

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x, indices, output_size):
        x = self.unpool(x, indices, output_size=output_size)
        x = F.leaky_relu(self.conv1(x))
        x = F.sigmoid(self.conv2(x))
        return x
    
class VGG16Autoencoder(nn.Module):
    def __init__(self):
        super(VGG16Autoencoder, self).__init__()
        
        # Encoder Blocks
        self.enc_conv1 = EncoderBlock(3, 64)
        self.enc_conv2 = EncoderBlock(64, 128)
        self.enc_conv3 = EncoderBlock(128, 256)
        self.enc_conv4 = EncoderBlock(256, 512)
        self.enc_conv5 = EncoderBlock(512, 512)
        
        # Flattening Block
        self.flat1 = nn.Linear(512 * 6 * 5, 2048)
        self.flat2 = nn.Linear(2048, 512)
        self.flat3 = nn.Linear(512, 32)
        self.flat4 = nn.Linear(32, 10)
        
        # Unflattening Block
        self.unflat1 = nn.Linear(10, 32)
        self.unflat2 = nn.Linear(32, 512)
        self.unflat3 = nn.Linear(512, 2048)
        self.unflat4 = nn.Linear(2048, 512 * 6 * 5)
        
        # Decoder Blocks
        self.dec_conv5 = DecoderBlock(512, 512)
        self.dec_conv4 = DecoderBlock(512, 256)
        self.dec_conv3 = DecoderBlock(256, 128)
        self.dec_conv2 = DecoderBlock(128, 64)
        self.dec_conv1 = DecoderBlock(64, 3)
        
    def encode(self, x):
        sizes = []
        indices = []

        x, idx1, s1 = self.enc_conv1(x); sizes.append(s1[-2:]); indices.append(idx1)
        x, idx2, s2 = self.enc_conv2(x); sizes.append(s2[-2:]); indices.append(idx2)
        x, idx3, s3 = self.enc_conv3(x); sizes.append(s3[-2:]); indices.append(idx3)
        x, idx4, s4 = self.enc_conv4(x); sizes.append(s4[-2:]); indices.append(idx4)
        x, idx5, s5 = self.enc_conv5(x); sizes.append(s5[-2:]); indices.append(idx5)
        
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.flat1(x))
        x = F.leaky_relu(self.flat2(x))
        x = F.leaky_relu(self.flat3(x))
        x = F.leaky_relu(self.flat4(x))
        
        return x, indices, sizes
    
    def decode(self, x, indices, sizes):
        x = F.leaky_relu(self.unflat1(x))
        x = F.leaky_relu(self.unflat2(x))
        x = F.leaky_relu(self.unflat3(x))
        x = F.leaky_relu(self.unflat4(x))
        x = x.view(-1, 512, 6, 5)
        
        x = self.dec_conv5(x, indices.pop(), sizes.pop())
        x = self.dec_conv4(x, indices.pop(), sizes.pop())
        x = self.dec_conv3(x, indices.pop(), sizes.pop())
        x = self.dec_conv2(x, indices.pop(), sizes.pop())
        x = self.dec_conv1(x, indices.pop(), sizes.pop())
        
        return x
    
    def forward(self, x):
        x, indices, sizes = self.encode(x)
        x_hat = self.decode(x, indices, sizes)
        return x_hat