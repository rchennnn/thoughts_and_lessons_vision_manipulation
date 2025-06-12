import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils.encoders import *

def get_embedding_name(embedder):
    mapper = {
        'clip': CLIPPolicy,
        'sam': SAMPolicy,
        'r3m': R3MPolicy,
        'dino': DinoV2Policy,
        'mae': MAEPolicy,
        'mvp': MVPPolicy,
        'vip': VIPPolicy,
        'vc1': VC1Policy,
        'moco': MoCoV3Policy,
        'ibot': IBOTPolicy
    }
    return mapper[embedder]

class LinearDecoder(nn.Module):
    def __init__(self, input_size):
        super(LinearDecoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)  # Adjusted to match the output of the last conv layer
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 3 * 224 * 224)  # Final layer to output 3 x 224 x 224
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))  # Convert input to 1024 dim tensor
        x = self.dropout(x)
        x = self.relu(self.fc2(x))  # Convert to 2048 dim tensor
        x = self.dropout(x)
        x = self.relu(self.fc3(x))  # Convert to 3 x 224 x 224
        x = self.fc4(x)  # Convert to 3 x 224 x 224
        x = x.view(-1, 3, 224, 224)  # Reshape to (batch_size, 3, 224, 224)
        x = self.sigmoid(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_size):
        super(Decoder, self).__init__()
        self.input_size = input_size
        # this is going to cost 9GB when doing 50176*50176 model
        # we cast it down and back up in the spirit of LoRA
        # reduces the size of the model by 10x
        self.fc1 = nn.Linear(input_size, 2048)
        self.fc2 = nn.Linear(2048, 50176)
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # Output: (batch_size, 128, 128, 128)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # Output: (batch_size, 64, 256, 256)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # Output: (batch_size, 32, 512, 512)
        self.conv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)     # Output: (batch_size, 3, 1024, 1024)
        # self.fc1 = nn.Linear(input_size, 512)  # Adjusted to match the output of the last conv layer
        # self.fc2 = nn.Linear(512, 1024)
        # self.fc3 = nn.Linear(1024, 2048)
        # self.fc4 = nn.Linear(2048, 3 * 224 * 224)  # Final layer to output 3 x 224 x 224
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.relu(self.fc1(x))  # Convert input to 1024 dim tensor
        # x = self.dropout(x)
        # x = self.relu(self.fc2(x))  # Convert to 2048 dim tensor
        # x = self.dropout(x)
        # x = self.relu(self.fc3(x))  # Convert to 3 x 224 x 224
        # x = self.fc4(x)  # Convert to 3 x 224 x 224
        # x = x.view(-1, 3, 224, 224)  # Reshape to (batch_size, 3, 224, 224)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 256, 14, 14)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        x = self.sigmoid(x)
        return x
    
class VariationalDecoder(nn.Module):
    def __init__(self, input_size):
        super(VariationalDecoder, self).__init__()
        self.input_size = input_size
        self.layer_norm = nn.LayerNorm(input_size)
        # this is going to cost 9GB when doing 50176*50176 model
        # we cast it down and back up in the spirit of LoRA
        # reduces the size of the model by 10x
        self.fc1 = nn.Linear(input_size, 2048)
        self.mu_layer = nn.Linear(2048, 2048)
        self.logvar_layer = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, 50176)
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # Output: (batch_size, 128, 128, 128)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # Output: (batch_size, 64, 256, 256)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # Output: (batch_size, 32, 512, 512)
        self.conv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)     # Output: (batch_size, 3, 1024, 1024)
        # self.fc1 = nn.Linear(input_size, 512)  # Adjusted to match the output of the last conv layer
        # self.fc2 = nn.Linear(512, 1024)
        # self.fc3 = nn.Linear(1024, 2048)
        # self.fc4 = nn.Linear(2048, 3 * 224 * 224)  # Final layer to output 3 x 224 x 224
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        
        
    def reparameterize(self, mu, logvar, deterministic=False):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) if not deterministic else 0
        return mu + eps * std

    def forward(self, x, deterministic=False):
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        x = self.reparameterize(mu, logvar, deterministic)
        
        x = self.fc2(x)
        x = x.view(-1, 256, 14, 14)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        x = self.sigmoid(x)
        return x, mu, logvar

if __name__ == "__main__":
    print("Decoder")
    # Create a random input tensor of arbitrary size
    random_input = torch.randn(1, 256)  # Example input size (1, 256)
    decoder = Decoder(input_size=256)  # Initialize the Decoder with the input size
    output = decoder(random_input)  # Forward pass through the decoder
    print("Output shape:", output.shape)  # Print the shape of the output


class PerceptualLoss(nn.Module):
    def __init__(self, encoder_name, device='cuda'):
        super(PerceptualLoss, self).__init__()
        encoder = get_embedding_name(encoder_name)()
        self.encoder, _ = encoder._load_encoder()
        self.preprocess = encoder.preprocess
        self.encoder.to(device)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
    def forward(self, input, target):
        input = self.preprocess(input)
        input = self.encoder(input)
        input = input.view(input.size(0), -1)
        return F.mse_loss(input, target)



class VGGPerceptualLoss(nn.Module):
    DEFAULT_FEATURE_LAYERS = [0, 1, 2, 3]
    IMAGENET_RESIZE = (224, 224)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    IMAGENET_SHAPE = (1, 3, 1, 1)

    def __init__(self, resize=True, feature_layers=None, style_layers=None, device='cuda'):
        super().__init__()
        self.resize = resize
        self.feature_layers = feature_layers or self.DEFAULT_FEATURE_LAYERS
        self.style_layers = style_layers or []
        features = torchvision.models.vgg16(pretrained=True).features
        self.blocks = nn.ModuleList([
            features[:4].eval(),
            features[4:9].eval(),
            features[9:16].eval(),
            features[16:23].eval(),
        ])
        for param in self.parameters():
            param.requires_grad = False
        # self.register_buffer("mean", torch.tensor(self.IMAGENET_MEAN).view(self.IMAGENET_SHAPE))
        # self.register_buffer("std", torch.tensor(self.IMAGENET_STD).view(self.IMAGENET_SHAPE))
        self.to(device)

    def _transform(self, tensor):
        if tensor.shape != self.IMAGENET_SHAPE:
            tensor = tensor.repeat(self.IMAGENET_SHAPE)
        tensor = (tensor - self.mean) / self.std
        if self.resize:
            tensor = nn.functional.interpolate(tensor, mode='bilinear', size=self.IMAGENET_RESIZE, align_corners=False)
        return tensor

    def _calculate_gram(self, tensor):
        act = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
        return act @ act.permute(0, 2, 1)

    def forward(self, output, target):
        # output, target = self._transform(output), self._transform(target)
        loss = 0.
        for i, block in enumerate(self.blocks):
            output, target = block(output), block(target)
            if i in self.feature_layers:
                loss += nn.functional.l1_loss(output, target)
            if i in self.style_layers:
                gram_output, gram_target = self._calculate_gram(output), self._calculate_gram(target)
                loss += nn.functional.l1_loss(gram_output, gram_target)
        return loss
    