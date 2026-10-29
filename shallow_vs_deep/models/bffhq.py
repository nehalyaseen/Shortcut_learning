import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.vision_transformer import VisionTransformer
from transformers import CLIPVisionModel, CLIPVisionConfig


#3layer
class mlp3(nn.Module):
    def __init__(self, input_size=224*224*3, hidden1=512, hidden2=256, num_classes=2):
        super(mlp3, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
#5layer
class mlp5(nn.Module):
    def __init__(self, input_size=224*224*3, hidden1=1024, hidden2=512, hidden3=256, hidden4=128, num_classes=2):
        super(mlp5, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, hidden4)
        self.fc5 = nn.Linear(hidden4, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
#Resnet

class resnet(nn.Module):
    def __init__(self, num_classes=2):
        super(resnet, self).__init__()
        
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)
    
#VGG
class vgg(nn.Module):
    def __init__(self, num_classes=2):
        super(vgg, self).__init__()
        self.backbone = models.vgg16(weights=None)
        
        # First layer: 3-channel MNIST
        self.backbone.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        
        # Remove last maxpool (to avoid collapsing to 0x0)
        self.backbone.features = self.backbone.features[:-1]
        
        # Adjust classifier
        self.backbone.classifier[6] = nn.Linear(4096, num_classes) #final feature map after 4 maxpool is 512

    def forward(self, x):
        return self.backbone(x)
    
#Vit

class vit(nn.Module):
    def __init__(self, num_classes=2):
        super(vit, self).__init__()
        
        self.backbone = VisionTransformer(                                          #original    
            image_size=(224),                                                    #image_size=224,
            patch_size=16,         # 8x8 patches  (64/8)x(32/8)=8x4 = 32 tokens      #16  should be divisible by 8
            num_layers=12,                                                           #12    
            num_heads=12,                                                            #12
            hidden_dim=768,                                                         #768
            mlp_dim=3072,                                                            #3072
            num_classes=num_classes                                                 #1000    
        )

    def forward(self, x):
        return self.backbone(x)
    

class clip(nn.Module):
    def __init__(self, num_classes=2):
        super(clip, self).__init__()

        # Build CLIP vision tower with random weights (no pretrained)
        config = CLIPVisionConfig(
            hidden_size=768,          # embedding dimension
            intermediate_size=3072,    # MLP size
            num_hidden_layers=12,      # transformer depth
            num_attention_heads=12,    # attention heads
            image_size=224,            # CIFAR-10 images are 32x32
            patch_size=16              # 8x8 = 64 tokens
        )
        self.backbone = CLIPVisionModel(config)

        # Classification head
        self.fc = nn.Linear(config.hidden_size, num_classes)

    def forward(self, x):
        # CLIPVisionModel returns dict with last_hidden_state + pooled_output
        outputs = self.backbone(x)
        pooled = outputs.pooler_output   # (batch, hidden_size)
        logits = self.fc(pooled)
        return logits
