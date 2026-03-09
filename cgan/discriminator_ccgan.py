import torch
import torch.nn as nn

class DiscriminatorCCGAN(nn.Module):
    def __init__(self, in_c=2, num_classes=2):
        super().__init__()
        # Nhãn được trộn vào kênh đầu vào (Mask, image, label)
        self.label_emb = nn.Embedding(num_classes, 256*256)
        
        def block(in_f, out_f, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, 4, stride, 1),
                nn.BatchNorm2d(out_f),
                nn.LeakyReLU(0.2)
            )

        self.model = nn.Sequential(
            block(in_c + 1, 64),  # 128x128
            block(64, 128),      # 64x64
            block(128, 256),     # 32x32
            block(256, 512, 1),  # 31x31
            nn.Conv2d(512, 1, 4, 1, 1) # PatchGAN output
        )

    def forward(self, x, labels):
        c = self.label_emb(labels).view(-1, 1, 256, 256)
        x = torch.cat([x, c], dim=1)
        return self.model(x)