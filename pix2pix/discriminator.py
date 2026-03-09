import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, in_c=2, base=64): # Nhận vào cặp (Mask, Image)
        super().__init__()
        def block(in_ch, out_ch, stride, use_bn=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride, 1, bias=not use_bn)]
            if use_bn: layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_c, base, stride=2, use_bn=False),   # 128x128
            *block(base, base * 2, stride=2),            # 64x64
            *block(base * 2, base * 4, stride=2),        # 32x32
            *block(base * 4, base * 8, stride=1),        # 31x31
            nn.Conv2d(base * 8, 1, 4, stride=1, padding=1) # Trả về ma trận đánh giá
        )

    def forward(self, x):
        return self.net(x)