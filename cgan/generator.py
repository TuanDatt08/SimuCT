import torch
import torch.nn as nn

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=not use_bn)]
        if use_bn: layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c), # Đã sửa lỗi TypeError ở đây
            nn.ReLU(inplace=True)
        ]
        if dropout: layers.append(nn.Dropout(0.5))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class UNetGenerator(nn.Module):
    def __init__(self, in_c=1, out_c=1, base=64):
        super().__init__()
        # Encoder
        self.d1 = DownBlock(in_c, base, use_bn=False) 
        self.d2 = DownBlock(base, base * 2)
        self.d3 = DownBlock(base * 2, base * 4)
        self.d4 = DownBlock(base * 4, base * 8)
        self.d5 = DownBlock(base * 8, base * 8)
        self.d6 = DownBlock(base * 8, base * 8)
        self.d7 = DownBlock(base * 8, base * 8)
        self.d8 = DownBlock(base * 8, base * 8, use_bn=False)

        # Decoder
        self.u1 = UpBlock(base * 8, base * 8, dropout=True)
        self.u2 = UpBlock(base * 16, base * 8, dropout=True)
        self.u3 = UpBlock(base * 16, base * 8, dropout=True)
        self.u4 = UpBlock(base * 16, base * 8)
        self.u5 = UpBlock(base * 16, base * 4)
        self.u6 = UpBlock(base * 8, base * 2)
        self.u7 = UpBlock(base * 4, base)

        self.out = nn.Sequential(
            nn.ConvTranspose2d(base * 2, out_c, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(d1); d3 = self.d3(d2); d4 = self.d4(d3)
        d5 = self.d5(d4); d6 = self.d6(d5); d7 = self.d7(d6); d8 = self.d8(d7)
        u1 = self.u1(d8)
        u2 = self.u2(torch.cat([u1, d7], 1)); u3 = self.u3(torch.cat([u2, d6], 1))
        u4 = self.u4(torch.cat([u3, d5], 1)); u5 = self.u5(torch.cat([u4, d4], 1))
        u6 = self.u6(torch.cat([u5, d3], 1)); u7 = self.u7(torch.cat([u6, d2], 1))
        return self.out(torch.cat([u7, d1], 1))