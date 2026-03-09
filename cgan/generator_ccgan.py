import torch
import torch.nn as nn

class UNetGeneratorCCGAN(nn.Module):
    def __init__(self, in_c=1, out_c=1, num_classes=2, base=64):
        super().__init__()
        # 1. NHÃN BỆNH LÝ
        self.label_emb = nn.Embedding(num_classes, 256*256) 
        
        # 2. ENCODER
        self.d1 = nn.Sequential(nn.Conv2d(in_c + 1, base, 4, 2, 1), nn.LeakyReLU(0.2)) 
        self.d2 = self._block(base, base*2)
        self.d3 = self._block(base*2, base*4)
        self.d4 = self._block(base*4, base*8)
        self.d5 = self._block(base*8, base*8)
        self.d6 = self._block(base*8, base*8)
        self.d7 = self._block(base*8, base*8)
        self.d8 = nn.Sequential(nn.Conv2d(base*8, base*8, 4, 2, 1), nn.LeakyReLU(0.2))

        # 3. DECODER (Skip-connections)
        self.u1 = self._up_block(base*8, base*8, dropout=True)
        self.u2 = self._up_block(base*16, base*8, dropout=True)
        self.u3 = self._up_block(base*16, base*8, dropout=True)
        self.u4 = self._up_block(base*16, base*8)
        self.u5 = self._up_block(base*16, base*4)
        self.u6 = self._up_block(base*8, base*2)
        self.u7 = self._up_block(base*4, base)
        self.final = nn.Sequential(nn.ConvTranspose2d(base*2, out_c, 4, 2, 1), nn.Tanh())

    def _block(self, in_f, out_f):
        return nn.Sequential(nn.Conv2d(in_f, out_f, 4, 2, 1), nn.BatchNorm2d(out_f), nn.LeakyReLU(0.2))
    
    def _up_block(self, in_f, out_f, dropout=False):
        layers = [nn.ConvTranspose2d(in_f, out_f, 4, 2, 1), nn.BatchNorm2d(out_f), nn.ReLU()]
        if dropout: layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x, labels):
        # Trộn nhãn vào ảnh đầu vào
        c = self.label_emb(labels).view(-1, 1, 256, 256)
        x = torch.cat([x, c], dim=1) 
        
        d1 = self.d1(x); d2 = self.d2(d1); d3 = self.d3(d2); d4 = self.d4(d3)
        d5 = self.d5(d4); d6 = self.d6(d5); d7 = self.d7(d6); d8 = self.d8(d7)
        u1 = self.u1(d8)
        u2 = self.u2(torch.cat([u1, d7], 1)); u3 = self.u3(torch.cat([u2, d6], 1))
        u4 = self.u4(torch.cat([u3, d5], 1)); u5 = self.u5(torch.cat([u4, d4], 1))
        u6 = self.u6(torch.cat([u5, d3], 1)); u7 = self.u7(torch.cat([u6, d2], 1))
        return self.final(torch.cat([u7, d1], 1))