import torch
from torch import nn
import torch.nn.functional as F
from .layers import init_weights
from .conv_lstm import ConvLSTM, SequenceWise2D
import math
    
    
class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm='batch', act='relu', kernel_size=3, padding=1):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'ins':
            self.norm = nn.InstanceNorm2d(out_channels)
        
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm='batch'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.conv1 = ConvUnit(in_channels, out_channels, norm=norm)
        self.conv2 = ConvUnit(out_channels, norm=norm)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    

class Resblock(nn.Module):
    def __init__(self, in_channels, norm = "batch"):
        super().__init__()
        self.conv1 = ConvUnit(in_channels=in_channels, out_channels=in_channels, norm=norm)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        if norm == "batch":
            self.norm2 = nn.BatchNorm2d(in_channels)
        elif norm == 'ins':
            self.norm2 = nn.InstanceNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.norm2(h)        
        h = self.relu(x+h)
        return h


class Encoder(nn.Module):
    def __init__(self, in_ch=1, ch=32, ch_mult=(1,2,4,8,16), norm='batch', **ignore_kwargs):
        super().__init__()   
        self.num_resolutions = len(ch_mult)
        self.num_blocks = 1
        
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            if i_level == 0:
                block_in = in_ch
            else:
                block_in = ch*ch_mult[i_level-1]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_blocks):
                block.append(ConvBlock(in_channels=block_in,
                                       out_channels=block_out,
                                       norm='batch'))
                block_in = block_out
                
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions-1:
                down.downsample = nn.MaxPool2d(2)
            self.down.append(down)

    def forward(self, x):
        # downsampling
        hs = [x]
        skip_h = []
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
            hs.append(h)
            skip_h.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))        
        return skip_h
    
    
class Decoder(nn.Module):
    def __init__(self, ch=32, ch_mult=(1,2,4,8,16), norm='batch', **ignore_kwargs):
        super().__init__()   
        self.num_resolutions = len(ch_mult)
        self.num_blocks = 1
        self.final_channels = 2
        
        self.up = nn.ModuleList()
        for i_level in range(self.num_resolutions-1):
            block_in = ch*ch_mult[-1-i_level]
            block_out = ch*ch_mult[-1-i_level-1]
            up = nn.Module()
            up.trans_conv = SequenceWise2D(nn.ConvTranspose2d(
                in_channels=block_in, out_channels=block_out, kernel_size=2, stride=2, padding=0))
            block = nn.ModuleList()
            for i_block in range(self.num_blocks):
                block.append(SequenceWise2D(ConvBlock(in_channels=block_in, out_channels=block_out, norm=norm)))
                block_in = block_out
            up.block = block
            
            self.up.append(up)

    def forward(self, x: list):
        h = x[-1]
        for i_level in range(self.num_resolutions-1):
            for i_block in range(self.num_blocks):
                h = self.up[i_level].trans_conv(h)
                h = torch.cat((h, x[-2-i_level]), axis=2)
                h = self.up[i_level].block[i_block](h)    
        return h
    

class Unet_base(nn.Module):
    def __init__(self, in_ch=1, ch=32, ch_mult=(1,2,4,8,16), **ignore_kwargs):
        super().__init__()
        self.in_ch = in_ch
        self.ch = ch
        self.ch_mult = tuple(ch_mult)
        self.num_blocks = 1
        
        self.encoder = Encoder(in_ch, ch, ch_mult)
        self.decoder = Decoder(ch, ch_mult)

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y, h
        
        
class CLM(nn.Module):
    def __init__(self, in_ch, ch, norm='batch', **ignore_kwargs):
        super().__init__()
        self.c_lstm = ConvLSTM(in_ch, ch, (3, 3), 1, True, True, False)
        if norm   == 'batch':
            self.bn = SequenceWise2D(nn.BatchNorm2d(ch))
        else:
            self.bn = SequenceWise2D(nn.InstanceNorm2d(ch))
    
    def forward(self, x):
        x, _ = self.c_lstm(x)
        x = self.bn(x[0])
        return x


class VELM(nn.Module):
    def __init__(self, ch=32, ch_mult=(1,2,4,8,16), ns=3, middle_ch=4, norm='bn', **ignore_kwargs):
        super().__init__()   
        self.num_resolutions = len(ch_mult)        
        self.velms = nn.ModuleList()
        self.ns = ns
        if norm == 'bn':
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = nn.InstanceNorm2d
        for i_level in range(self.num_resolutions):
            block = nn.Sequential()
            block.append(SequenceWise2D(
                nn.Conv2d(ch*ch_mult[i_level]*ns, ch*ch_mult[i_level]*middle_ch, 1,
                          groups=ch*ch_mult[i_level])))
            block.append(SequenceWise2D(norm_layer(ch*ch_mult[i_level]*middle_ch)))
            block.append(nn.ReLU())
            block.append(SequenceWise2D(
                nn.Conv2d(ch*ch_mult[i_level]*middle_ch, ch*ch_mult[i_level], 1,
                          groups=ch*ch_mult[i_level])))
            block.append(SequenceWise2D(norm_layer(ch*ch_mult[i_level])))
            block.append(nn.ReLU())
            self.velms.append(block)
    
    def forward(self, h):
        assert len(h) == self.num_resolutions
        ns = h[0].size(1)
        for i in range(self.num_resolutions):
            hi_tmp = []
            for j in range(ns-2):
                n, t, c, _h, w = h[i].shape
                hi_tmp.append(h[i][:, j:j+3].transpose(1, 2).reshape(n, 3*c, _h, w))
            hi = torch.stack(hi_tmp, dim=1)
            h[i] = self.velms[i](hi)
        
        return h
    

class TLM(nn.Module):
    def __init__(self, ch, num_anchor=1, anchor=[[50, 50]], in_sequence=True, use_clm=True, norm='batch', **ignore_kwargs):
        super().__init__()
        self.num_anchor = num_anchor
        self.anchor = anchor
        self.in_head = ConvUnit(ch, ch, norm=norm)
        self.main_head = ConvBlock(ch, ch, norm=norm)
        self.pred_head = ConvUnit(ch, num_anchor*5, act='sigmoid', kernel_size=1, padding=0, norm=norm)
        self.use_clm = use_clm
        self.up = nn.UpsamplingBilinear2d(scale_factor=16)
        if in_sequence:
            self.in_head = SequenceWise2D(self.in_head)
            self.main_head = SequenceWise2D(self.main_head)
            self.pred_head = SequenceWise2D(self.pred_head)
            self.up = SequenceWise2D(self.up)
        if use_clm:            
            self.clm = CLM(ch, ch, norm=norm)
            self.pred_head_clm = SequenceWise2D(ConvUnit(ch, num_anchor*5, act='sigmoid', kernel_size=1, padding=0, norm=norm))
    
    def forward(self, x):
        x = self.in_head(x)
        x = self.main_head(x)
        y = self.pred_head(x)
        x_fuse = self.up(x)        
        if self.use_clm:
            x_clm = self.clm(x)
            y_clm = self.pred_head_clm(x_clm)
            x_clm_fuse = self.up(x_clm)
        
            return y, y_clm, x_fuse, x_clm_fuse
        return y, None, x_fuse, None
    
    
class CAPUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=2, ch=32, ch_mult=(1,2,4,8,16), norm='batch', fuse=3, velm_ns=3, 
                 anchors=[[[70, 70]]], return_feat=False, **ignore_kwargs):
        super().__init__()
        assert fuse in [1, 2, 3, 4]
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.fuse = fuse
        self.ch = ch
        self.ch_mult = tuple(ch_mult)
        self.tml_use_clm = fuse in [1, 2, 3]
        self.velm_ns = velm_ns
        self.return_feat = return_feat
        self.anchors = torch.tensor(anchors)
        self.encoder = SequenceWise2D(Encoder(in_ch, ch, ch_mult, norm=norm))
        self.decoder = Decoder(ch, ch_mult, norm=norm)
        self.clm_seg = CLM(ch*ch_mult[0], ch*ch_mult[0], norm=norm)
        self.tlm = TLM(ch*ch_mult[-1], num_anchor=len(anchors), use_clm=self.tml_use_clm, anchor=torch.tensor(anchors), norm=norm)
        self.velm = VELM(ch, ch_mult, velm_ns, middle_ch=4, norm=norm)
        
        if fuse in [1, 2, 4]:
            self.fuse_pre = SequenceWise2D(ConvUnit(ch*ch_mult[-1]+ch*ch_mult[0], ch*ch_mult[0], norm=norm))
        if fuse in [1, 3]:
            self.fuse_clm_pre = SequenceWise2D(ConvUnit(ch*ch_mult[-1]+ch*ch_mult[0], ch*ch_mult[0], norm=norm))
            
        self.seg_out = nn.ModuleList([
            SequenceWise2D(nn.Conv2d(ch*ch_mult[0], out_ch, kernel_size=1)),
            SequenceWise2D(nn.BatchNorm2d(out_ch)) if norm=='batch' else SequenceWise2D(nn.InstanceNorm2d(out_ch)),
            nn.Sigmoid()
        ])
        self.seg_out_clm = nn.ModuleList([
            SequenceWise2D(nn.Conv2d(ch*ch_mult[0], out_ch, kernel_size=1)),
            SequenceWise2D(nn.BatchNorm2d(out_ch)) if norm=='batch' else SequenceWise2D(nn.InstanceNorm2d(out_ch)),
            nn.Sigmoid()
        ])
        self.__init_weight()
        
    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.InstanceNorm2d):
                init_weights(m, init_type='kaiming')
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.velm(x)
        x_seg = self.decoder(x)
        
        # TLM
        y_det, y_det_clm, x_det_fuse, x_det_clm_fuse = self.tlm(x[-1])
        
        if self.fuse in [1,2,4]:
            x_seg = torch.cat((x_seg, x_det_fuse), dim=2)
            x_seg = self.fuse_pre(x_seg)
        if self.fuse == 3:
            x_seg = torch.cat((x_seg, x_det_clm_fuse), dim=2)
            x_seg = self.fuse_clm_pre(x_seg)
            
        # CLM
        x_clm = self.clm_seg(x_seg)
        if self.fuse == 1:
            x_clm = torch.cat((x_clm, x_det_clm_fuse), dim=2)
            x_clm = self.fuse_clm_pre(x_clm)
        
        if self.return_feat:
            return x_seg, x_clm, y_det, y_det_clm
        
        # Out Seg
        for i in range(3):
            x_seg = self.seg_out[i](x_seg)
            x_clm = self.seg_out[i](x_clm)
        y_seg = x_seg
        y_seg_clm = x_clm
        return y_seg, y_seg_clm, y_det, y_det_clm
        
        
class UNet_CLM_VELM(nn.Module):
    def __init__(self, in_ch=1, out_ch=2, ch=32, ch_mult=(1,2,4,8,16), velm_ns=3, return_feat=False, return_velm=False, norm='batch', velm_version=1, **ignore_kwargs):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.ch = ch
        self.ch_mult = tuple(ch_mult)
        self.velm_ns = velm_ns
        self.return_feat = return_feat
        self.return_velm = return_velm
        
        self.encoder = SequenceWise2D(Encoder(in_ch, ch, ch_mult, norm=norm))
        self.decoder = Decoder(ch, ch_mult, norm=norm)
        self.clm_seg = CLM(ch*ch_mult[0], ch*ch_mult[0], norm=norm)
        self.velm = VELM(ch, ch_mult, velm_ns, middle_ch=4, norm=norm)
        if not return_feat:
            self.seg_out = nn.ModuleList([
                SequenceWise2D(nn.Conv2d(ch*ch_mult[0], out_ch, kernel_size=1)),
                SequenceWise2D(nn.BatchNorm2d(out_ch)) if norm=='batch' else SequenceWise2D(nn.InstanceNorm2d(out_ch)),
                nn.Sigmoid()
            ])
            self.seg_out_clm = nn.ModuleList([
                SequenceWise2D(nn.Conv2d(ch*ch_mult[0], out_ch, kernel_size=1)),
                SequenceWise2D(nn.BatchNorm2d(out_ch)) if norm=='batch' else SequenceWise2D(nn.InstanceNorm2d(out_ch)),
                nn.Sigmoid()
            ])
        self.__init_weight()
        
    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.InstanceNorm2d):
                init_weights(m, init_type='kaiming')
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.velm(x)
        x_seg = self.decoder(x)
            
        # CLM
        x_clm = self.clm_seg(x_seg)
        if self.return_feat:
            return x_seg, x_clm
        
        # Out Seg
        for i in range(3):
            x_seg = self.seg_out[i](x_seg)
            x_clm = self.seg_out[i](x_clm)
        y_seg = x_seg
        y_seg_clm = x_clm
        return y_seg, y_seg_clm
    
    def get_velm_fea(self, x):
        x = self.encoder(x)
        x = self.velm(x)
        return x
    
    def decode_fea2seg(self, x):
        x_seg = self.decoder(x)
        # CLM
        x_clm = self.clm_seg(x_seg)
        if self.return_feat:
            return x_seg, x_clm
        
        # Out Seg
        for i in range(3):
            x_seg = self.seg_out[i](x_seg)
            x_clm = self.seg_out[i](x_clm)
        y_seg = x_seg
        y_seg_clm = x_clm
        return y_seg, y_seg_clm
    