import math

# import numpy as np
# import scipy.io as io
import torch
from torch import nn
# import torch.nn.functional as F
import torch.nn.init as init

from models.STNet import SpatialTransformer
try:
    from models.blocks import LinearBlock, Conv2dBlock, ResBlocks
except:
    from blocks import LinearBlock, Conv2dBlock, ResBlocks

# import sys
# sys.path.append('..')
# from modules import modulated_deform_conv

class Generator(nn.Module):   
    def __init__(self, img_size=64, sty_dim=64, n_res=2, use_sn=False, use_stn=False):
        super(Generator, self).__init__()
        print("Init Generator")

        self.nf = 64 
        self.nf_mlp = 256

        self.decoder_norm = 'adain'

        self.adaptive_param_getter = get_num_adain_params
        self.adaptive_param_assign = assign_adain_params

        print("GENERATOR NF : ", self.nf)

        s0 = 16
        n_downs = 2
        nf_dec = 256

        self.cnt_encoder = ContentEncoder(img_size, self.nf, n_downs, n_res, 'in', 'relu', 'reflect', use_stn=use_stn)
        self.decoder = Decoder(nf_dec, sty_dim, n_downs, n_res, self.decoder_norm, self.decoder_norm, 'relu', 'reflect', use_sn=use_sn)
        self.mlp = MLP(sty_dim, self.adaptive_param_getter(self.decoder), self.nf_mlp, 3, 'none', 'relu')

        self.apply(weights_init('kaiming'))

    def forward(self, x_src, s_ref):
        c_src, skip1, skip2 = self.cnt_encoder(x_src)
        x_out = self.decode(c_src, s_ref, skip1, skip2)
        return x_out

    def decode(self, cnt, sty, skip1, skip2):
        adapt_params = self.mlp(sty)
        self.adaptive_param_assign(adapt_params, self.decoder)
        out = self.decoder(cnt, skip1, skip2)
        return out

    def _initialize_weights(self, mode='fan_in'):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()


class Decoder(nn.Module):
    def __init__(self, nf_dec, sty_dim, n_downs, n_res, res_norm, dec_norm, act, pad, use_sn=False):
        super(Decoder, self).__init__()
        print("Init Decoder")

        nf = nf_dec
        self.model = nn.ModuleList()
        self.model.append(ResBlocks(n_res, nf, res_norm, act, pad, use_sn=use_sn))

        self.model.append(nn.Upsample(scale_factor=2))
        self.model.append(Conv2dBlock(nf, nf//2, 5, 1, 2, norm=dec_norm, act=act, pad_type=pad, use_sn=use_sn))
        nf //= 2

        self.model.append(nn.Upsample(scale_factor=2))
        self.model.append(Conv2dBlock(2*nf, nf//2, 5, 1, 2, norm=dec_norm, act=act, pad_type=pad, use_sn=use_sn))
        nf //= 2

        self.model.append(Conv2dBlock(2*nf, 3, 7, 1, 3, norm='none', act='tanh', pad_type=pad, use_sn=use_sn))
        self.model = nn.Sequential(*self.model)
        # self.dcn = modulated_deform_conv.ModulatedDeformConvPack(64, 64, kernel_size=(3, 3), stride=1, padding=1, groups=1, deformable_groups=1, double=True).cuda()
        self.conv1 = Conv2dBlock(128, 64, 3, 1, padding=1)
        # self.dcn_2 = modulated_deform_conv.ModulatedDeformConvPack(128, 128, kernel_size=(3, 3), stride=1, padding=1, groups=1, deformable_groups=1, double=True).cuda()
        self.conv2 = Conv2dBlock(256, 128, 3, 1, padding=1)

    def forward(self, x, skip1, skip2):
        output = x
        # print(" x.shape", x.shape)
        for i in range(len(self.model)):
            output = self.model[i](output)

            if i == 2: 
                # print("2")
                # print("output.shape: ", output.shape, ", skip2.shape: ", skip2.shape)
                deformable_concat = torch.cat((output, skip2), dim=1)
                # print("deformable_concat.shape: ", deformable_concat.shape)
                # concat_pre, offset2 = self.dcn_2(deformable_concat, skip2)
                concat_pre = self.conv2(deformable_concat)
                # print("concat_pre.shape: ", concat_pre.shape)
                output = torch.cat((concat_pre, output), dim=1)
                # print("output.shape: ", output.shape)

            if i == 4:
                # print("4")
                # print("output.shape: ", output.shape, ", skip1.shape: ", skip1.shape)
                deformable_concat = torch.cat((output, skip1), dim=1)
                # print("deformable_concat.shape: ", deformable_concat.shape)
                # concat_pre, offset1 = self.dcn(deformable_concat, skip1)
                concat_pre = self.conv1(deformable_concat)
                # print("concat_pre.shape: ", concat_pre.shape)
                output = torch.cat((concat_pre, output), dim=1)
                # print("output.shape: ", output.shape)
        
        # print("final output.shape", output.shape)
        # offset_sum1 = torch.mean(torch.abs(offset1))
        # offset_sum2 = torch.mean(torch.abs(offset2))
        # offset_sum = (offset_sum1+offset_sum2)/2
        return output # , offset_sum


class ContentEncoder(nn.Module):
    def __init__(self, img_size, nf_cnt, n_downs, n_res, norm, act, pad, use_sn=False, use_stn=False):
        super(ContentEncoder, self).__init__()
        print("Init ContentEncoder")

        nf = nf_cnt
        # let input size be 128
        # 128
        # self.dcn1 = modulated_deform_conv.ModulatedDeformConvPack(3, 64, kernel_size=(7, 7), stride=1, padding=3, groups=1, deformable_groups=1).cuda()
        self.conv1 = Conv2dBlock(3, 64, 7, 1, padding=3)
        # 128
        # self.dcn2 = modulated_deform_conv.ModulatedDeformConvPack(64, 128, kernel_size=(4, 4), stride=2, padding=1, groups=1, deformable_groups=1).cuda()
        self.conv2 = Conv2dBlock(64, 128, 4, 2, padding=1)
        # 64
        # self.dcn3 = modulated_deform_conv.ModulatedDeformConvPack(128, 256, kernel_size=(4, 4), stride=2, padding=1, groups=1, deformable_groups=1).cuda()
        self.conv3 = Conv2dBlock(128, 256, 4, 2, padding=1)
        # 32
        self.IN1 = nn.InstanceNorm2d(64)
        self.IN2 = nn.InstanceNorm2d(128)
        self.IN3 = nn.InstanceNorm2d(256)
        self.activation = nn.ReLU(inplace=True)

        # 32
        self.model = nn.ModuleList()
        self.model.append(ResBlocks(n_res, 256, norm=norm, act=act, pad_type=pad, use_sn=use_sn))
        self.model = nn.Sequential(*self.model)
        # 32

        self.use_stn = use_stn
        if use_stn:
            self.stn1 = SpatialTransformer(3, img_size, fill_background=True, use_dropout=True)
            self.stn2 = SpatialTransformer(64, img_size, fill_background=False, use_dropout=True)

    def forward(self, x):
        # x, _ = self.dcn1(x, x)
        if self.use_stn:
            x = self.stn1(x)
        x = self.conv1(x)
        x = self.IN1(x)
        x = self.activation(x)
        skip1 = x
        # x, _ = self.dcn2(x, x)
        if self.use_stn:
            x = self.stn2(x)
        x = self.conv2(x)
        x = self.IN2(x)
        x = self.activation(x)
        skip2 = x
        # x, _ = self.dcn3(x, x)
        x = self.conv3(x)
        x = self.IN3(x)
        x = self.activation(x)
        x = self.model(x)
        return x, skip1, skip2


class MLP(nn.Module):
    def __init__(self, nf_in, nf_out, nf_mlp, num_blocks, norm, act, use_sn=False):
        super(MLP, self).__init__()
        self.model = nn.ModuleList()
        nf = nf_mlp
        self.model.append(LinearBlock(nf_in, nf, norm=norm, act=act, use_sn=use_sn))
        for _ in range(num_blocks - 2):
            self.model.append(LinearBlock(nf, nf, norm=norm, act=act, use_sn=use_sn))
        self.model.append(LinearBlock(nf, nf_out, norm='none', act='none', use_sn=use_sn))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun


def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaIN2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]


def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaIN2d":
            num_adain_params += 2*m.num_features
    return num_adain_params
