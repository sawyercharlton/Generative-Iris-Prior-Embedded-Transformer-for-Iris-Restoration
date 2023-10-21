import math
import random
import torch
from basicsr.archs.stylegan2_arch import (EqualConv2d, ScaledLeakyReLU)
from basicsr.ops.fused_act import FusedLeakyReLU
from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn
from torch.nn import functional as F
from .stylegan2_arch import (ConvLayer, EqualLinear, ResBlock, Generator)
from .restormer_arch import TransformerBlock, OverlapPatchEmbed, Downsample 

class StyleGAN2GeneratorSFT(Generator):
    """StyleGAN2 Generator with SFT modulation (Spatial Feature Transform).

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Layer number of MLP style layers. Default: 8.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        resample_kernel (list[int]): A list indicating the 1D resample kernel magnitude. A cross production will be
            applied to extent 1D resample kernel to 2D resample kernel. Default: (1, 3, 3, 1).
        lr_mlp (float): Learning rate multiplier for mlp layers. Default: 0.01.
        narrow (float): The narrow ratio for channels. Default: 1.
        sft_half (bool): Whether to apply SFT on half of the input channels. Default: False.
    """

    def __init__(self,
                 out_size,
                 num_style_feat=512,
                 num_mlp=8,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 lr_mlp=0.01,
                 narrow=1,
                 sft_half=False):
        super(StyleGAN2GeneratorSFT, self).__init__(
            size = out_size,
            style_dim=num_style_feat,
            n_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            blur_kernel=blur_kernel,
            lr_mlp=lr_mlp)
        self.narrow = narrow
        self.sft_half = sft_half


    def forward(
        self,
        noises,
        conditions,
        one_id=False,
        return_latents=False,
        truncation=1,
        truncation_latent=None,
        texture=None,
    ):       
    
        styles = self.style(noises)

        if texture is None:
            texture = [None] * self.num_layers

        if truncation < 1:
            styles = truncation_latent + truncation * (styles - truncation_latent)

        if styles.ndim < 3:
            latent = styles.unsqueeze(1).repeat(1, self.n_latent, 1)

        else:
            latent = styles

        # main generation
        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], texture=texture[0], one_id=one_id)

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, texture1, texture2, to_rgb in zip(self.convs[::2], self.convs[1::2], texture[1::2],
                                                        texture[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i],  texture=texture1, one_id=one_id)

            # the conditions may have fewer levels
            if i < len(conditions):
                # SFT part to combine the conditions
                if self.sft_half:  # only apply SFT to half of the channels
                    out_same, out_sft = torch.split(out, int(out.size(1) // 2), dim=1)
                    out_sft = out_sft * conditions[i - 1] + conditions[i]
                    out = torch.cat([out_same, out_sft], dim=1)
                else:  # apply SFT to all the channels
                    out = out * conditions[i - 1] + conditions[i]

            out = conv2(out, latent[:, i + 1], texture=texture2, one_id=one_id)
            skip = to_rgb(out, latent[:, i + 2], skip)  # feature back to the rgb space
            i += 2

        image = skip

        if return_latents:
            return image, latent
        else:
            return image


class ConvUpLayer(nn.Module):
    """Convolutional upsampling layer. It uses bilinear upsampler + Conv.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: 1
        padding (int): Zero-padding added to both sides of the input. Default: 0.
        bias (bool): If ``True``, adds a learnable bias to the output. Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
        activate (bool): Whether use activateion. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 bias_init_val=0,
                 activate=True):
        super(ConvUpLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # self.scale is used to scale the convolution weights, which is related to the common initializations.
        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

        if bias and not activate:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter('bias', None)

        # activation
        if activate:
            if bias:
                self.activation = FusedLeakyReLU(out_channels)
            else:
                self.activation = ScaledLeakyReLU(0.2)
        else:
            self.activation = None

    def forward(self, x):
        # bilinear upsample
        out = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # conv
        out = F.conv2d(
            out,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        # activation
        if self.activation is not None:
            out = self.activation(out)
        return out


class ResUpBlock(nn.Module):
    """Residual block with upsampling.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
    """

    def __init__(self, in_channels, out_channels):
        super(ResUpBlock, self).__init__()

        self.conv1 = ConvLayer(in_channels, in_channels, 3, bias=True, activate=True)
        self.conv2 = ConvUpLayer(in_channels, out_channels, 3, stride=1, padding=1, bias=True, activate=True)
        self.skip = ConvUpLayer(in_channels, out_channels, 1, bias=False, activate=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.skip(x)
        out = (out + skip) / math.sqrt(2)
        return out

 
@ARCH_REGISTRY.register()
class Gformer(nn.Module):

    def __init__(
            self,
            out_size,
            num_style_feat=512,
            channel_multiplier=1,
            resample_kernel=(1, 3, 3, 1),
            decoder_load_path=None,
            fix_decoder=True,
            # for stylegan decoder
            num_mlp=8,
            lr_mlp=0.01,
            input_is_latent=False,
            different_w=False,
            narrow=1,
            sft_half=False):

        super(Gformer, self).__init__()
        self.input_is_latent = input_is_latent
        self.different_w = different_w
        self.num_style_feat = num_style_feat

        unet_narrow = narrow * 0.5  # by default, use a half of input channels
        channels = {
            '4': int(512 * unet_narrow),
            '8': int(512 * unet_narrow),
            '16': int(512 * unet_narrow),
            '32': int(512 * unet_narrow),
            '64': int(256 * channel_multiplier * unet_narrow),
            '128': int(128 * channel_multiplier * unet_narrow),
            '256': int(64 * channel_multiplier * unet_narrow),
            '512': int(32 * channel_multiplier * unet_narrow),
            '1024': int(16 * channel_multiplier * unet_narrow)
        }

        self.log_size = int(math.log(out_size, 2))  
        first_out_size = 2**(int(math.log(out_size, 2)))

        # self.conv_body_first = ConvLayer(3, channels[f'{first_out_size}'], 1, bias=True, activate=True)
        self.patch_embed = OverlapPatchEmbed(3, channels[f'{first_out_size}'])
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=channels[f'{first_out_size}'], num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') for i in range(1)])
        
        self.down1_2 = Downsample(channels[f'{first_out_size}']) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=channels['128'], num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') for i in range(1)])
        
        self.down2_3 = Downsample(channels['128']) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=channels['64'], num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') for i in range(1)])

        self.down3_4 = Downsample(channels['64']) ## From Level 3 to Level 4
        self.encoder_level4 = nn.Sequential(*[TransformerBlock(dim=channels['32'], num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') for i in range(1)])

        # downsample
        in_channels = channels[f'{first_out_size}'] # 256
        self.conv_body_down = nn.ModuleList()
        for i in range(self.log_size, 2, -1):
            out_channels = channels[f'{2**(i - 1)}'] # 128, 64, 32, 16, 8, 4
            self.conv_body_down.append(ResBlock(in_channels, out_channels, blur_kernel=[1, 3, 3, 1]))
            in_channels = out_channels

        self.final_conv = ConvLayer(in_channels, channels['4'], 3, bias=True, activate=True)

        # upsample
        in_channels = channels['4']
        self.conv_body_up = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}'] # 4, 8, 16, 32, 64, 128
            self.conv_body_up.append(ResUpBlock(in_channels, out_channels))
            in_channels = out_channels

        # to RGB
        self.toRGB = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            self.toRGB.append(EqualConv2d(channels[f'{2**i}'], 3, 1, stride=1, padding=0, bias=True))

        if different_w:
            linear_out_channel = (int(math.log(out_size, 2)) * 2 - 2) * num_style_feat # 14 * 512
        else:
            linear_out_channel = num_style_feat

        self.final_linear = EqualLinear(
            channels['4'] * 4 * 4, linear_out_channel, bias=True, bias_init=0, lr_mul=1, activation=None)

        # the decoder: stylegan2 generator with SFT modulations
        self.stylegan_decoder = StyleGAN2GeneratorSFT(
            out_size=out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            blur_kernel=[1, 3, 3, 1],
            lr_mlp=lr_mlp,
            narrow=narrow,
            sft_half=sft_half)

        # load pre-trained stylegan2 model if necessary
        if decoder_load_path:
            self.stylegan_decoder.load_state_dict(
                torch.load(decoder_load_path, map_location=lambda storage, loc: storage)["g_ema"])
        # fix decoder without updating params
        if fix_decoder:
            for _, param in self.stylegan_decoder.named_parameters():
                param.requires_grad = False

        # for SFT modulations (scale and shift)
        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}'] # 8, 16, 32, 64, 128, 256
            if sft_half:
                sft_out_channels = out_channels
            else:
                sft_out_channels = out_channels * 2
            self.condition_scale.append(
                nn.Sequential(
                    EqualConv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True, bias_init_val=0),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(out_channels, sft_out_channels, 3, stride=1, padding=1, bias=True, bias_init_val=1)))
            self.condition_shift.append(
                nn.Sequential(
                    EqualConv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True, bias_init_val=0),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(out_channels, sft_out_channels, 3, stride=1, padding=1, bias=True, bias_init_val=0)))

    def forward(self, x, return_latents=False, return_rgb=True, randomize_noise=True):

        conditions = []
        unet_skips = []
        out_rgbs = []

        inp_enc_level1 = self.patch_embed(x)
        out_enc_level1 = self.encoder_level1(inp_enc_level1) # 256
        
        inp_enc_level2 = self.down1_2(out_enc_level1) 
        out_enc_level2 = self.encoder_level2(inp_enc_level2) # 128
        unet_skips.insert(0, out_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) # 64
        unet_skips.insert(0, out_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)   
        out_enc_level4 = self.encoder_level4(inp_enc_level4) # 32
        unet_skips.insert(0, out_enc_level4)

        out_enc_level5 = self.conv_body_down[3](out_enc_level4) # 16
        unet_skips.insert(0, out_enc_level5)

        out_enc_level6 = self.conv_body_down[4](out_enc_level5) # 8
        unet_skips.insert(0, out_enc_level6)

        out_enc_level7 = self.conv_body_down[5](out_enc_level6) # 4
        unet_skips.insert(0, out_enc_level7)

        # encoder
        # feat = self.conv_body_first(x)
        # for i in range(self.log_size - 2):
        #     feat = self.conv_body_down[i](feat)
        #     unet_skips.insert(0, feat) # 4, 8, 16, 32, 64, 128
        feat = self.final_conv(out_enc_level7)

        # style code
        style_code = self.final_linear(feat.view(feat.size(0), -1))
        if self.different_w:
            style_code = style_code.view(style_code.size(0), -1, self.num_style_feat)

        # decode
        for i in range(self.log_size - 2):
            # add unet skip
            feat = feat + unet_skips[i]
            # ResUpLayer
            feat = self.conv_body_up[i](feat)
            # generate scale and shift for SFT layers
            scale = self.condition_scale[i](feat)
            conditions.append(scale.clone())
            shift = self.condition_shift[i](feat)
            conditions.append(shift.clone())
            # generate rgb images
            if return_rgb:
                out_rgbs.append(self.toRGB[i](feat))

        # decoder
        image = self.stylegan_decoder(style_code,
                                         conditions,
                                         return_latents=return_latents)

        return image, out_rgbs
