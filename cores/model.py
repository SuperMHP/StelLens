import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from einops import rearrange


import pdb
import cv2
import numpy as np
from math import exp
from torch.autograd import Variable
from torch import einsum

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    #ssim_map = (2*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)
    return ssim_map

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2, err=None):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        if err is not None:
            ssim_err = 1/F.conv2d(1/(err+1e-6), self.window**2, padding = self.window_size//2, groups = channel)
        else:
            ssim_err = None
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average), ssim_err

# ssim = SSIM()




##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))

        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



class resblock(nn.Module):
    def __init__(self, dim):

        super(resblock, self).__init__()
        # self.norm = LayerNorm(dim, LayerNorm_type='BiasFree')

        self.body = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        res = self.body((x))
        res += x
        return res


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Downsample_FLC2DRC(nn.Module):
    def __init__(self, n_feat):
        super(Downsample_FLC2DRC, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//16, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(4))

    def forward(self, x):
        return self.body(x)

class Downsample_HST_RATIO(nn.Module):
    def __init__(self, n_feat):
        super(Downsample_HST_RATIO, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//16, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(4))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Transformer Block with cross attention for 1-D condition
class TransformerBlock_CA(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_CA, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.layer_norm = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, condition):
        x = x + self.attn(self.norm1(x), self.layer_norm(condition.transpose(2, 1)).transpose(1, 2))
        x = x + self.ffn(self.norm2(x))

        return x


## Transformer Block with cross attention for 2-D condition
class TransformerBlock2D_CA(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock2D_CA, self).__init__()

        self.norm1_1 = LayerNorm(dim, LayerNorm_type)
        self.norm1_2 = LayerNorm(dim, LayerNorm_type)
        self.attn = CrossAttention2D(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, condition):
        x = x + self.attn(self.norm1_1(x), self.norm1_2(condition))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Cross Attention for 2-D condtion input
class CrossAttention2D(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention2D, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x_q, x_kv):
        '''
        input:
            x_q: for query
            x_kv: for key and value
        normally:
            feature is x_q, and prompt is x_kv
            this imple that feature will select the prompt
        cross attention as no skip connection in default
        '''
        b,c,h,w = x_q.shape
        q = self.q_dwconv(self.q(x_q))
        kv = self.kv_dwconv(self.kv(x_kv))
        k,v = kv.chunk(2, dim=1)  
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Cross Attention for 1-D condtion input
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()

        self.num_heads = num_heads
        self.temperature = dim ** -0.5
        # self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.to_k = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)

        self.to_v = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)

        # self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x, condition=None):

        b, c, h, w = x.shape
        _, _, L = condition.shape

        # q = self.q_dwconv(self.to_q(x))
        q = self.to_q(x)
        k = self.to_k(condition)
        v = self.to_v(condition)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) L -> b head c L', head=self.num_heads)
        v = rearrange(v, 'b (head c) L -> b head c L', head=self.num_heads)

        # q = torch.nn.functional.normalize(q, dim=-1)
        # k = torch.nn.functional.normalize(k, dim=-1)

        attn = torch.einsum('b h d q, b h d k -> b h q k', q, k)
        attn = attn * self.temperature
        attn = attn.softmax(dim=-1)

        out = torch.einsum('b h q k, b h d k -> b h d q', attn, v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        

    def forward(self,x):
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt

class SequentialWithArgs(nn.Sequential):
    def forward(self, x, *args):
        for module in self:
            x = module(x, *args)
        return x



def depth_wise_conv(x, depthwise_kernels):
    """
    x: [B, C, H, W]
    kernels: [B, C, kernel_size, kernel_size]
    """
    B, C, H, W = x.shape
    _, _, kernel_size, _ = depthwise_kernels.shape

    # depthwise_kernels = kernels.repeat(1, C, 1, 1)  # (B, C, kernel_size, kernel_size)

    outputs = []
    for i in range(B):

        out  = F.conv2d(x[i:i+1], depthwise_kernels[i].unsqueeze(1), groups=C, padding=kernel_size//2)
        outputs.append(out)

    outputs = torch.cat(outputs, dim=0)
    return outputs


##########################################################################
##---------- PromptIR with Cross Attention -----------------------
class PromptIR_CA(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        prompt = True,
        dual_pixel_task = True,
        **kargs,
    ):

        super(PromptIR_CA, self).__init__(**kargs)

        self.patch_embed = OverlapPatchEmbed(int(inp_channels // 2), dim)

        self.err_patch_embed = OverlapPatchEmbed(int(inp_channels // 2), dim)

        self.reduce_channel = nn.Conv2d(dim *2 ** 1, dim, kernel_size=1, bias=bias)

        # starburst condition embed
        self.sdss2drc_condition_embed1 = OverlapPatchEmbed(1, int(dim))
        self.sdss2drc_condition_embed2 = OverlapPatchEmbed(1, int(dim*2**1))
        self.sdss2drc_condition_embed3 = OverlapPatchEmbed(1, int(dim*2**2))
        self.sdss2drc_condition_embed4 = OverlapPatchEmbed(1, int(dim*2**3))



        self.flc2drc_condition_embed3 = OverlapPatchEmbed(1, int(dim*2**2))
        self.flc2drc_condition_embed2 = OverlapPatchEmbed(1, int(dim*2**1))
        self.flc2drc_condition_embed1 = OverlapPatchEmbed(1, int(dim*2**1))

        # self.flc2drc_condition_down = Downsample_FLC2DRC(dim*2**2)
        # self.flc2drc_condition_up3_2 = Upsample(int(dim*2**2))
        # self.flc2drc_condition_up2_1 = Upsample(int(dim*2**1))
        # self.flc2drc_condition_mlp = nn.Conv2d(dim, int(dim*2**1), kernel_size=3, stride=1, padding=1, bias=False)

        
        # ratio feature map embed
        self.sdss_ratio_condition_embed = OverlapPatchEmbed(2, dim)
        self.sdss_ratio_condition_down1_2 = Downsample(dim)
        self.sdss_ratio_condition_down2_3 = Downsample(int(dim*2**1))
        self.sdss_ratio_condition_down3_4 = Downsample(int(dim*2**2))

        self.hst_ratio_condition_embed = OverlapPatchEmbed(2, dim*2**2)
        self.hst_ratio_condition_down = Downsample_HST_RATIO(dim*2**2)
        self.hst_ratio_condition_up3_2 = Upsample(int(dim*2**2))
        self.hst_ratio_condition_up2_1 = Upsample(int(dim*2**1))
        self.hst_ratio_condition_mlp = nn.Conv2d(dim, int(dim*2**1), kernel_size=3, stride=1, padding=1, bias=False)



        # time embedding
        self.ground_time_embed1 = nn.Sequential(nn.Linear(dim, dim*2**1), nn.SiLU(), nn.Linear(dim*2**1, dim))
        self.ground_time_embed2 = nn.Sequential(nn.Linear(dim, dim*2**1), nn.SiLU(), nn.Linear(dim*2**1, dim*2**1))
        self.ground_time_embed3 = nn.Sequential(nn.Linear(dim, dim*2**2), nn.SiLU(), nn.Linear(dim*2**2, dim*2**2))
        self.ground_time_embed4 = nn.Sequential(nn.Linear(dim, dim*2**3), nn.SiLU(), nn.Linear(dim*2**3, dim*2**3))

        self.space_time_embed3 = nn.Sequential(nn.Linear(dim, dim*2**2), nn.SiLU(), nn.Linear(dim*2**2, dim*2**2))
        self.space_time_embed2 = nn.Sequential(nn.Linear(dim, dim*2**1), nn.SiLU(), nn.Linear(dim*2**1, dim*2**1))
        self.space_time_embed1 = nn.Sequential(nn.Linear(dim, dim*2**1), nn.SiLU(), nn.Linear(dim*2**1, dim*2**1))


        # sdss filter embedding
        self.ground_filter_conv1 = nn.Sequential(nn.Conv2d(4, int(dim // 2), kernel_size=3, stride=2, padding=1),
                                                 nn.BatchNorm2d(int(dim // 2)),
                                                 nn.ReLU(), 
                                                 nn.Conv2d(int(dim // 2), int(dim), kernel_size=3, stride=2, padding=1),
                                                 nn.BatchNorm2d(int(dim)),
                                                 nn.ReLU(),
        )

        self.ground_filter_conv2 = nn.Sequential(nn.Conv2d(4, int(dim // 2), kernel_size=3, stride=2, padding=1),
                                                 nn.BatchNorm2d(int(dim // 2)),
                                                 nn.ReLU(), 
                                                 nn.Conv2d(int(dim // 2), int(dim), kernel_size=3, stride=2, padding=1),
                                                 nn.BatchNorm2d(int(dim)),
                                                 nn.ReLU(),
        )

        self.ground_filter_conv3 = nn.Sequential(nn.Conv2d(4, int(dim // 2), kernel_size=3, stride=2, padding=1),
                                                 nn.BatchNorm2d(int(dim // 2)),
                                                 nn.ReLU(), 
                                                 nn.Conv2d(int(dim // 2), int(dim), kernel_size=3, stride=2, padding=1),
                                                 nn.BatchNorm2d(int(dim)),
                                                 nn.ReLU(),
        )

        self.ground_filter_conv4 = nn.Sequential(nn.Conv2d(4, int(dim // 2), kernel_size=3, stride=2, padding=1),
                                                 nn.BatchNorm2d(int(dim // 2)),
                                                 nn.ReLU(), 
                                                 nn.Conv2d(int(dim // 2), int(dim), kernel_size=3, stride=2, padding=1),
                                                 nn.BatchNorm2d(int(dim)),
                                                 nn.ReLU(),
        )

        self.ground_filter_conv5 = nn.Sequential(nn.Conv2d(4, int(dim // 2), kernel_size=3, stride=2, padding=1),
                                                 nn.BatchNorm2d(int(dim // 2)),
                                                 nn.ReLU(), 
                                                 nn.Conv2d(int(dim // 2), int(dim), kernel_size=3, stride=2, padding=1),
                                                 nn.BatchNorm2d(int(dim)),
                                                 nn.ReLU(),
        )

        self.ground_filter_fc = nn.Sequential(nn.Linear(int(dim * 5), int(dim)), nn.SiLU(), nn.Linear(dim, dim))

        # hst filter embedding
        self.space_filter_conv1 = nn.Sequential(nn.Conv2d(4, int(dim // 2), kernel_size=3, stride=2, padding=1),
                                                 nn.BatchNorm2d(int(dim // 2)),
                                                 nn.ReLU(), 
                                                 nn.Conv2d(int(dim // 2), int(dim), kernel_size=3, stride=2, padding=1),
                                                 nn.BatchNorm2d(int(dim)),
                                                 nn.ReLU(),
        )


        # self.space_filter_conv2 = nn.Sequential(nn.Conv2d(8, int(dim // 4), kernel_size=3, stride=2, padding=1),
        #                                          nn.BatchNorm2d(int(dim // 4)),
        #                                          nn.ReLU(), 
        #                                          nn.Conv2d(int(dim // 4), int(dim // 2), kernel_size=3, stride=2, padding=1),
        #                                          nn.BatchNorm2d(int(dim // 2)),
        #                                          nn.ReLU(),
        # )


        self.ground_filter_embed1 = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.ground_filter_embed2 = nn.Sequential(nn.Linear(dim, dim*2**1), nn.SiLU(), nn.Linear(dim*2**1, dim*2**1))
        self.ground_filter_embed3 = nn.Sequential(nn.Linear(dim, dim*2**2), nn.SiLU(), nn.Linear(dim*2**2, dim*2**2))
        self.ground_filter_embed4 = nn.Sequential(nn.Linear(dim, dim*2**3), nn.SiLU(), nn.Linear(dim*2**3, dim*2**3))

        self.space_filter_embed3 = nn.Sequential(nn.Linear(dim, dim*2**2), nn.SiLU(), nn.Linear(dim*2**2, dim*2**2))
        self.space_filter_embed2 = nn.Sequential(nn.Linear(dim, dim*2**1), nn.SiLU(), nn.Linear(dim*2**1, dim*2**1))
        self.space_filter_embed1 = nn.Sequential(nn.Linear(dim, dim*2**1), nn.SiLU(), nn.Linear(dim*2**1, dim*2**1))

        
        self.prompt = prompt
        
        if self.prompt:
            self.prompt1 = PromptGenBlock(prompt_dim=64,prompt_len=10,prompt_size = 64,lin_dim = 96)
            self.prompt2 = PromptGenBlock(prompt_dim=128,prompt_len=10,prompt_size = 32,lin_dim = 192)
            self.prompt3 = PromptGenBlock(prompt_dim=320,prompt_len=10,prompt_size = 16,lin_dim = 384)  # lin_dim: feature dimension
        
        
        # self.chnl_reduce1 = nn.Conv2d(64,64,kernel_size=1,bias=bias)
        # self.chnl_reduce2 = nn.Conv2d(128,128,kernel_size=1,bias=bias)
        # self.chnl_reduce3 = nn.Conv2d(320,256,kernel_size=1,bias=bias)

        self.ground_time_ca1 = SequentialWithArgs(*[TransformerBlock_CA(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])
        self.ground_filter_ca1 = SequentialWithArgs(*[TransformerBlock_CA(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])

        self.sdss2drc_condition_ca1 = SequentialWithArgs(*[TransformerBlock2D_CA(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])

        self.sdss_ratio_condition_ca1 = SequentialWithArgs(*[TransformerBlock2D_CA(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])

        # self.reduce_noise_channel_1 = nn.Conv2d(dim + 64,dim,kernel_size=1,bias=bias)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        # self.encoder_level1 = SequentialWithArgs(*[TransformerBlock_CA(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2

        # self.reduce_noise_channel_2 = nn.Conv2d(int(dim*2**1) + 128,int(dim*2**1),kernel_size=1,bias=bias)
        self.ground_time_ca2 = SequentialWithArgs(*[TransformerBlock_CA(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])
        self.ground_filter_ca2 = SequentialWithArgs(*[TransformerBlock_CA(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])
        self.sdss2drc_condition_ca2 = SequentialWithArgs(*[TransformerBlock2D_CA(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])

        self.sdss_ratio_condition_ca2 = SequentialWithArgs(*[TransformerBlock2D_CA(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])

        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        # self.encoder_level2 = SequentialWithArgs(*[TransformerBlock_CA(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3

        self.ground_time_ca3 = SequentialWithArgs(*[TransformerBlock_CA(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])
        self.ground_filter_ca3 = SequentialWithArgs(*[TransformerBlock_CA(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])
        self.sdss2drc_condition_ca3 = SequentialWithArgs(*[TransformerBlock2D_CA(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])

        self.sdss_ratio_condition_ca3 = SequentialWithArgs(*[TransformerBlock2D_CA(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])


        # self.reduce_noise_channel_3 = nn.Conv2d(int(dim*2**2) + 256,int(dim*2**2),kernel_size=1,bias=bias)
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        # self.encoder_level3 = SequentialWithArgs(*[TransformerBlock_CA(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4

        self.ground_time_ca4 = SequentialWithArgs(*[TransformerBlock_CA(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])
        self.ground_filter_ca4 = SequentialWithArgs(*[TransformerBlock_CA(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])
        self.sdss2drc_condition_ca4 = SequentialWithArgs(*[TransformerBlock2D_CA(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])

        self.sdss_ratio_condition_ca4 = SequentialWithArgs(*[TransformerBlock2D_CA(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])


        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        # self.latent = SequentialWithArgs(*[TransformerBlock_CA(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])


        self.up4_3 = Upsample(int(dim*2**2)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**1)+192, int(dim*2**2), kernel_size=1, bias=bias)
        self.noise_level3 = TransformerBlock(dim=int(dim*2**2) + 512, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level3 = nn.Conv2d(int(dim*2**2)+512,int(dim*2**2),kernel_size=1,bias=bias)

        self.expand_level3 = nn.Conv2d(int(dim*2**1), int(dim*2**2), kernel_size=1, bias=bias)

        self.space_time_ca3 = SequentialWithArgs(*[TransformerBlock_CA(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])
        self.space_filter_ca3 = SequentialWithArgs(*[TransformerBlock_CA(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])
        self.flc2drc_condition_ca3 = SequentialWithArgs(*[TransformerBlock2D_CA(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])
 
        self.hst_ratio_condition_ca3 = SequentialWithArgs(*[TransformerBlock2D_CA(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])


        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        # self.decoder_level3 = SequentialWithArgs(*[TransformerBlock_CA(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.noise_level2 = TransformerBlock(dim=int(dim*2**1) + 224, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(int(dim*2**1)+224,int(dim*2**2),kernel_size=1,bias=bias)

        self.space_time_ca2 = SequentialWithArgs(*[TransformerBlock_CA(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])
        self.space_filter_ca2 = SequentialWithArgs(*[TransformerBlock_CA(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])
        self.flc2drc_condition_ca2 = SequentialWithArgs(*[TransformerBlock2D_CA(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])


        self.hst_ratio_condition_ca2 = SequentialWithArgs(*[TransformerBlock2D_CA(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])



        self.expand_level2 = nn.Conv2d(int(dim), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        # self.decoder_level2 = SequentialWithArgs(*[TransformerBlock_CA(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])


        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.noise_level1 = TransformerBlock(dim=int(dim*2**1)+64, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(int(dim*2**1)+64,int(dim*2**1),kernel_size=1,bias=bias)

        self.space_time_ca1 = SequentialWithArgs(*[TransformerBlock_CA(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])
        self.space_filter_ca1 = SequentialWithArgs(*[TransformerBlock_CA(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])
        self.flc2drc_condition_ca1 = SequentialWithArgs(*[TransformerBlock2D_CA(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])


        self.hst_ratio_condition_ca1 = SequentialWithArgs(*[TransformerBlock2D_CA(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(1)])



        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        # self.decoder_level1 = SequentialWithArgs(*[TransformerBlock_CA(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.expand_level1 = nn.Conv2d(int(dim // 2), int(dim), kernel_size=1, bias=bias)

        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward_input(self, inp_img, source_err, exp_times, source_filter_infos, target_filter_infos, sdss2drc_condition, flc2drc_condition, sdss_ratio_feat_map, hst_ratio_feat_map):
        
        exp_times = torch.cat([\
            torch.cos(exp_times.unsqueeze(-1)*1/(65536)**torch.linspace(0, 1, 24).unsqueeze(0).unsqueeze(0).cuda()),
            torch.sin(exp_times.unsqueeze(-1)*1/(65536)**torch.linspace(0, 1, 24).unsqueeze(0).unsqueeze(0).cuda())
        ],-1)

        # import pdb; pdb.set_trace()
        ground_exp_time, space_exp_time = exp_times[:,0], exp_times[:,1]
        
        inp_enc_level1 = self.patch_embed(inp_img)
        inp_err = self.err_patch_embed(source_err) # [B, C, H, W]
        inp_enc_level1 = torch.cat([inp_enc_level1, inp_err], 1)   # [B, 2C, H, W] 
        inp_enc_level1 = self.reduce_channel(inp_enc_level1)  # [B, C, H, W]

        ori_inp_enc_level1 = inp_enc_level1

        # inp_enc_level1 += self.time_embed(exp_times[:,0]).unsqueeze(-1).unsqueeze(-1)
        ground_exp_time_embed1 = self.ground_time_embed1(ground_exp_time).unsqueeze(-1)

        source_filter_infos = torch.cat([self.ground_filter_conv1(source_filter_infos[:, 0]).flatten(2, 3), 
                                        self.ground_filter_conv2(source_filter_infos[:, 1]).flatten(2, 3),
                                        self.ground_filter_conv3(source_filter_infos[:, 2]).flatten(2, 3),
                                        self.ground_filter_conv4(source_filter_infos[:, 3]).flatten(2, 3),
                                        self.ground_filter_conv5(source_filter_infos[:, 4]).flatten(2, 3),], 1).transpose(2, 1) # [B, L, 5C]

        source_filter_infos = self.ground_filter_fc(source_filter_infos)

        # sdss2drc_condition1 = self.sdss2drc_condition_embed(sdss2drc_condition)

        sdss_ratio_feat_map1 = self.sdss_ratio_condition_embed(sdss_ratio_feat_map)


        # import pdb; pdb.set_trace()

        

        ground_filter_embed1 = self.ground_filter_embed1(source_filter_infos).transpose(2, 1)   # [B, C, L]

        inp_enc_level1 = self.ground_time_ca1(inp_enc_level1, ground_exp_time_embed1)
        inp_enc_level1 = self.ground_filter_ca1(inp_enc_level1, ground_filter_embed1)

        sdss2drc_condition_resize1 = F.interpolate(sdss2drc_condition, size=(33, 33), mode='bilinear', align_corners=False)
        sdss2drc_condition_resize1 = self.sdss2drc_condition_embed1(sdss2drc_condition_resize1)

        # import pdb; pdb.set_trace()

        inp_enc_level1 = depth_wise_conv(inp_enc_level1, sdss2drc_condition_resize1) + inp_enc_level1


        # inp_enc_level1 = self.sdss2drc_condition_ca1(inp_enc_level1, sdss2drc_condition1)

        inp_enc_level1 = self.sdss_ratio_condition_ca1(inp_enc_level1, sdss_ratio_feat_map1)


        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        
        inp_enc_level2 = self.down1_2(out_enc_level1)

        # inp_enc_level2 += self.down_time_embed2(exp_times[:,0]).unsqueeze(-1).unsqueeze(-1)
        ground_exp_time_embed2 = self.ground_time_embed2(ground_exp_time).unsqueeze(-1)
        ground_filter_embed2 = self.ground_filter_embed2(source_filter_infos).transpose(2, 1)

        inp_enc_level2 = self.ground_time_ca2(inp_enc_level2, ground_exp_time_embed2)
        inp_enc_level2 = self.ground_filter_ca2(inp_enc_level2, ground_filter_embed2)
        # sdss2drc_condition2 = self.sdss2drc_condition_down1_2(sdss2drc_condition1)

        sdss_ratio_feat_map2 = self.sdss_ratio_condition_down1_2(sdss_ratio_feat_map1)


        # import pdb; pdb.set_trace()
        # inp_enc_level2 = self.sdss2drc_condition_ca2(inp_enc_level2, sdss2drc_condition2)
        sdss2drc_condition_resize2 = F.interpolate(sdss2drc_condition, size=(17, 17), mode='bilinear', align_corners=False)
        sdss2drc_condition_resize2 = self.sdss2drc_condition_embed2(sdss2drc_condition_resize2)
        # import pdb; pdb.set_trace()

        inp_enc_level2 = depth_wise_conv(inp_enc_level2, sdss2drc_condition_resize2) + inp_enc_level2

        inp_enc_level2 = self.sdss_ratio_condition_ca2(inp_enc_level2, sdss_ratio_feat_map2)


        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        # inp_enc_level3 += self.down_time_embed3(exp_times[:,0]).unsqueeze(-1).unsqueeze(-1)
        ground_exp_time_embed3 = self.ground_time_embed3(ground_exp_time).unsqueeze(-1)
        ground_filter_embed3 = self.ground_filter_embed3(source_filter_infos).transpose(2, 1)

        inp_enc_level3 = self.ground_time_ca3(inp_enc_level3, ground_exp_time_embed3)
        inp_enc_level3 = self.ground_filter_ca3(inp_enc_level3, ground_filter_embed3)
        # sdss2drc_condition3 = self.sdss2drc_condition_down2_3(sdss2drc_condition2)

        sdss_ratio_feat_map3 = self.sdss_ratio_condition_down2_3(sdss_ratio_feat_map2)

        # inp_enc_level3 = self.sdss2drc_condition_ca3(inp_enc_level3, sdss2drc_condition3)

        sdss2drc_condition_resize3 = F.interpolate(sdss2drc_condition, size=(9, 9), mode='bilinear', align_corners=False)
        sdss2drc_condition_resize3 = self.sdss2drc_condition_embed3(sdss2drc_condition_resize3)
        # import pdb; pdb.set_trace()

        inp_enc_level3 = depth_wise_conv(inp_enc_level3, sdss2drc_condition_resize3) + inp_enc_level3

        inp_enc_level3 = self.sdss_ratio_condition_ca3(inp_enc_level3, sdss_ratio_feat_map3)

        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        

        # import pdb; pdb.set_trace()
        ground_exp_time_embed4 = self.ground_time_embed4(ground_exp_time).unsqueeze(-1)
        ground_filter_embed4 = self.ground_filter_embed4(source_filter_infos).transpose(2, 1)

        inp_enc_level4 = self.ground_time_ca4(inp_enc_level4, ground_exp_time_embed4)
        inp_enc_level4 = self.ground_filter_ca4(inp_enc_level4, ground_filter_embed4)
        # sdss2drc_condition4 = self.sdss2drc_condition_down3_4(sdss2drc_condition3)

        sdss_ratio_feat_map4 = self.sdss_ratio_condition_down3_4(sdss_ratio_feat_map3)

        # inp_enc_level4 = self.sdss2drc_condition_ca4(inp_enc_level4, sdss2drc_condition4)
        sdss2drc_condition_resize4 = F.interpolate(sdss2drc_condition, size=(5, 5), mode='bilinear', align_corners=False)
        sdss2drc_condition_resize4 = self.sdss2drc_condition_embed4(sdss2drc_condition_resize4)
        # import pdb; pdb.set_trace()

        inp_enc_level4 = depth_wise_conv(inp_enc_level4, sdss2drc_condition_resize4) + inp_enc_level4



        inp_enc_level4 = self.sdss_ratio_condition_ca4(inp_enc_level4, sdss_ratio_feat_map4)

        # inp_enc_level4 += self.down_time_embed4(exp_times[:,0]).unsqueeze(-1).unsqueeze(-1)
        latent = self.latent(inp_enc_level4)
        if self.prompt:
            dec3_param = self.prompt3(latent)

            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)
                        
        inp_dec_level3 = self.up4_3(latent)

        # target_filter_infos = torch.cat([self.space_filter_conv1(target_filter_infos[:, 0]).flatten(2, 3), self.space_filter_conv2(target_filter_infos[:, 1]).flatten(2, 3)], 1).transpose(2, 1) # [B, L, C]
        target_filter_infos = self.space_filter_conv1(target_filter_infos).flatten(2, 3).transpose(2, 1) # [B, L, C]

        # import pdb; pdb.set_trace()

        # inp_dec_level3 += self.up_time_embed4(exp_times[:,1]).unsqueeze(-1).unsqueeze(-1)
        _, c3, _, _ = out_enc_level3.shape

        # inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = torch.cat([inp_dec_level3, self.expand_level3(out_enc_level3[:, 0: int(c3 // 2)])], 1)   # [:, 0: int(c3 // 2)]

        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        
        space_exp_time_embed3 = self.space_time_embed3(space_exp_time).unsqueeze(-1)
        space_filter_embed3 = self.space_filter_embed3(target_filter_infos).transpose(2, 1)
        # flc2drc_condition3 = self.flc2drc_condition_embed(flc2drc_condition)

        hst_ratio_feat_map3 = self.hst_ratio_condition_embed(hst_ratio_feat_map)


        inp_dec_level3 = self.space_time_ca3(inp_dec_level3, space_exp_time_embed3)
        inp_dec_level3 = self.space_filter_ca3(inp_dec_level3, space_filter_embed3)

        # import pdb; pdb.set_trace()
        # flc2drc_condition3 = self.flc2drc_condition_down(flc2drc_condition3)
        hst_ratio_feat_map3 = self.hst_ratio_condition_down(hst_ratio_feat_map3)

        # inp_dec_level3 = self.flc2drc_condition_ca3(inp_dec_level3, flc2drc_condition3)
        # import pdb; pdb.set_trace()

        flc2drc_condition_resize3 = F.interpolate(flc2drc_condition, size=(5, 5), mode='bilinear', align_corners=False)
        flc2drc_condition_resize3 = self.flc2drc_condition_embed3(flc2drc_condition_resize3)
        inp_dec_level3 = depth_wise_conv(inp_dec_level3, flc2drc_condition_resize3) + inp_dec_level3


        out_dec_level3 = self.decoder_level3(inp_dec_level3) 
        if self.prompt:
            dec2_param = self.prompt2(out_dec_level3)
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        # import pdb; pdb.set_trace()

        _, c2, _, _ = out_enc_level2.shape
        inp_dec_level2 = torch.cat([inp_dec_level2, self.expand_level2(out_enc_level2[:, 0: int(c2 // 2)])], 1)   # [:, 0: int(c2 // 2)]

        # inp_dec_level2 += self.up_time_embed3(exp_times[:,1]).unsqueeze(-1).unsqueeze(-1)
        # inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        space_exp_time_embed2 = self.space_time_embed2(space_exp_time).unsqueeze(-1)
        space_filter_embed2 = self.space_filter_embed2(target_filter_infos).transpose(2, 1)
        # flc2drc_condition2 = self.flc2drc_condition_up3_2(flc2drc_condition3)

        hst_ratio_feat_map2 = self.hst_ratio_condition_up3_2(hst_ratio_feat_map3)


        inp_dec_level2 = self.space_time_ca2(inp_dec_level2, space_exp_time_embed2)
        inp_dec_level2 = self.space_filter_ca2(inp_dec_level2, space_filter_embed2)
        # inp_dec_level2 = self.flc2drc_condition_ca2(inp_dec_level2, flc2drc_condition2)

        flc2drc_condition_resize2 = F.interpolate(flc2drc_condition, size=(9, 9), mode='bilinear', align_corners=False)
        flc2drc_condition_resize2 = self.flc2drc_condition_embed2(flc2drc_condition_resize2)
        inp_dec_level2 = depth_wise_conv(inp_dec_level2, flc2drc_condition_resize2) + inp_dec_level2

        inp_dec_level2 = self.hst_ratio_condition_ca2(inp_dec_level2, hst_ratio_feat_map2)


        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        if self.prompt:
           
            dec1_param = self.prompt1(out_dec_level2)
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        # inp_dec_level1 += self.up_time_embed2(exp_times[:,1]).unsqueeze(-1).unsqueeze(-1)
        _, c1, _, _ = out_enc_level1.shape
        inp_dec_level1 = torch.cat([inp_dec_level1, self.expand_level1(out_enc_level1[:, 0: int(c1 // 2)])], 1) # [:, 0: int(c1 // 2)]
        # inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        
        space_exp_time_embed1 = self.space_time_embed1(space_exp_time).unsqueeze(-1)
        space_filter_embed1 = self.space_filter_embed1(target_filter_infos).transpose(2, 1)
        # flc2drc_condition1 = self.flc2drc_condition_mlp(self.flc2drc_condition_up2_1(flc2drc_condition2))

        hst_ratio_feat_map1 = self.hst_ratio_condition_mlp(self.hst_ratio_condition_up2_1(hst_ratio_feat_map2))

        inp_dec_level1 = self.space_time_ca1(inp_dec_level1, space_exp_time_embed1)
        inp_dec_level1 = self.space_filter_ca1(inp_dec_level1, space_filter_embed1)
        # import pdb; pdb.set_trace()
        # inp_dec_level1 = self.flc2drc_condition_ca1(inp_dec_level1, flc2drc_condition1)

        flc2drc_condition_resize1 = F.interpolate(flc2drc_condition, size=(17, 17), mode='bilinear', align_corners=False)
        flc2drc_condition_resize1 = self.flc2drc_condition_embed1(flc2drc_condition_resize1)
        inp_dec_level1 = depth_wise_conv(inp_dec_level1, flc2drc_condition_resize1) + inp_dec_level1



        inp_dec_level1 = self.hst_ratio_condition_ca1(inp_dec_level1, hst_ratio_feat_map1)

        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(ori_inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        # out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1
    

    def forward(self,inputs,targets=None, epoch=None):
        pred_img = self.forward_input(inputs, targets['source_err'], targets['exp_times'], targets['source_filter_infos'], targets['target_filter_infos'], targets['sdss2drc_condition'], targets['flc2drc_condition'], targets['sdss_ratio_feat_map'], targets['hst_ratio_feat_map'])
        pred_img, pred_img_un = pred_img[:,0:1], pred_img[:,1:2]
        pred_img_un = pred_img_un.clamp(-20, 20)
        if self.training:

            # standard nll loss
            # uncertainty_loss = (2 ** 0.5 * torch.exp(-pred_img_un) * targets['mask'] * (torch.abs(pred_img * targets['mask'] - targets['target'] * targets['mask'])) + 0.5 * pred_img_un * targets['mask']).sum() / (targets['mask'].sum() + 1e-3)
            
            # uncertainty + GT loss
            pred_img_un_test = 0.5 * torch.logsumexp(torch.stack([2 * pred_img_un, -torch.log(targets['err'])], dim=0), dim=0)
            uncertainty_loss = (2 ** 0.5 * torch.exp(-pred_img_un_test) * targets['mask'] * (torch.abs(pred_img * targets['mask'] - targets['target'] * targets['mask'])) + pred_img_un_test * targets['mask']).sum() / (targets['mask'].sum() + 1e-3)

            losses = dict(
                         uncertainty_loss=uncertainty_loss,
            )

            total_loss = torch.stack([*losses.values()]).sum()



            return total_loss, losses
        return dict(pred_img = pred_img,
                    pred_img_un = pred_img_un,
        )
