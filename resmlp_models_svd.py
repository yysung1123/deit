# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import  PatchEmbed , _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_,  DropPath


__all__ = [
    'resMLP_12', 'resMLP_24', 'resMLP_36', 'resmlpB_24'
]


class SVD_Linear(nn.Module):
    def __init__(self,ratio=0.8,in_channel=1024,out_channel=4096,bias=True):
        super().__init__()
        self.num_components = int(min(in_channel,out_channel)*ratio)
        
        self.S=torch.nn.Linear(in_channel,self.num_components,bias=False)
        self.D=torch.nn.Linear(self.num_components,out_channel,bias=bias)
        
    def forward(self,x,scaling_factor=None):
        x = self.S(x)
        x = self.D(x)

        return x
    
class DropSVD(nn.Module):
    def __init__(self,ratio,num_components):
        super().__init__()
        self.ratio = ratio
        self.num_components = num_components
        self.rand = torch.distributions.uniform.Uniform(self.ratio, 1.0)
        
    def forward(self,x):
        if self.training:
            idx = int(self.rand.sample().item() * self.num_components)
        else:
            idx = int(self.ratio * self.num_components)
        mask = torch.zeros((1, self.num_components)).to(x.device)
        mask[:,:idx] = 1
        x = x * mask
        return x
    
class DropSVD_Linear(nn.Module):
    def __init__(self,ratio=0.8,in_channel=1024,out_channel=4096,bias=True):
        super().__init__()
        self.num_components = min(in_channel,out_channel)
        
        self.S=torch.nn.Linear(in_channel,self.num_components,bias=False)
        self.dropsvd = DropSVD(ratio, self.num_components)
        self.D=torch.nn.Linear(self.num_components,out_channel,bias=bias)
        
    def forward(self,x,scaling_factor=None):
        x = self.S(x)
        x = self.dropsvd(x)
        x = self.D(x)

        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,layer=None,is_svd=None,SVD_Config=None,drop_svd=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.is_svd = is_svd
        self.drop_svd = drop_svd
        if ((not self.is_svd and not self.drop_svd) or self.is_svd == None):
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
        elif self.is_svd:
            
            ratio1 = SVD_Config[layer]['fc1']
            ratio2 = SVD_Config[layer]['fc2']
            if(ratio1==None):
                self.fc1 = nn.Linear(in_features, hidden_features)
            else:
                
                self.fc1 = SVD_Linear(ratio1,in_features, hidden_features)
            if(ratio2==None):
                self.fc2 = nn.Linear(hidden_features, out_features)
            else:
                
                self.fc2 = SVD_Linear(ratio2,hidden_features, out_features)
        elif self.drop_svd:
            ratio1 = SVD_Config[layer]['fc1']
            ratio2 = SVD_Config[layer]['fc2']
            if ratio1==None:
                self.fc1 = nn.Linear(in_features, hidden_features)
            else:
                self.fc1 = DropSVD_Linear(ratio1,in_features, hidden_features)
            if ratio2==None:
                self.fc2 = nn.Linear(hidden_features, out_features)
            else:
                self.fc2 = DropSVD_Linear(ratio2,hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.alpha * x + self.beta    
    
class layers_scale_mlp_blocks(nn.Module):

    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU,init_values=1e-4,num_patches = 196,layer=None,is_svd=None,SVD_Config=None,drop_svd=False):
        super().__init__()
        self.norm1 = Affine(dim)
        self.attn = nn.Linear(num_patches, num_patches)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = Affine(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(4.0 * dim), act_layer=act_layer, drop=drop,layer=layer,is_svd=is_svd,SVD_Config=SVD_Config,drop_svd=drop_svd)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x).transpose(1,2)).transpose(1,2))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x 


class resmlp_models(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,drop_rate=0.,
                 Patch_layer=PatchEmbed,act_layer=nn.GELU,
                drop_path_rate=0.0,init_scale=1e-4,is_svd = False,SVD_Config =None,is_relu=False,drop_svd=False):
        super().__init__()


        if(is_relu):
            act_layer=nn.ReLU
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  

        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=int(in_chans), embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        dpr = [drop_path_rate for i in range(depth)]

        self.blocks = nn.ModuleList([
            layers_scale_mlp_blocks(
                dim=embed_dim,drop=drop_rate,drop_path=dpr[i],
                act_layer=act_layer,init_values=init_scale,
                num_patches=num_patches,is_svd=is_svd, SVD_Config =SVD_Config,layer=str(i),drop_svd=drop_svd)
            for i in range(depth)])


        self.norm = Affine(embed_dim)



        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)

        for i , blk in enumerate(self.blocks):
            x  = blk(x)

        x = self.norm(x)
        x = x.mean(dim=1).reshape(B,1,-1)

        return x[:, 0]

    def forward(self, x):
        x  = self.forward_features(x)
        x = self.head(x)
        return x 

@register_model
def resmlp_12(pretrained=False,dist=False,is_svd=None,SVD_Config=None, **kwargs):
    model = resmlp_models(
        patch_size=16, embed_dim=384, depth=12,
        Patch_layer=PatchEmbed,
        init_scale=0.1,is_svd=is_svd,SVD_Config=SVD_Config,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        if dist:
          url_path = "https://dl.fbaipublicfiles.com/deit/resmlp_12_dist.pth"
        else:
          url_path = "https://dl.fbaipublicfiles.com/deit/resmlp_12_no_dist.pth"
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url_path,
            map_location="cpu", check_hash=True
        )
            
        model.load_state_dict(checkpoint)
    return model
  
@register_model
def resmlp_24(pretrained=False,dist=False,dino=False,is_svd=None,SVD_Config=None,is_relu=False, **kwargs):
    model = resmlp_models(
        patch_size=16, embed_dim=384, depth=24,
        Patch_layer=PatchEmbed,
        init_scale=1e-5,is_svd=is_svd,SVD_Config=SVD_Config,is_relu=is_relu,**kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        if dist:
          url_path = "https://dl.fbaipublicfiles.com/deit/resmlp_24_dist.pth"
        elif dino:
          url_path = "https://dl.fbaipublicfiles.com/deit/resmlp_24_dino.pth"
        else:
          url_path = "https://dl.fbaipublicfiles.com/deit/resmlp_24_no_dist.pth"
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url_path,
            map_location="cpu", check_hash=True
        )
            
        model.load_state_dict(checkpoint)
    return model
  
@register_model
def resmlp_36(pretrained=False,dist=False, **kwargs):
    model = resmlp_models(
        patch_size=16, embed_dim=384, depth=36,
        Patch_layer=PatchEmbed,
        init_scale=1e-6,**kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        if dist:
          url_path = "https://dl.fbaipublicfiles.com/deit/resmlp_36_dist.pth"
        else:
          url_path = "https://dl.fbaipublicfiles.com/deit/resmlp_36_no_dist.pth"
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url_path,
            map_location="cpu", check_hash=True
        )
            
        model.load_state_dict(checkpoint)
    return model

@register_model
def resmlpB_24(pretrained=False,dist=False, in_22k = False, is_svd=None,SVD_Config=None,**kwargs):
    model = resmlp_models(
        patch_size=8, embed_dim=768, depth=24,
        Patch_layer=PatchEmbed,
        init_scale=1e-6,is_svd=is_svd,SVD_Config=SVD_Config,**kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        if dist:
          url_path = "https://dl.fbaipublicfiles.com/deit/resmlpB_24_dist.pth"
        elif in_22k:
          url_path = "https://dl.fbaipublicfiles.com/deit/resmlpB_24_22k.pth"
        else:
          url_path = "https://dl.fbaipublicfiles.com/deit/resmlpB_24_no_dist.pth"
            
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url_path,
            map_location="cpu", check_hash=True
        )
            
        model.load_state_dict(checkpoint)
    
    return model
