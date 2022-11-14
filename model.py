
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "self-classifier")) 
from src.utils import trunc_normal_
#In this file we will develop the model by using ResNet-18 as the backbone


class Model(nn.Module):
    def __init__(self, base_model, dim=128, hidden_dim=4096, cls_size=10, num_cls=1,
                 num_hidden=3, backbone_dim=512, use_bn=False, fixed_cls=False, no_leaky=False):
        super(Model, self).__init__()
        """
        base_model: the backbone model
        dim: the dimension of the embedding
        hidden_dim: the dimension of the hidden layer in MLP
        cls_size: classification layer size in MLP
        num_cls: the number of tasks
        num_hidden: the number of hidden layers in MLP
        backbone_dim: the dimension of the backbone output
        """
        self.cls_size = cls_size
        self.dim = dim if num_hidden > 0 else backbone_dim
        self.hidden_dim = hidden_dim
        self.num_cls = num_cls
        self.num_hidden = num_hidden
        self.backbone_dim = backbone_dim
        self.use_bn = use_bn
        self.fixed_cls = fixed_cls
        self.no_leaky = no_leaky

        # backbone
        self.backbone = base_model
        self.backbone.fc = nn.Identity()
        
        # classification head
        self.mlp_head = MLPHead(in_dim=self.backbone_dim,
                                out_dim=self.dim,
                                use_bn=self.use_bn,
                                nlayers=self.num_hidden,
                                hidden_dim=self.hidden_dim,
                                no_leaky=self.no_leaky)
        # classification head

        #for cls_i in range(self.num_cls): --> this is the original code. Since we have only one task, we will not use a loop
        if len(self.cls_size)==1:
            self.cls_size = self.cls_size[0]
        cls_layer_i = nn.utils.weight_norm(nn.Linear(dim, self.cls_size, bias=False))
        cls_layer_i.weight_g.data.fill_(1)
        setattr(self, "cls_0", cls_layer_i)
        
        if self.fixed_cls:
            for param in getattr(self, "cls_0").parameters():
                param.requires_grad = False
                
                
    def forward(self, x, cls_num=None, return_embds=False):
        if isinstance(x, list):  # multiple views
            bs_size = x[0].shape[0]
            if return_embds:
                # run backbone forward pass separately on each resolution input.
                idx_crops = th.cumsum(th.unique_consecutive(th.Tensor([inp.shape[-1] for inp in x]), return_counts=True)[1], 0)
                start_idx = 0
                for end_idx in idx_crops:
                    _out = self.backbone(th.cat(x[start_idx: end_idx]))
                    if start_idx == 0:
                        output = _out #shape: (4, 512)
                    else:
                        output = th.cat((output, _out))#shape: (batch_size*n_augm, 512)
                    start_idx = end_idx

                # run classification head forward pass on concatenated features
                embds = self.mlp_head(output)#shape: (batch_size*n_augm, 128)
                # convert back to list of views
                embds = [embds[x: x + bs_size] for x in range(0, len(embds), bs_size)]
                return embds
            else:  # input is embds
                # concatenate features
                x = th.cat(x, 0)#shape: (batch_size*n_augm, 128)

                # apply classifiers
                if cls_num is None:
                    # apply all classifiers
                    out = [getattr(self, "cls_0")(x) for cls in range(self.num_cls)]#shape: [ (batch_size*n_augm, 5) ]
                else:
                    # apply only cls_num
                    out = getattr(self, "cls_0")(x)

                # convert to list of lists (classifiers and views)
                output = [[out[cls][x: x + bs_size] for x in range(0, len(out[cls]), bs_size)]
                          for cls in range(len(out))]
        else:  # single view
            x = self.backbone(x)
            x = self.mlp_head(x)

            if return_embds:
                return x
            else:
                # apply classifiers
                if cls_num is None:
                    # apply all classifiers
                    output = [getattr(self, "cls_%d" % cls)(x) for cls in range(self.num_cls)]
                else:
                    # apply only cls_num
                    output = getattr(self, "cls_%d" % cls_num)(x)

        return output#[ [probs1, probs2, ..,probsBatchSize]*num_augms ]
    
    
class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, nlayers=3, hidden_dim=4096, no_leaky=False):
        super().__init__()
        if nlayers == 0:
            self.mlp = nn.Identity()
        elif nlayers == 1:
            self.mlp = nn.Linear(in_dim, out_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if no_leaky:
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.LeakyReLU(inplace=True))
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                if no_leaky:
                    layers.append(nn.ReLU(inplace=True))
                else:
                    layers.append(nn.LeakyReLU(inplace=True))
            layers.append(nn.Linear(hidden_dim, out_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, p=2, dim=1, eps=1e-7)
        return x