import torch
import torch.nn as nn
import time
from torch.nn.functional import adaptive_avg_pool3d
from functools import partial
from .backbone import SwinTransformer3D as Backbone
from .head import VQAHead


class BaseEvaluator(nn.Module):
    def __init__(self,
                 backbone=dict(),
                 vqa_head=dict(),
                 teacher=None,
                 multi=False,
                 backbone_f=None,
                 vqa_head_f=dict(),
                 vqa_head_w=dict(in_channels=1536),
                ):
        super().__init__()
        self.backbone = Backbone(**backbone)
        self.vqa_head = VQAHead(**vqa_head)
        self.p3d = partial(adaptive_avg_pool3d, output_size=1)
        if backbone_f is not None:
            self.backbone_f = Backbone(**backbone_f)
            self.vqa_head_f = VQAHead(**vqa_head_f)
            self.vqa_head_w = VQAHead(**vqa_head_w)
                
    def forward(self, vclip, fvclip=None, inference=True, **kwargs):
        if inference:
            self.eval()
            with torch.no_grad():
                feat = self.backbone(vclip)
                score = self.vqa_head(feat)
                if fvclip is not None:
                    feat_f = self.backbone_f(fvclip)
                    score_f = self.vqa_head_f(feat_f)
                    feat_w = torch.cat((self.p3d(feat), self.p3d(feat_f)), 1)
                    score_w = self.vqa_head_w(feat_w)
                    score = torch.cat((score, score_f, score_w), -1)
            self.train()
            return score
        else:
            raise NotImplementedError               
                
        
            