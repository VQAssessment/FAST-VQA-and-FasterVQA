import torch

from fastvqa.models import BaseEvaluator
from fastvqa.datasets import SampleFrames, get_fragments


class VQAModel:
    def __init__(self, pretrained=False, pretrained_path=None, device='cpu'):
        self.model = BaseEvaluator().to(device)
        self.sampler = SampleFrames(32, 2, 4)
        if pretrained:
            if pretrained_path is None:
                raise NotImplementedError('Cannot directly get web pretrained path now.')
            state_dict = torch.load(pretrained_path, map_location=device)

            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                from collections import OrderedDict
                i_state_dict = OrderedDict()
                for key in state_dict.keys():
                    if 'cls' in key:
                        tkey = key.replace('cls', 'vqa')
                        i_state_dict[tkey] = state_dict[key]
                    else:
                        i_state_dict[key] = state_dict[key]

            self.model.load_state_dict(i_state_dict)        
            
    def __call__(self, x, 
                 verbose=False,
                 local_scores=False, 
                 return_fragments=False):
        ## x in (C, T, H, W)
        sampled_frames = torch.from_numpy(self.sampler(x.shape[1])).long()
        if verbose:
            print(sampled_frames)
        x = x[:, sampled_frames]
        x = get_fragments(x)
        x = x.reshape((3, 4, 32) + x.shape[2:]).transpose(0,1)
        y = self.model(x)
        if verbose:
            print(x.shape, y.shape)
        return {'score': torch.mean(y).item(),
                'local_scores': y.cpu().numpy() if local_scores else None,
                'input_tensor': x.cpu().numpy() if return_fragments else None,
               }
                


def deep_end_to_end_vqa(pretrained=False, pretrained_path=None, device='cpu'):
    return VQAModel(pretrained, pretrained_path=pretrained_path, device=device)
