import torch

from fastvqa.models import BaseEvaluator
from fastvqa.datasets import SampleFrames, get_fragments


class VQAModel:
    def __init__(
        self, pretrained=False, pretrained_path=None, model_type="fast", device="cpu"
    ):

        self.sampler = (
            SampleFrames(32, 2, 4) if model_type == "fast" else SampleFrames(16, 2, 4)
        )
        self.fragments = 7 if model_type == "fast" else 4
        backbone_hp = (
            dict(window_size=(4, 4, 4), frag_biases=[False, False, True, False])
            if model_type == "faster"
            else dict()
        )
        self.model = BaseEvaluator(backbone=backbone_hp).to(device)
        self.mean = torch.FloatTensor([0.4850, 0.4560, 0.4060]).to(device)
        self.std = torch.FloatTensor([0.2290, 0.2240, 0.2250]).to(device)

        if pretrained:
            if pretrained_path is None:
                raise NotImplementedError(
                    "Cannot directly get web pretrained path now."
                )
            else:
                self.load_pretrained(pretrained_path, device)

        print(
            f"""Successfully loaded pretrained=[{pretrained}] {model_type}-vqa model from pretrained_path=[{pretrained_path}].
                  Please make sure the input is [torch.tensor] in [(C,T,H,W)] layout and with data range [0,1]."""
        )

    def load_pretrained(self, pretrained_path, device):
        state_dict = torch.load(pretrained_path, map_location=device)

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            from collections import OrderedDict

            i_state_dict = OrderedDict()
            for key in state_dict.keys():
                if "cls" in key:
                    tkey = key.replace("cls", "vqa")
                    i_state_dict[tkey] = state_dict[key]
                else:
                    i_state_dict[key] = state_dict[key]

        self.model.load_state_dict(i_state_dict)

    def __call__(self, x, verbose=False, local_scores=False, return_fragments=False):
        ## x in (C, T, H, W)
        sampled_frames = torch.from_numpy(self.sampler(x.shape[1])).long()
        if verbose:
            print(sampled_frames)
        x = x[:, sampled_frames]
        x = get_fragments(x, self.fragments)
        x = ((x.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)
        x = x.reshape((3, 4, 32) + x.shape[2:]).transpose(0, 1)
        y = self.model(x)
        if verbose:
            print(x.shape, y.shape)
        return {
            "score": torch.mean(y).item(),
            "local_scores": y.cpu().numpy() if local_scores else None,
            "input_tensor": x.cpu().numpy() if return_fragments else None,
        }


def deep_end_to_end_vqa(
    pretrained=False, pretrained_path=None, model_type="fast", device="cpu"
):
    return VQAModel(
        pretrained,
        pretrained_path=pretrained_path,
        model_type=model_type,
        device=device,
    )
