import random
import numpy as np
from tqdm import tqdm
import os
import cv2
import argparse
import torch
from torchvision.io import write_video
from fastvqa import VQAInferenceDataset, BaseEvaluator


def get_vis_dataset(args, model_type="fast"):
    dataset_path = f"{args.pdpath}/{args.dataset}"
    inference_set = VQAInferenceDataset(
        f"{dataset_path}/labels.txt",
        dataset_path,
        fragments=7 if model_type == "fast" else 4,
        clip_len=32 if model_type == "fast" else 16,
        nfrags=1,
        num_clips=1,
        aligned=32 if model_type == "fast" else 8,
        phase="train",
    )
    return inference_set


def t_rescale(pr, gt=None):
    if gt is None:
        pr = (pr - pr.mean()) / pr.std()
    else:
        pr = (pr - pr.mean()) / pr.std() * gt.std() + gt.mean()
    return pr


def save_visualizations(args, inference_set, model=None, device="cpu", seed=42):
    os.makedirs(
        f"{args.save_dir}/{args.dataset.lower()}_{args.model_type}", exist_ok=True
    )
    random.seed(seed)
    mean, std = np.array([123.675, 116.28, 103.53]), np.array([58.395, 57.12, 57.375])

    results = []

    for _ in tqdm(range(args.vs)):
        q = random.randrange(len(inference_set))
        data = inference_set.__getitem__(q, need_original_frames=True)
        vfrag, video = data["video"], data["original_video"]
        if model is not None:
            vfrag = vfrag.to(device)
            with torch.no_grad():
                result = torch.nn.functional.interpolate(
                    model(vfrag), scale_factor=(2, 32, 32), mode="nearest"
                ).cpu()
            results.append(
                (vfrag, video, result, data["original_shape"], data["gt_label"], q)
            )
        else:
            results.append(
                (vfrag, video, None, data["original_shape"], data["gt_label"], q)
            )

    if results[0][2] is not None:
        res_res = t_rescale(torch.cat([r[2] for r in results], 0))
    else:
        res_res = None

    for i, result in enumerate(tqdm(results)):
        vfrag, video, _, shape, label, q = result
        if res_res is not None:
            result = (
                torch.cat(
                    (
                        res_res[i],
                        -res_res[i],
                        torch.zeros_like(res_res[i]),
                    ),
                    0,
                )
                .permute(1, 2, 3, 0)
                .cpu()
                .numpy()
            )

        frag = vfrag.squeeze(0).permute(1, 2, 3, 0).cpu().numpy() * std + mean

        if res_res is not None:
            frag = (frag - result * frag.mean() / 2).clip(0, 255)

        video = video.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()

        save_dir = f"{args.save_dir}/{args.dataset.lower()}_{args.model_type}/{q}_{label:.2f}_{shape}"

        os.makedirs(save_dir, exist_ok=True)
        write_video(f"{save_dir}/fr.mp4", frag, 15)
        write_video(f"{save_dir}/vr.mp4", video, 15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="LIVE_VQC",
        help="the inference dataset name, can add XXX,a,b to evaluate XXX from [",
    )
    parser.add_argument(
        "--pdpath", type=str, default="../datasets/", help="the inference dataset path"
    )
    parser.add_argument(
        "-v", "--vs", type=int, default=16, help="num of visualizations"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="demo_fragments_with_originals",
        help="results_dir",
    )
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        default="fast",
        help="choose whether to use FAST-VQA or the FASTER-VQA",
    )
    parser.add_argument(
        "-nm",
        "--need_model",
        action="store_true",
        help="need the rendering of local quality maps on fragments",
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.model_type == "fast":
        backbone_hp = dict(window_size=(8, 7, 7), frag_biases=[True, True, True, False])
    else:
        backbone_hp = dict(
            window_size=(4, 4, 4), frag_biases=[False, False, True, False]
        )

    if args.need_model:

        model = BaseEvaluator(backbone_hp).to(device)

        load_path = f"pretrained_weights/{args.model_type}_vqa_v0_3.pth"
        state_dict = torch.load(load_path, map_location=device)

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

        model.load_state_dict(i_state_dict)

    dataset = get_vis_dataset(args, args.model_type)

    if args.need_model:
        save_visualizations(args, dataset, model=model, device=device)
    else:
        save_visualizations(args, dataset)


if __name__ == "__main__":
    main()
