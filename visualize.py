import random
import numpy as np
import os
import cv2
import argparse
from torchvision.io import write_video


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


def save_visualizations(args, seed=42):
    os.makedirs(
        f"{args.save_dir}/{args.dataset.lower()}_{args.model_type}", exist_ok=True
    )
    random.seed(seed)
    mean, std = np.array([123.675, 116.28, 103.53]), np.array([58.395, 57.12, 57.375])
    for _ in range(args.vs):
        q = random.randrange(len(inference_set))
        data = inference_set.__getitem__(q, need_original_frames=True)
        frag, video = data["video"], data["original_video"]
        frag = frag.squeeze(0).permute(1, 2, 3, 0).cpu().numpy() * std + mean
        video = video.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
        shape, label = data["original_shape"], data["gt_label"]
        save_dir = f"{args.save_dir}/{args.dataset.lower()}_{args.model_type}/{q}_{label:.2f}_{shape}"
        fourcc = cv2.VideoWriter.fourcc("m", "p", "4", "v")

        os.makedirs(save_dir, exist_ok=True)
        write_video(f"{save_dir}/fr.mp4", frag, 15)
        for i in range(frag.shape[0]):
            cv2.imwrite(f"{save_dir}/fr_{i}.jpg", frag[i][:, :, ::-1])
        write_video(f"{save_dir}/vr.mp4", video, 15)
        for i in range(video.shape[0]):
            cv2.imwrite(f"{save_dir}/vr_{i}.jpg", video[i][:, :, ::-1])


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

    args = parser.parse_args()

    get_vis_dataset(args, args.model_type)
    save_visualizations(args)


if __name__ == "__main__":
    main()
