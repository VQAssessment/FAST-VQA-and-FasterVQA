import torch
import os.path as osp
from fastvqa.models import BaseEvaluator
from fastvqa.datasets import FragmentVideoDataset

import argparse

from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
import numpy as np

from time import time
from tqdm import tqdm

from thop import profile

def rescale(pr, gt=None):
    if gt is None:
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr


all_datasets = ["LIVE_VQC", "KoNViD", "CVD2014", "LSVQ", "YouTubeUGC"]

def profile_inference(inf_set, model, device):
    video = {}
    data = inf_set[0]
    video = data["video"].to(device).squeeze(0)
    with torch.no_grad():
        flops, params = profile(model, (video, ))
    print(f"The FLOps of the Variant is {flops/1e9:.1f}G, with Params {params/1e6:.2f}M.")


def deterministic_split(dataset_path, ann_file, start=0, end=-1):
    video_infos = []
    with open(ann_file, "r") as fin:
        for line in fin:
            line_split = line.strip().split(",")
            filename, _, _, label = line_split
            label = float(label)
            filename = osp.join(dataset_path, filename)
            video_infos.append(dict(filename=filename, label=label))
    return video_infos[start:end]


def predict_dataset(args, dataset, dataset_hp, model, device):

    print(f"Predicting video quality on dataset: {dataset}.")

    ## getting datasets (if you want to load from existing VQA datasets)
    dataset_name = dataset
    if "," in dataset_name:
        dataset_name, start, end = dataset_name.split(",")
        assert dataset_name in all_datasets
        dataset_path = f"{args.pdpath}/{dataset_name}"

        test_infos = deterministic_split(
            dataset_path,
            f"examplar_data_labels/{dataset_name}/labels.txt",
            int(start),
            int(end),
        )

        inference_set = FragmentVideoDataset(
            test_infos,
            dataset_path,
            **dataset_hp,
        )
    else:
        dataset_path = f"{args.pdpath}/{dataset_name}"
        inference_set = FragmentVideoDataset(
            f"examplar_data_labels/{dataset_name}/labels.txt",
            dataset_path,
            **dataset_hp,
        )

    print(f"Inference on Dataset {args.dataset} in {dataset_path}")

    ## run inference for a whole testing database
    ## and get the accuracy for this database

    inference_loader = torch.utils.data.DataLoader(
        inference_set, batch_size=1, num_workers=6
    )
    results = []
    
    profile_inference(inference_set, model, device)



    # avoid GPU out of memory, this is set for Tesla V100, please scale based on your GPU
    max_testing_views = 24

    for i, data in enumerate(tqdm(inference_loader)):
        result = dict()
        vfrag = data["video"].to(device).squeeze(0)

        with torch.no_grad():
            if vfrag.shape[0] > max_testing_views:
                res_collections = []

                for i in range(vfrag.shape[0] // max_testing_views):
                    if args.reduction:
                        res_ = (
                            model(
                                vfrag[
                                    i * max_testing_views : (i + 1) * max_testing_views
                                ]
                            )
                            .reshape(max_testing_views // 4, -1)
                            .mean(1)
                        )
                    else:
                        res_ = model(
                            vfrag[i * max_testing_views : (i + 1) * max_testing_views]
                        )
                    res_collections.append(res_)
                result["pr_labels"] = torch.cat(res_collections, 0).cpu().numpy()

            else:
                if args.reduction:
                    result["pr_labels"] = (
                        model(vfrag).reshape(args.famount, -1).mean(1).cpu().numpy()
                    )
                else:
                    result["pr_labels"] = model(vfrag).cpu().numpy()
        result["gt_label"] = data["gt_label"].item()
        result["frame_inds"] = data["frame_inds"]
        del data
        results.append(result)

    # generate the demo video for video quality localization
    gt_labels = [r["gt_label"] for r in results]
    pr_labels = [np.mean(r["pr_labels"][:]) for r in results]
    pr_labels = rescale(pr_labels, gt_labels)
    
    import pickle as pkl
    with open("gresult.pkl","wb") as f:
        pkl.dump({"gt": gt_labels, "pr": pr_labels}, f)

    srocc = spearmanr(gt_labels, pr_labels)[0]
    plcc = pearsonr(gt_labels, pr_labels)[0]
    krocc = kendallr(gt_labels, pr_labels)[0]
    rmse = np.sqrt(((gt_labels - pr_labels) ** 2).mean())

    print(
        f"""For dataset {dataset_name} with {len(inference_set)} videos,
        the accuracy of the model is as follows:
        SROCC: {srocc:.4f}
        PLCC:  {plcc:.4f}
        KROCC: {krocc:.4f}
        RMSE:  {rmse:.4f}."""
    )

    torch.save(
        results,
        f"{args.save_dir}/results_{dataset.lower()}_s{args.fsize}*{args.fsize}_ens{args.famount}.pkl",
    )


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
        "-s", "--fsize", choices=[8, 16, 32], default=32, help="size of fragment strips"
    )
    parser.add_argument(
        "-a", "--famount", type=int, default=1, help="sample amount of fragment strips"
    )
    parser.add_argument("--save_dir", type=str, default="results", help="results_dir")
    parser.add_argument("-c", "--cache", action="store_true", help="use_cache_dataset")
    parser.add_argument(
        "-r", "--reduction", action="store_true", help="reduce_local_quality_maps"
    )
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        default="fast",
        help="choose whether to use FAST-VQA or the FASTER-VQA",
    )

    args = parser.parse_args()

    # adaptively choose the device

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # defining model and loading checkpoint

    if args.model_type == "fast":
        ## Hyper Parameters for FAST-VQA fine-tune
        dataset_hp = dict(
            fragments=7,
            fsize=args.fsize,
            nfrags=args.famount,
            cache_in_memory=False,
            clip_len=32,
            aligned=32,
        )
        backbone_hp = dict(window_size=(8, 7, 7), frag_biases=[True, True, True, False])
    else:
        # Hyper Parameters for FASTER-VQA fine-tune
        dataset_hp = dict(
            fragments=4,
            fsize=args.fsize,
            nfrags=args.famount,
            cache_in_memory=args.cache,
            clip_len=16,
            aligned=8,
        )
        backbone_hp = dict(
            window_size=(4, 4, 4), frag_biases=[False, False, True, False]
        )

    model = BaseEvaluator(backbone_hp).to(device)
    if args.fsize != 32:
        raise NotImplementedError(
            "Version 0.x does not support fragment size other than 32."
        )
    load_path = f"pretrained_weights/{args.model_type}_vqa_v0_3.pth"
    state_dict = torch.load(load_path, map_location="cpu")

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

    if args.dataset == "all":
        for dataset in all_datasets:
            start = time()
            predict_dataset(args, dataset, dataset_hp, model, device)
            end = time()
            print(f"Time: {end - start:.4e}s.")
    else:
        start = time()
        predict_dataset(args, args.dataset, dataset_hp, model, device)
        end = time()
        print(f"Time: {end - start:.4e}s.")


if __name__ == "__main__":
    main()
