import torch
import cv2
import random
import os.path as osp
from fastvqa.models import BaseEvaluator
from fastvqa.datasets import (
    FragmentVideoDataset,
    FastVQAPlusPlusDataset,
    get_spatial_fragments,
)

import argparse

from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
import numpy as np

from time import time
from tqdm import tqdm
import pickle
import math

import wandb


def rank_loss(y_pred, y):
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
        torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
    ).float()


def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()


def train_test_split(dataset_path, ann_file, ratio=0.8, seed=42):
    random.seed(seed)
    video_infos = []
    with open(ann_file, "r") as fin:
        for line in fin.readlines():
            line_split = line.strip().split(",")
            filename, _, _, label = line_split
            label = float(label)
            filename = osp.join(dataset_path, filename)
            video_infos.append(dict(filename=filename, label=label))
    random.shuffle(video_infos)
    return (
        video_infos[: int(ratio * len(video_infos))],
        video_infos[int(ratio * len(video_infos)) :],
    )


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


def rescale(pr, gt=None):
    if gt is None:
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr


all_datasets = ["LIVE_VQC", "KoNViD", "CVD2014", "YouTubeUGC", "LSVQ"]


def generate_dataset(args, dataset, seed=42, dataset_hp=dict()):

    if "-" in dataset:
        finetune_dataset_name = dataset.split("-")[0]
        if finetune_dataset_name == "LSVQ":
            train_infos = f"examplar_data_labels/train_labels.txt"
        else:
            train_infos = f"examplar_data_labels/{finetune_dataset_name}/labels.txt"
        train_dataset_path = f"{args.pdpath}/{finetune_dataset_name}"
        val_dataset_name = dataset.split("-")[1]
        val_dataset_path = f"{args.pdpath}/{val_dataset_name}"
        val_infos = f"examplar_data_labels/{val_dataset_name}/labels.txt"
        finetune_set = FastVQAPlusPlusDataset(
            train_infos,
            train_dataset_path,
            num_clips=1,
            phase="train",
            **dataset_hp,
        )
        if "," in val_dataset_name:
            dataset_name, start, end = val_dataset_name.split(",")
            assert dataset_name in all_datasets
            val_dataset_path = f"{args.pdpath}/{dataset_name}"

            val_infos = deterministic_split(
                val_dataset_path,
                f"examplar_data_labels/{dataset_name}/labels.txt",
                int(start),
                int(end),
            )

        validation_set = FastVQAPlusPlusDataset(
            val_infos,
            val_dataset_path,
            num_clips=1 if args.model_type == 'fast-pp' else 4,
            **dataset_hp,
        )

    else:
        print(f"Predicting video quality on dataset: {dataset}.")

        ## getting datasets (if you want to load from existing VQA datasets)
        dataset_name = dataset
        dataset_path = f"{args.pdpath}/{dataset_name}"

        train_infos, val_infos = train_test_split(
            dataset_path, f"examplar_data_labels/{dataset_name}/labels.txt", seed=seed
        )

        finetune_set = FastVQAPlusPlusDataset(
            train_infos,
            dataset_path,
            num_clips=1,
            phase="train",
            **dataset_hp,
        )

        validation_set = FastVQAPlusPlusDataset(
            val_infos,
            dataset_path,
            num_clips=1 if args.model_type == 'fast-pp' else 4,
            **dataset_hp,
        )

        print(
            f"Fine-tuning on Dataset {args.dataset} in {dataset_path}, with hyper-parameters {dataset_hp}."
        )

    return finetune_set, validation_set


def generate_train_test_loader(args, seed=42, dataset_hp=dict()):

    dataset = args.dataset

    ft_set, val_set = generate_dataset(args, dataset, seed=seed, dataset_hp=dataset_hp)

    print(len(ft_set), len(val_set))

    ft_loader = torch.utils.data.DataLoader(
        ft_set, batch_size=args.bs, num_workers=1, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=1, num_workers=6, pin_memory=True
    )

    return ft_loader, val_loader


def finetune_epoch(ft_loader, model, model_ema, optimizer, scheduler, device, epoch=-1):
    model.train()
    for i, data in enumerate(tqdm(ft_loader, desc=f"Training in epoch {epoch}")):
        optimizer.zero_grad()
        vfrag = data["video"].to(device).squeeze(1)
        y = data["gt_label"].float().detach().to(device).unsqueeze(-1)
        frame_inds = data["frame_inds"]
        y_pred = model(vfrag, inference=False).mean((-3, -2, -1))
        p_loss, r_loss = plcc_loss(y_pred, y), rank_loss(y_pred, y)
        loss = p_loss + 0.1 * r_loss
        wandb.log(
            {
                "train/plcc_loss": p_loss.item(),
                "train/rank_loss": r_loss.item(),
                "train/total_loss": loss.item(),
            }
        )

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if model_ema is not None:
            model_params = dict(model.named_parameters())
            model_ema_params = dict(model_ema.named_parameters())
            for k in model_params.keys():
                model_ema_params[k].data.mul_(0.999).add_(
                    model_params[k].data, alpha=1 - 0.999
                )
    model.eval()


def inference_set(args, inf_loader, model, device, best_, save_model=False):

    results = []

    best_s, best_p, best_k, best_r = best_

    for i, data in enumerate(tqdm(inf_loader, desc="Validating")):
        result = dict()
        vfrag = data["video"].to(device).squeeze(0)
        with torch.no_grad():
            result["pr_labels"] = model(vfrag).cpu().numpy()
        result["gt_label"] = data["gt_label"].item()
        # result['frame_inds'] = data['frame_inds']
        # del data
        results.append(result)

    ## generate the demo video for video quality localization
    gt_labels = [r["gt_label"] for r in results]
    pr_labels = [np.mean(r["pr_labels"][:]) for r in results]
    pr_labels = rescale(pr_labels, gt_labels)

    s = spearmanr(gt_labels, pr_labels)[0]
    p = pearsonr(gt_labels, pr_labels)[0]
    k = kendallr(gt_labels, pr_labels)[0]
    r = np.sqrt(((gt_labels - pr_labels) ** 2).mean())

    wandb.log({"val/SRCC": s, "val/PLCC": p, "val/KRCC": k, "val/RMSE": r})

    if s + p > best_s + best_p and save_model:
        state_dict = model.state_dict()
        torch.save(
            {
                "state_dict": state_dict,
                "validation_results": best_,
            },
            f"pretrained_weights/{args.model_type}_vqa_dev_from_{args.dataset}.devpt",
        )

    best_s, best_p, best_k, best_r = (
        max(best_s, s),
        max(best_p, p),
        max(best_k, k),
        min(best_r, r),
    )

    wandb.log(
        {
            "val/best_SRCC": best_s,
            "val/best_PLCC": best_p,
            "val/best_KRCC": best_k,
            "val/best_RMSE": best_r,
        }
    )

    print(
        f"For {len(inf_loader)} videos, \nthe accuracy of the model is as follows:\n  SROCC: {s:.4f} best: {best_s:.4f} \n  PLCC:  {p:.4f} best: {best_p:.4f}  \n  KROCC: {k:.4f} best: {best_k:.4f} \n  RMSE:  {r:.4f} best: {best_r:.4f}."
    )

    return best_s, best_p, best_k, best_r

    # torch.save(results, f'{args.save_dir}/results_{dataset.lower()}_s{32}*{32}_ens{args.famount}.pkl')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        choices=[
            "LIVE_VQC",
            "LIVE_VQA",
            "KoNViD",
            "CVD2014",
            "YouTubeUGC",
            "LIVE_Qualcomm",
            "LSVQ-LIVE_VQC",
            "LSVQ-KoNViD",
            "KoNViD-LIVE_VQC",  ## use LIVE-VQC as validation set (cross)
            "LIVE_VQC-KoNViD",  ## use KoNViD-1k as validation set (cross)
            "LSVQ-LSVQ,0,7186",  ## use LSVQ-test as validation set
            "LSVQ-LSVQ,7186,10759",  ## LSVQ-1080p as validation set
        ],
        default="LIVE_VQC",
        help="the finetune dataset name",
    )
    parser.add_argument(
        "--pdpath", type=str, default="../datasets/", help="the inference dataset name"
    )
    parser.add_argument("-b", "--bs", type=int, default=16, help="batchsize")
    parser.add_argument(
        "-a", "--famount", type=int, default=1, help="sample amount of fragment strips"
    )
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        default="fast",
        help="choose whether to use FAST-VQA (fast) or the FAST-VQA-M (fast-m); in development: FAST-VQA++ (fast-pp)",
    )
    parser.add_argument(
        "-lep", "--l_num_epochs", type=int, default=10, help="linear finetune epochs"
    )
    parser.add_argument(
        "-ep", "--num_epochs", type=int, default=20, help="finetune epochs"
    )
    parser.add_argument(
        "-wep", "--warmup_epochs", type=float, default=2.5, help="warmup epochs"
    )
    parser.add_argument(
        "-ema",
        "--exponentially_moving_average",
        action="store_true",
        help="apply_ema_decay",
    )
    parser.add_argument(
        "-s",
        "--save_model",
        action="store_true",
        help="save_the_model_for_the_best_epoch",
    )
    parser.add_argument("--save_dir", type=str, default="results", help="results_dir")
    parser.add_argument("-c", "--cache", action="store_true", help="use_cache_dataset")
    parser.add_argument(
        "-var",
        "--from_ar",
        action="store_true",
        help="use_features_from_action_recognition",
    )

    args = parser.parse_args()

    ## adaptively choose the device

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    ## defining model and loading checkpoint

    bests_ = []

    torch.save(
        {"results": bests_},
        f'{args.save_dir}/results_{args.model_type}_finetune_{args.dataset.lower()}_s{32}*{32}_ens{args.famount}{"" if not args.from_ar else "_from_ar"}.pkl',
    )

    if args.model_type == "fast":
        ## Hyper Parameters for FAST-VQA fine-tune
        dataset_hp = dict(
            fragments=(1, 7, 7),
            fsize=(32, 32, 32),
            nfrags=args.famount,
            cache_in_memory=False,
            aligned=32,
            fallback_type='upsample',
        )
        backbone_hp = dict(window_size=(8, 7, 7), frag_biases=[True, True, True, False])
    elif args.model_type == "fast-m":
        # Hyper Parameters for FASTER-VQA fine-tune
        dataset_hp = dict(
            fragments=(1, 4, 4),
            fsize=(16, 32, 32),
            nfrags=args.famount,
            cache_in_memory=args.cache,
            aligned=8,
            fallback_type='upsample',
        )
        backbone_hp = dict(
            window_size=(4, 4, 4), frag_biases=[False, False, True, False]
        )
    elif args.model_type == "fast-pp":
        ## Hyper Parameters for FAST-VQA fine-tune
        dataset_hp = dict(
            fragments=(4, 8, 8),
            fsize=(8, 32, 32),
            nfrags=args.famount,
            cache_in_memory=False,
            aligned=8,
        )
        backbone_hp = dict(
            window_size=[(2, 8, 8), (2, 4, 4), (2, 2, 2), (8, 8, 8)],
            frag_biases=[False, False, False, False],
        )

    total_splits = 1 if "LSVQ" in args.dataset else 10
    print(total_splits)
    for i in range(total_splits):
        run = wandb.init(
            project=f"end_to_end_vqa_runs, {args.model_type}",
            name=f"{args.dataset}_run_{i}",
            reinit=True,
        )
        model = BaseEvaluator(backbone_hp).to(device)

        if args.from_ar:
            load_path = "../model_baselines/NetArch/swin_tiny_patch244_window877_kinetics400_1k.pth"
            #load_path = f"pretrained_weights/fast_vqa_v0_3.pth"
        else:
            if 32 != 32:
                raise NotImplementedError(
                    "Version 0.x only supports 32*32 finetune on fragments."
                )
            load_path = (
                f"pretrained_weights/{args.model_type}_vqa_v0_3.pth"
            )
        state_dict = torch.load(load_path, map_location=device)

        if "state_dict" in state_dict:
            ### migrate training weights from mmaction
            state_dict = state_dict["state_dict"]
            from collections import OrderedDict

            i_state_dict = OrderedDict()
            for key in state_dict.keys():
                if "cls" in key:
                    tkey = key.replace("cls", "vqa")
                    if args.l_num_epochs == 0 and not args.from_ar:
                        i_state_dict[tkey] = state_dict[key]
                else:
                    i_state_dict[key] = state_dict[key]

        t_state_dict = model.state_dict()
        for key, value in t_state_dict.items():
            if key in i_state_dict and i_state_dict[key].shape != value.shape:
                i_state_dict.pop(key)
        model.load_state_dict(i_state_dict, strict=False)

        if args.exponentially_moving_average:
            from copy import deepcopy

            model_ema = deepcopy(model)
        else:
            model_ema = None

        ft_loader, val_loader = generate_train_test_loader(
            args, seed=42 * (i + 1), dataset_hp=dataset_hp
        )

        # finetune the model
        print(len(ft_loader), len(val_loader))

        optimizer = torch.optim.AdamW(
            lr=1e-3,
            params=[
                {"params": model.backbone.parameters(), "lr": 1e-4},
                {"params": model.vqa_head.parameters(), "lr": 1e-3},
            ],
        )

        warmup_iter = int(args.warmup_epochs * len(ft_loader))
        max_iter = int((args.num_epochs + args.l_num_epochs) * len(ft_loader))
        lr_lambda = (
            lambda cur_iter: cur_iter / warmup_iter
            if cur_iter <= warmup_iter
            else 0.5 * (1 + math.cos(math.pi * (cur_iter - warmup_iter) / max_iter))
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[lr_lambda, lr_lambda]
        )

        best_ = -1, -1, -1, 1000
        best_ = inference_set(
            args,
            val_loader,
            model_ema if model_ema is not None else model,
            device,
            best_,
            save_model=False,
        )

        print(
            f"""Before the finetune process on {args.dataset} with {len(val_loader)} videos, 
            the accuracy of the model is as follows:
            SROCC: {best_[0]:.4f}
            PLCC:  {best_[1]:.4f}
            KROCC: {best_[2]:.4f}
            RMSE:  {best_[3]:.4f}."""
        )

        for param in model.backbone.parameters():
            param.requires_grad = False

        for epoch in range(args.l_num_epochs):
            print(f"Split {i}, Linear Epoch {epoch}:")
            finetune_epoch(
                ft_loader, model, model_ema, optimizer, scheduler, device, epoch
            )
            best_ = inference_set(
                args,
                val_loader,
                model_ema if model_ema is not None else model,
                device,
                best_,
                save_model=args.save_model,
            )

        print(
            f"""For the linear transfer process on {args.dataset} with {len(val_loader)} videos,
            the best validation accuracy of the model is as follows:
            SROCC: {best_[0]:.4f}
            PLCC:  {best_[1]:.4f}
            KROCC: {best_[2]:.4f}
            RMSE:  {best_[3]:.4f}."""
        )

        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(args.num_epochs):
            print(f"Split {i}, Finetune Epoch {epoch}:")
            finetune_epoch(
                ft_loader, model, model_ema, optimizer, scheduler, device, epoch
            )
            best_ = inference_set(
                args,
                val_loader,
                model_ema if model_ema is not None else model,
                device,
                best_,
                save_model=args.save_model,
            )

        print(
            f"""For the finetune process on {args.dataset} with {len(val_loader)} videos,
            the best validation accuracy of the model is as follows:
            SROCC: {best_[0]:.4f}
            PLCC:  {best_[1]:.4f}
            KROCC: {best_[2]:.4f}
            RMSE:  {best_[3]:.4f}."""
        )

        bests_.append(best_)
        del model

        torch.save(
            {"results": bests_},
            f'{args.save_dir}/results_{args.model_type}_finetune_{args.dataset.lower()}_s{32}*{32}_ens{args.famount}{"" if not args.from_ar else "_from_ar"}.pkl',
        )

        run.finish()

    torch.save(
        {"results": bests_},
        f'{args.save_dir}/results_{args.model_type}_finetune_{args.dataset.lower()}_s{32}*{32}_ens{args.famount}{"" if not args.from_ar else "_from_ar"}.pkl',
    )


if __name__ == "__main__":
    main()
