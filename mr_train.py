import torch
import cv2
import random
import os.path as osp
import fastvqa.models as models
import fastvqa.datasets as datasets

import argparse

from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
import numpy as np

from time import time
from tqdm import tqdm
import pickle
import math

import wandb
import yaml

from thop import profile


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

def rescaled_l2_loss(y_pred, y):
    y_pred_rs = (y_pred - y_pred.mean()) / y_pred.std()
    y_rs = (y - y.mean()) / (y.std() + eps)
    return torch.nn.functional.mse_loss(y_pred_rs, y_rs)

def rplcc_loss(y_pred, y, eps=1e-8):
    ## Literally (1 - PLCC) / 2
    cov = torch.cov(y_pred, y)
    std = (torch.std(y_pred) + eps) * (torch.std(y) + eps)
    return (1 - cov / std) / 2

def self_similarity_loss(f, f_hat):
    return 1 - torch.nn.functional.cosine_similarity(f.mean((-3,-2,-1)), f_hat.detach().mean(-3,-2,-1), dim=1).mean()

def contrastive_similarity_loss(f, f_hat, eps=1e-8):
    intra_similarity = torch.nn.functional.cosine_similarity(f.mean((-3,-2,-1)), f_hat.detach().mean(-3,-2,-1), dim=1).mean()
    cross_similarity = torch.nn.functional.cosine_similarity(f.mean((-3,-2,-1)), f_hat.detach().mean(-3,-2,-1), dim=0).mean()
    return (1 - intra_similarity) / (1 - cross_similarity + eps)

def rescale(pr, gt=None):
    if gt is None:
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr

sample_types=["resize", "fragments", "crop", "arp_resize", "arp_fragments"]




def finetune_epoch(ft_loader, model, model_ema, optimizer, scheduler, device, epoch=-1):
    model.train()
    for i, data in enumerate(tqdm(ft_loader, desc=f"Training in epoch {epoch}")):
        optimizer.zero_grad()
        video = {}
        for key in sample_types:
            if key in data:
                video[key] = data[key].to(device)
        y = data["gt_label"].float().detach().to(device).unsqueeze(-1)
        frame_inds = data["frame_inds"]
        y_pred = model(video, inference=False).mean((-3, -2, -1))
        p_loss, r_loss = plcc_loss(y_pred, y), rank_loss(y_pred, y)
        loss = p_loss + 0.1 * r_loss
        wandb.log(
            {
                "train/plcc_loss": p_loss.item(),
                "train/rank_loss": r_loss.item(),
                "train/"
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

    
def profile_inference(inf_set, model, device):
    video = {}
    data = inf_set[0]
    for key in sample_types:
        if key in data:
            video[key] = data[key].to(device).unsqueeze(0)
    with torch.no_grad():
        flops, params = profile(model, (video, ))
    print(f"The FLOps of the Variant is {flops/1e9:.1f}G, with Params {params/1e6:.2f}M.")

def inference_set(inf_loader, model, device, best_, save_model=False, suffix='s', save_name="divide"):

    results = []

    best_s, best_p, best_k, best_r = best_

    for i, data in enumerate(tqdm(inf_loader, desc="Validating")):
        result = dict()
        video = {}
        for key in sample_types:
            if key in data:
                video[key] = data[key].to(device)#.unsqueeze(0)
        with torch.no_grad():
            result["pr_labels"] = model(video).cpu().numpy()
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

    wandb.log({f"val/SRCC-{suffix}": s, f"val/PLCC-{suffix}": p, f"val/KRCC-{suffix}": k, f"val/RMSE-{suffix}": r})

    if s + p > best_s + best_p and save_model:
        state_dict = model.state_dict()
        torch.save(
            {
                "state_dict": state_dict,
                "validation_results": best_,
            },
            f"pretrained_weights/{save_name}_{suffix}_dev_v0.0.pth",
        )

    best_s, best_p, best_k, best_r = (
        max(best_s, s),
        max(best_p, p),
        max(best_k, k),
        min(best_r, r),
    )

    wandb.log(
        {
            f"val/best_SRCC-{suffix}": best_s,
            f"val/best_PLCC-{suffix}": best_p,
            f"val/best_KRCC-{suffix}": best_k,
            f"val/best_RMSE-{suffix}": best_r,
        }
    )

    print(
        f"For {len(inf_loader)} videos, \nthe accuracy of the model: [{suffix}] is as follows:\n  SROCC: {s:.4f} best: {best_s:.4f} \n  PLCC:  {p:.4f} best: {best_p:.4f}  \n  KROCC: {k:.4f} best: {best_k:.4f} \n  RMSE:  {r:.4f} best: {best_r:.4f}."
    )

    return best_s, best_p, best_k, best_r

    # torch.save(results, f'{args.save_dir}/results_{dataset.lower()}_s{32}*{32}_ens{args.famount}.pkl')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--opt", type=str, default="./options/divide/add.yml", help="the option file"
    )

    args = parser.parse_args()
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    print(opt)
    
    
    

    ## adaptively choose the device

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## defining model and loading checkpoint

    bests_ = []
    
    model = getattr(models, opt["model"]["type"])(**opt["model"]["args"]).to(device)
    
    train_dataset = getattr(datasets, opt["data"]["train"]["type"])(opt["data"]["train"]["args"])
    val_dataset = getattr(datasets, opt["data"]["val"]["type"])(opt["data"]["val"]["args"])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt["batch_size"], num_workers=opt["num_workers"], shuffle=True,
    )
    
    
    val_loader =  torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=opt["num_workers"], pin_memory=True,
    )

    
    run = wandb.init(
        project=opt["wandb"]["project_name"],
        name=opt["name"],
        reinit=True,
    )

    state_dict = torch.load(opt["load_path"], map_location=device)

    if "state_dict" in state_dict:
        ### migrate training weights from mmaction
        state_dict = state_dict["state_dict"]
        from collections import OrderedDict

        i_state_dict = OrderedDict()
        for key in state_dict.keys():
            if "cls" in key:
                tkey = key.replace("cls", "vqa")
            elif "backbone" in key:
                i_state_dict["fragments_"+key] = state_dict[key]
                i_state_dict["resize_"+key] = state_dict[key]
            else:
                i_state_dict[key] = state_dict[key]
    t_state_dict = model.state_dict()
    for key, value in t_state_dict.items():
        if key in i_state_dict and i_state_dict[key].shape != value.shape:
            i_state_dict.pop(key)
    model.load_state_dict(i_state_dict, strict=False)
    
    print(model)

    if opt["ema"]:
        from copy import deepcopy
        model_ema = deepcopy(model)
    else:
        model_ema = None

    profile_inference(val_dataset, model, device)    

    # finetune the model
    print(len(val_loader), len(train_loader))
    
    
    param_groups=[]
    
    for key, value in dict(model.named_children()).items():
        if "backbone" in key:
            param_groups += [{"params": value.parameters(), "lr": opt["optimizer"]["lr"] * opt["optimizer"]["backbone_lr_mult"]}]
        else:
            param_groups += [{"params": value.parameters(), "lr": opt["optimizer"]["lr"]}]
            
    optimizer = torch.optim.AdamW(lr=opt["optimizer"]["lr"], params=param_groups,
                                  weight_decay=opt["optimizer"]["wd"],
                                 )

    warmup_iter = int(opt["warmup_epochs"] * len(train_loader))
    max_iter = int((opt["num_epochs"] + opt["l_num_epochs"]) * len(train_loader))
    lr_lambda = (
        lambda cur_iter: cur_iter / warmup_iter
        if cur_iter <= warmup_iter
        else 0.5 * (1 + math.cos(math.pi * (cur_iter - warmup_iter) / max_iter))
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_lambda,
    )

    best_ = -1, -1, -1, 1000
    best_n = best_

    for key, value in dict(model.named_children()).items():
        if "backbone" in key:
            for param in value.parameters():
                param.requires_grad = False

    for epoch in range(opt["l_num_epochs"]):
        print(f"Linear Epoch {epoch}:")
        finetune_epoch(
            train_loader, model, model_ema, optimizer, scheduler, device, epoch
        )
        best_ = inference_set(
            val_loader,
            model_ema if model_ema is not None else model,
            device, best_, save_model=opt["save_model"], save_name=opt["name"],
        )
        if model_ema is not None:
            best_n = inference_set(
                val_loader,
                model,
                device, best_n, save_model=opt["save_model"], save_name=opt["name"],
                suffix = 'n',
            )
        else:
            best_n = best_

    print(
        f"""For the linear transfer process on with {len(val_loader)} videos,
        the best validation accuracy of the model-s is as follows:
        SROCC: {best_[0]:.4f}
        PLCC:  {best_[1]:.4f}
        KROCC: {best_[2]:.4f}
        RMSE:  {best_[3]:.4f}."""
    )
    
    print(
        f"""For the linear transfer process on with {len(val_loader)} videos,
        the best validation accuracy of the model-n is as follows:
        SROCC: {best_n[0]:.4f}
        PLCC:  {best_n[1]:.4f}
        KROCC: {best_n[2]:.4f}
        RMSE:  {best_n[3]:.4f}."""
    )
    
    for key, value in dict(model.named_children()).items():
        if "backbone" in key:
            for param in value.parameters():
                param.requires_grad = True

    for epoch in range(opt["num_epochs"]):
        print(f"Finetune Epoch {epoch}:")
        finetune_epoch(
            train_loader, model, model_ema, optimizer, scheduler, device, epoch
        )
        best_ = inference_set(
            val_loader,
            model_ema if model_ema is not None else model,
            device, best_, save_model=opt["save_model"], save_name=opt["name"],
        )
        if model_ema is not None:
            best_n = inference_set(
                val_loader,
                model,
                device, best_n, save_model=opt["save_model"],
                suffix='n', save_name=opt["name"],
            )
        else:
            best_n = best_
            
    print(
        f"""For the fintuning process on with {len(val_loader)} videos,
        the best validation accuracy of the model-s is as follows:
        SROCC: {best_[0]:.4f}
        PLCC:  {best_[1]:.4f}
        KROCC: {best_[2]:.4f}
        RMSE:  {best_[3]:.4f}."""
    )
    
    print(
        f"""For the linear transfer process on with {len(val_loader)} videos,
        the best validation accuracy of the model-n is as follows:
        SROCC: {best_n[0]:.4f}
        PLCC:  {best_n[1]:.4f}
        KROCC: {best_n[2]:.4f}
        RMSE:  {best_n[3]:.4f}."""
    )

    run.finish()



if __name__ == "__main__":
    main()
