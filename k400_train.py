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

from functools import reduce
from thop import profile
import copy

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


def ce_loss(y_pred, y):
    return torch.nn.functional.cross_entropy(y_pred, y.long().flatten().detach())


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

def self_similarity_loss(f, f_hat, f_hat_detach=False):
    if f_hat_detach:
        f_hat = f_hat.detach()
    return 1 - torch.nn.functional.cosine_similarity(f, f_hat, dim=1).mean()

def contrastive_similarity_loss(f, f_hat, f_hat_detach=False, eps=1e-8):
    if f_hat_detach:
        f_hat = f_hat.detach()
    intra_similarity = torch.nn.functional.cosine_similarity(f, f_hat, dim=1).mean()
    cross_similarity = torch.nn.functional.cosine_similarity(f, f_hat, dim=0).mean()
    return (1 - intra_similarity) / (1 - cross_similarity + eps)

def rescale(pr, gt=None):
    if gt is None:
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr

sample_types=["resize", "fragments", "crop", "arp_resize", "arp_fragments"]




def finetune_epoch(ft_loader, model, model_ema, optimizer, scheduler, device, epoch=-1, 
                   need_upsampled=False, need_feat=False, need_fused=False, need_separate_sup=False):
    model.train()
    for i, data in enumerate(tqdm(ft_loader, desc=f"Training in epoch {epoch}")):
        optimizer.zero_grad()
        video = {}
        for key in sample_types:
            if key in data:
                video[key] = data[key].to(device)
        

        
        y = data["gt_label"].float().detach().to(device).unsqueeze(-1)
        scores = model(video, inference=False,
                            reduce_scores=False) 
        if len(scores) > 1:
            y_pred = reduce(lambda x,y:x+y, scores)
        else:
            y_pred = scores[0]
        y_pred = y_pred.mean((-3, -2, -1))
        
        frame_inds = data["frame_inds"]
        # Plain Supervised Loss
        loss = ce_loss(y_pred, y)

        wandb.log({"train/total_loss": loss.item(),})

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        #ft_loader.dataset.refresh_hypers()

        
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


    best_m, best_t1, best_t5 = best_
    
    confusion_matrix = torch.zeros(400,400)
    t5_confusion_matrix = torch.zeros(400,400)
    
 
    for i, data in enumerate(tqdm(inf_loader, desc="Validating")):
        result = dict()
        video = {}
        for key in sample_types:
            if key in data:
                video[key] = data[key].to(device)
                ## Reshape into clips
                b, c, t, h, w = video[key].shape
                video[key] = video[key].reshape(b, c, data["num_clips"], t // data["num_clips"], h, w).permute(0,2,1,3,4,5).reshape(b * data["num_clips"], c, t // data["num_clips"], h, w) 
        with torch.no_grad():
            y = model(video).mean((-3,-2,-1)).cpu().numpy()
            y_sort = np.argsort(y).flatten()[::-1]
            y = y.argmax()
         
        confusion_matrix[y][data["gt_label"]] += 1
        for i in y_sort[:5]:
            t5_confusion_matrix[i][data["gt_label"]] += 1
        del video
        # result['frame_inds'] = data['frame_inds']
        # del data
        
    t1 = torch.sum(torch.diag(confusion_matrix)) / torch.sum(confusion_matrix)
    t5 = torch.sum(torch.diag(t5_confusion_matrix)) / torch.sum(confusion_matrix)
    m = (torch.diag(confusion_matrix) / torch.sum(confusion_matrix, 1)).mean()
    
    
    

    wandb.log({f"val_{suffix}/MAcc-{suffix}": m, f"val_{suffix}/T1Acc-{suffix}": t1, f"val_{suffix}/T5Acc-{suffix}": t5,})
    
    torch.cuda.empty_cache()
    


    if t1 > best_t1 and save_model:
        state_dict = model.state_dict()
        torch.save(
            {
                "state_dict": state_dict,
                "validation_results": best_,
            },
            f"pretrained_weights/{save_name}_{suffix}_dev_v0.0.pth",
        )
        
    best_t5 = max(t5, best_t5)
    best_t1 = max(t1, best_t1)
    best_m = max(m, best_m)


    print(
        f"For {len(inf_loader)} videos, \nthe accuracy of the model: [{suffix}] is as follows:\n  MAcc: {m:.4f} best: {best_m:.4f} \n  T1Acc:  {t1:.4f} best: {best_t1:.4f}  \n  T5Acc: {t5:.4f} best: {best_t5:.4f}."
    )

    return best_m, best_t1, best_t5

    # torch.save(results, f'{args.save_dir}/results_{dataset.lower()}_s{32}*{32}_ens{args.famount}.pkl')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--opt", type=str, default="./options/divide/mradd.yml", help="the option file"
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
    
    if opt.get("split_seed", -1) > 0:
        num_splits = 10
    else:
        num_splits = 1
        
    for split in range(num_splits):
        
        if opt.get("split_seed", -1) > 0:
            split_duo = train_test_split(opt["data"]["train"]["args"]["data_prefix"],
                                         opt["data"]["train"]["args"]["anno_file"], 
                                         seed=opt["split_seed"] * split)
            opt["data"]["train"]["args"]["anno_file"], opt["data"]["val"]["args"]["anno_file"] = split_duo

        train_datasets = {}
        for key in opt["data"]:
            if key.startswith("train"):
                train_dataset = getattr(datasets, opt["data"][key]["type"])(opt["data"][key]["args"])
                train_datasets[key] = train_dataset
        
        train_loaders = {}
        for key, train_dataset in train_datasets.items():
            train_loaders[key] = torch.utils.data.DataLoader(
                train_dataset, batch_size=opt["batch_size"], num_workers=opt["num_workers"], shuffle=True,
            )
        
        val_datasets = {}
        for key in opt["data"]:
            if key.startswith("val"):
                val_datasets[key] = getattr(datasets, 
                                            opt["data"][key]["type"])(opt["data"][key]["args"])


        val_loaders = {}
        for key, val_dataset in val_datasets.items():
            val_loaders[key] = torch.utils.data.DataLoader(
                val_dataset, batch_size=1, num_workers=opt["num_workers"], pin_memory=True,
            )


        run = wandb.init(
            project=opt["wandb"]["project_name"],
            name=opt["name"]+f'_{split}' if num_splits > 1 else opt["name"],
            reinit=True,
        )
            
        #print(model)

        if opt["ema"]:
            from copy import deepcopy
            model_ema = deepcopy(model)
        else:
            model_ema = None

        #profile_inference(val_dataset, model, device)    

        # finetune the model
        param_groups=[]

        for key, value in dict(model.named_children()).items():
            if "backbone" in key:
                param_groups += [{"params": value.parameters(), "lr": opt["optimizer"]["lr"] * opt["optimizer"]["backbone_lr_mult"]}]
            else:
                param_groups += [{"params": value.parameters(), "lr": opt["optimizer"]["lr"]}]

        optimizer = torch.optim.AdamW(lr=opt["optimizer"]["lr"], params=param_groups,
                                      weight_decay=opt["optimizer"]["wd"],
                                     )
        warmup_iter = 0
        for train_loader in train_loaders.values():
            warmup_iter += int(opt["warmup_epochs"] * len(train_loader))
        max_iter = int((opt["num_epochs"] + opt["l_num_epochs"]) * len(train_loader))
        lr_lambda = (
            lambda cur_iter: cur_iter / warmup_iter
            if cur_iter <= warmup_iter
            else 0.5 * (1 + math.cos(math.pi * (cur_iter - warmup_iter) / max_iter))
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda,
        )

        bests = {}
        bests_n = {}
        for key in val_loaders:
            bests[key] = 0,0,0
            bests_n[key] = 0,0,0
        

        for key, value in dict(model.named_children()).items():
            if "backbone" in key:
                for param in value.parameters():
                    param.requires_grad = False

        for epoch in range(opt["l_num_epochs"]):
            print(f"Linear Epoch {epoch}:")
            for key, train_loader in train_loaders.items():
                finetune_epoch(
                    train_loader, model, model_ema, optimizer, scheduler, device, epoch,
                    opt.get("need_upsampled", False), opt.get("need_feat", False), opt.get("need_fused", False),
                )
            for key in val_loaders:
                bests[key] = inference_set(
                    val_loaders[key],
                    model_ema if model_ema is not None else model,
                    device, bests[key], save_model=opt["save_model"], save_name=opt["name"],
                    suffix = key+"_s",
                )
                if model_ema is not None:
                    bests_n[key] = inference_set(
                        val_loaders[key],
                        model,
                        device, bests_n[key], save_model=opt["save_model"], save_name=opt["name"],
                        suffix = key+'_n',
                    )
                else:
                    bests_n[key] = bests[key]

        if opt["l_num_epochs"] >= 0:
            for key in val_loaders:
                print(
                    f"""For the linear transfer process on {key} with {len(val_loaders[key])} videos,
                    the best validation accuracy of the model-s is as follows:
                    SROCC: {bests[key][0]:.4f}
                    PLCC:  {bests[key][1]:.4f}
                    KROCC: {bests[key][2]:.4f}."""
                )

                print(
                    f"""For the linear transfer process on {key} with {len(val_loaders[key])} videos,
                    the best validation accuracy of the model-n is as follows:
                    SROCC: {bests_n[key][0]:.4f}
                    PLCC:  {bests_n[key][1]:.4f}
                    KROCC: {bests_n[key][2]:.4f}."""
                )

        for key, value in dict(model.named_children()).items():
            if "backbone" in key:
                for param in value.parameters():
                    param.requires_grad = True
                    
        

        #best_ = inference_set(
        #    val_loader,
        #    model_ema if model_ema is not None else model,
        #    device, best_, save_model=False, save_name=opt["name"],
        #)
        
        
        for epoch in range(opt["num_epochs"]):
            print(f"Finetune Epoch {epoch}:")



            for key, train_loader in train_loaders.items():
                finetune_epoch(
                    train_loader, model, model_ema, optimizer, scheduler, device, epoch,
                    opt.get("need_upsampled", False), opt.get("need_feat", False), opt.get("need_fused", False),
                )
            for key in val_loaders:
                bests[key] = inference_set(
                    val_loaders[key],
                    model_ema if model_ema is not None else model,
                    device, bests[key], save_model=opt["save_model"], save_name=opt["name"],
                    suffix=key+"_s",
                )
                if model_ema is not None:
                    bests_n[key] = inference_set(
                        val_loaders[key],
                        model,
                        device, bests_n[key], save_model=opt["save_model"], save_name=opt["name"],
                        suffix = key+'_n',
                    )
                else:
                    bests_n[key] = bests[key]
                    
        if opt["num_epochs"] > 0:
            for key in val_loaders:
                print(
                    f"""For the finetuning process on {key} with {len(val_loaders[key])} videos,
                    the best validation accuracy of the model-s is as follows:
                    SROCC: {bests[key][0]:.4f}
                    PLCC:  {bests[key][1]:.4f}
                    KROCC: {bests[key][2]:.4f}
                    RMSE:  {bests[key][3]:.4f}."""
                )

                print(
                    f"""For the finetuning process on {key} with {len(val_loaders[key])} videos,
                    the best validation accuracy of the model-n is as follows:
                    SROCC: {bests_n[key][0]:.4f}
                    PLCC:  {bests_n[key][1]:.4f}
                    KROCC: {bests_n[key][2]:.4f}
                    RMSE:  {bests_n[key][3]:.4f}."""
                )
            
        run.finish()
    
    



if __name__ == "__main__":
    main()
