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


def rescale(pr, gt=None):
    if gt is None:
        print(np.mean(pr), np.std(pr))
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        print(np.mean(pr), np.std(pr), np.std(gt), np.mean(gt))
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr

sample_types=["resize", "fragments", "crop", "arp_resize", "arp_fragments"]

def profile_inference(inf_set, model, device):
    video = {}
    data = inf_set[0]
    for key in sample_types:
        if key in data:
            video[key] = data[key].to(device)
            c, t, h, w = video[key].shape
            video[key] = video[key].reshape(1, c, data["num_clips"][key], t // data["num_clips"][key], h, w).permute(0,2,1,3,4,5).reshape( data["num_clips"][key], c, t // data["num_clips"][key], h, w) 
    with torch.no_grad():
        flops, params = profile(model, (video, ))
    print(f"The FLOps of the Variant is {flops/1e9:.1f}G, with Params {params/1e6:.2f}M.")

def inference_set(inf_loader, model, device, best_, save_model=False, suffix='s'):

    results = []

    best_s, best_p, best_k, best_r = best_

    for i, data in enumerate(tqdm(inf_loader, desc="Validating")):
        result = dict()
        video = {}
        for key in sample_types:
            if key in data:
                video[key] = data[key].to(device)
                b, c, t, h, w = video[key].shape
                video[key] = video[key].reshape(b, c, data["num_clips"][key], t // data["num_clips"][key], h, w).permute(0,2,1,3,4,5).reshape(b * data["num_clips"][key], c, t // data["num_clips"][key], h, w) 
        with torch.no_grad():
            labels = model(video,reduce_scores=False)
            labels = [np.mean(l.cpu().numpy()) for l in labels]
            result["pr_labels"] = labels
        result["gt_label"] = data["gt_label"].item()
        result["name"] = data["name"]
        # result['frame_inds'] = data['frame_inds']
        # del data
        results.append(result)

    
    ## generate the demo video for video quality localization
    gt_labels = [r["gt_label"] for r in results]
    pr_labels = 0
    weights = [0.3, 0.7]
    for i, w in zip(range(len(results[0]["pr_labels"])), weights):
        print(i, w)
        pr_labels += rescale([np.mean(r["pr_labels"][i]) for r in results]) * w
        
    pr_labels = rescale(pr_labels, gt_labels)

    s = spearmanr(gt_labels, pr_labels)[0]
    p = pearsonr(gt_labels, pr_labels)[0]
    k = kendallr(gt_labels, pr_labels)[0]
    r = np.sqrt(((gt_labels - pr_labels) ** 2).mean())
    
    
    results = sorted(results, key=lambda x: x["pr_labels"])

    wandb.log({f"val/SRCC-{suffix}": s, f"val/PLCC-{suffix}": p, f"val/KRCC-{suffix}": k, f"val/RMSE-{suffix}": r})

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

    return best_s, best_p, best_k, best_r, pr_labels

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--opt", type=str, default="./options/fast/fast-b.yml", help="the option file"
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

    state_dict = torch.load(opt["test_load_path"], map_location=device)["state_dict"]
    
    if "test_load_path_aux" in opt:
        new_state_dict = torch.load(opt["test_load_path_aux"], map_location=device)["state_dict"]
        
        from collections import OrderedDict
        
        fusion_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("vqa_head"):
                ki = k.replace("vqa", "fragments")
            else:
                ki = k
            fusion_state_dict[ki] = v
            
        for k, v in new_state_dict.items():
            if k.startswith("vqa_head"):
                ki = k.replace("vqa", "resize")
            else:
                ki = k
            fusion_state_dict[ki] = v
        
        state_dict = fusion_state_dict

    model.load_state_dict(state_dict, strict=True)
    
    
    for key in opt["data"].keys():
        
        if "val" not in key and "test" not in key:
            continue
        
        run = wandb.init(
            project=opt["wandb"]["project_name"],
            name=opt["name"]+"_Test_"+key,
            reinit=True,
        )
        
        val_dataset = getattr(datasets, opt["data"][key]["type"])(opt["data"][key]["args"])


        val_loader =  torch.utils.data.DataLoader(
            val_dataset, batch_size=1, num_workers=opt["num_workers"], pin_memory=True,
        )



        profile_inference(val_dataset, model, device)    

        # test the model
        print(len(val_loader))

        best_ = -1, -1, -1, 1000


        best_ = inference_set(
            val_loader,
            model,
            device, best_,
        )

        print(
            f"""Testing result on: [{len(val_loader)}] videos:
            SROCC: {best_[0]:.4f}
            PLCC:  {best_[1]:.4f}
            KROCC: {best_[2]:.4f}
            RMSE:  {best_[3]:.4f}."""
        )
        
        with open("results/"+opt["name"]+"_Test_"+key+".txt", "w") as f:
            for label in best_[-1]:
                f.write(f"{label}\n")

        run.finish()



if __name__ == "__main__":
    main()
