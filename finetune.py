import torch
import cv2
import random
import os.path as osp
from fastvqa.models import BaseEvaluator
from fastvqa.datasets import VQAInferenceDataset, get_fragments

import argparse

from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
import numpy as np

from time import time
from tqdm import tqdm
import pickle

def rank_loss(y_pred, y):
    ranking_loss = torch.nn.functional.relu((y_pred-y_pred.t()) * torch.sign((y.t()-y)))
    scale = 1 + torch.max(ranking_loss)
    return (torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0]-1) / scale).float()


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
    with open(ann_file, 'r') as fin:
        for line in fin:
            line_split = line.strip().split(',')
            filename, _, _, label = line_split
            label = float(label)
            filename = osp.join(dataset_path, filename)
            video_infos.append(dict(filename=filename, label=label))
    random.shuffle(video_infos)
    return video_infos[:int(ratio * len(video_infos))], video_infos[int(ratio * len(video_infos)):]

def rescale(pr, gt=None):
    if gt is None:
        pr = ((pr - np.mean(pr)) / np.std(pr))
    else:
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr

all_datasets = ['LIVE_VQC', 'KoNViD', 'CVD2014']

def generate_dataset(args, dataset, seed=42):
    
    print(f'Predicting video quality on dataset: {dataset}.')
    
    ## getting datasets (if you want to load from existing VQA datasets)
    dataset_name = dataset
    dataset_path = f'{args.pdpath}/{dataset_name}'
    
    train_infos, val_infos = train_test_split(dataset_path, f'examplar_data_labels/{dataset_name}/labels.txt', seed=seed)

    finetune_set = VQAInferenceDataset(train_infos,
                                        dataset_path,
                                        num_clips = 1,
                                        fsize = args.fsize, 
                                        fragments = 224 // args.fsize, 
                                        nfrags = args.famount,
                                        cache_in_memory = args.cache,
                                        phase='train',
                                       )
    
    validation_set = VQAInferenceDataset(val_infos,
                                        dataset_path,
                                        num_clips = 4,
                                        fsize = args.fsize, 
                                        fragments = 224 // args.fsize, 
                                        nfrags = args.famount,
                                        cache_in_memory = args.cache,
                                       )
    
    
    print(f'Fine-tuning on Dataset {args.dataset} in {dataset_path}.')
    
    return finetune_set, validation_set

def generate_train_test_loader(args, seed=42):
   
    dataset = args.dataset
    
    ft_set, val_set = generate_dataset(args, dataset, seed=seed)
    
    print(len(ft_set), len(val_set))
    
    ft_loader = torch.utils.data.DataLoader(ft_set, batch_size=8, num_workers=6, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, num_workers=6, pin_memory=True)
    
    return ft_loader, val_loader
    
    
def finetune_epoch(ft_loader, model, optimizer, device):
    model.train()
    for i, data in enumerate(tqdm(ft_loader, desc='Training')):  
        optimizer.zero_grad()
        vfrag = data['video'].to(device).squeeze(1)
        y = data['gt_label'].float().detach().to(device).unsqueeze(-1)
        frame_inds = data['frame_inds']
        y_pred = model(vfrag, inference=False).mean(-1).mean(-1).mean(-1)
        loss = plcc_loss(y_pred, y) + 0.1 * rank_loss(y_pred, y)
        loss.backward()
        optimizer.step()
        
    

def inference_set(inf_loader, model, device, best_):
    
    results = []
    
    best_s, best_p, best_k, best_r = best_

    for i, data in enumerate(tqdm(inf_loader, desc='Validating')):
        result = dict()
        vfrag = data['video'].to(device).squeeze(0)
        with torch.no_grad():
            result['pr_labels'] = model(vfrag).cpu().numpy()
        result['gt_label'] = data['gt_label'].item()
        #result['frame_inds'] = data['frame_inds']
        #del data
        results.append(result)
        
    ## generate the demo video for video quality localization
    gt_labels = [r['gt_label'] for r in results]
    pr_labels = [np.mean(r['pr_labels'][:]) for r in results]
    pr_labels = rescale(pr_labels, gt_labels)

    s = spearmanr(gt_labels, pr_labels)[0]
    p = pearsonr(gt_labels, pr_labels)[0]
    k = kendallr(gt_labels, pr_labels)[0]
    r = np.sqrt(((gt_labels - pr_labels) ** 2).mean())
    
    best_s, best_p, best_k, best_r = max(best_s, s), max(best_p, p), max(best_k, k), min(best_r, r)
    

    print(f'For {len(inf_loader)} videos, \nthe accuracy of the model is as follows:\n  SROCC: {s:.4f} best: {best_s:.4f} \n  PLCC:  {p:.4f} best: {best_p:.4f}  \n  KROCC: {k:.4f} best: {best_k:.4f} \n  RMSE:  {r:.4f} best: {best_r:.4f}.')
    
    return best_s, best_p, best_k, best_r
    
    #torch.save(results, f'{args.save_dir}/results_{dataset.lower()}_s{args.fsize}*{args.fsize}_ens{args.famount}.pkl')


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', choices=['LIVE_VQC', 'KoNViD', 'CVD2014'], default='LIVE_VQC', help='the finetune dataset name')
    parser.add_argument('--pdpath', type=str, default='../datasets/', help='the inference dataset name')
    parser.add_argument('-s', '--fsize', choices=[8, 16, 32], default=32, help='size of fragment strips')
    parser.add_argument('-a', '--famount', type=int, default=1, help='sample amount of fragment strips')
    parser.add_argument('-lep', '--l_num_epochs', type=int, default=10, help='linear finetune epochs')
    parser.add_argument('-ep', '--num_epochs', type=int, default=20, help='finetune epochs')
    parser.add_argument('--save_dir', type=str, default='results', help='results_dir')
    parser.add_argument('-c', '--cache', action='store_true', help='use_cache_dataset')
    parser.add_argument('-var', '--from_ar', action='store_true', help='use_features_from_action_recognition')


    
    args = parser.parse_args()

    ## adaptively choose the device
    

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    ## defining model and loading checkpoint

    
    bests_ = []
    
    torch.save({'results': bests_}, f'{args.save_dir}/results_finetune_{args.dataset.lower()}_s{args.fsize}*{args.fsize}_ens{args.famount}{"" if not args.from_ar else "_from_ar"}.pkl')
    
    for i in range(10):
        model = BaseEvaluator().to(device)

        if args.from_ar:
            load_path = '../model_baselines/NetArch/swin_tiny_patch244_window877_kinetics400_1k.pth'
        else:
            if args.fsize != 32:
                raise NotImplementedError('Version 0.2.0 only supports 32*32 finetune on fragments.')
            load_path = f'pretrained_weights/fast_vqa_v0_3.pth'
        state_dict = torch.load(load_path, map_location='cpu')

        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            from collections import OrderedDict
            i_state_dict = OrderedDict()
            for key in state_dict.keys():
                if 'cls' in key:
                    tkey = key.replace('cls', 'vqa')
                    if args.l_num_epochs == 0 and not args.from_ar:
                        i_state_dict[tkey] = state_dict[key]
                else:
                    i_state_dict[key] = state_dict[key]

        model.load_state_dict(i_state_dict, strict=False)

        ft_loader, val_loader = generate_train_test_loader(args, seed=42*(i+1))

        ## finetune the model
        print(len(ft_loader), len(val_loader))
        optimizer = torch.optim.AdamW(lr=1e-4, params=model.parameters())

        best_ = -1, -1, -1, 1000
        best_ = inference_set(val_loader, model, device, best_)

        print(f'Before the finetune process on {args.dataset} with {len(val_loader)} videos, \nthe accuracy of the model is as follows:\n  SROCC: {best_[0]:.4f}\n  PLCC:  {best_[1]:.4f}\n  KROCC: {best_[2]:.4f}\n  RMSE:  {best_[3]:.4f}.')

        for param in model.backbone.parameters():
            param.requires_grad = False

        for epoch in range(args.l_num_epochs):
            print(f'Split {i}, Linear Epoch {epoch}:')
            finetune_epoch(ft_loader, model, optimizer, device)
            best_ = inference_set(val_loader, model, device, best_)

        print(f'For the linear transfer process on {args.dataset} with {len(val_loader)} videos, \nthe best validation accuracy of the model is as follows:\n  SROCC: {best_[0]:.4f}\n  PLCC:  {best_[1]:.4f}\n  KROCC: {best_[2]:.4f}\n  RMSE:  {best_[3]:.4f}.')

        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(args.num_epochs):
            print(f'Split {i}, Finetune Epoch {epoch}:')
            finetune_epoch(ft_loader, model, optimizer, device)
            best_ = inference_set(val_loader, model, device, best_)

        print(f'For the finetune process on {args.dataset} with {len(val_loader)} videos, \nthe best validation accuracy of the model is as follows:\n  SROCC: {best_[0]:.4f}\n  PLCC:  {best_[1]:.4f}\n  KROCC: {best_[2]:.4f}\n  RMSE:  {best_[3]:.4f}.')
    
        bests_.append(best_)
        del model
        
        
    torch.save({'results': bests_}, f'{args.save_dir}/results_finetune_{args.dataset.lower()}_s{args.fsize}*{args.fsize}_ens{args.famount}{"" if not args.from_ar else "_from_ar"}.pkl')
    

if __name__ == '__main__':
    main()


    
    
    