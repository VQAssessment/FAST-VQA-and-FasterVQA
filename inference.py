import torch
import cv2
import random
from fastvqa.models import BaseEvaluator
from fastvqa.datasets import VQAInferenceDataset, get_fragments

import argparse

from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
import numpy as np

from time import time
from tqdm import tqdm
import pickle

def rescale(pr, gt=None):
    if gt is None:
        pr = ((pr - np.mean(pr)) / np.std(pr))
    else:
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr

all_datasets = ['LIVE_VQC', 'KoNViD', 'CVD2014', 'LSVQ']

def predict_dataset(args, dataset, model, device):
    
    print(f'Predicting video quality on dataset: {dataset}.')
    
    ## getting datasets (if you want to load from existing VQA datasets)
    dataset_name = dataset
    dataset_path = f'{args.pdpath}/{dataset_name}'

    inference_set = VQAInferenceDataset(f'examplar_data_labels/{dataset_name}/labels.txt',
                                        dataset_path,
                                        fsize = args.fsize, 
                                        fragments = 224 // args.fsize, 
                                        nfrags = args.famount,
                                        cache_in_memory = args.cache)
    
    print(f'Inference on Dataset {args.dataset} in {dataset_path}')

    ## run inference for a whole testing database
    ## and get the accuracy for this database

    inference_loader = torch.utils.data.DataLoader(inference_set, batch_size=1, num_workers=6)
    results = []

    for i, data in enumerate(tqdm(inference_loader)):
        result = dict()
        vfrag = data['video'].to(device).squeeze(0)
        with torch.no_grad():
            result['pr_labels'] = model(vfrag).cpu().numpy()
        result['gt_label'] = data['gt_label'].item()
        result['frame_inds'] = data['frame_inds']
        del data
        results.append(result)
        
    ## generate the demo video for video quality localization
    gt_labels = [r['gt_label'] for r in results]
    pr_labels = [np.mean(r['pr_labels'][:]) for r in results]
    pr_labels = rescale(pr_labels, gt_labels)

    srocc = spearmanr(gt_labels, pr_labels)[0]
    plcc = pearsonr(gt_labels, pr_labels)[0]
    krocc = kendallr(gt_labels, pr_labels)[0]
    rmse = np.sqrt(((gt_labels - pr_labels) ** 2).mean())

    print(f'For dataset {dataset_name} with {len(inference_set)} videos, \nthe accuracy of the model is as follows:\n  SROCC: {srocc:.4f}\n  PLCC:  {plcc:.4f}\n  KROCC: {krocc:.4f}\n  RMSE:  {rmse:.4f}.')
    
    torch.save(results, f'{args.save_dir}/results_{dataset.lower()}_s{args.fsize}*{args.fsize}_ens{args.famount}.pkl')


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', choices=['LIVE_VQC', 'LSVQ', 'KoNViD', 'CVD2014', 'all'], default='LIVE_VQC', help='the inference dataset name')
    parser.add_argument('--pdpath', type=str, default='../datasets/', help='the inference dataset name')
    parser.add_argument('-s', '--fsize', choices=[8, 16, 32], default=32, help='size of fragment strips')
    parser.add_argument('-a', '--famount', type=int, default=1, help='sample amount of fragment strips')
    parser.add_argument('--save_dir', type=str, default='results', help='results_dir')
    parser.add_argument('--cache', action='store_true', help='use_cache_dataset')
    
    args = parser.parse_args()

    ## adaptively choose the device
    

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    ## defining model and loading checkpoint

    model = BaseEvaluator().to(device)
    if args.fsize != 32:
        raise NotImplementedError('Version 0.2.0 does not support fragment size other than 32.')
    load_path = f'pretrained_weights/fast_vqa_v0_3.pth'
    state_dict = torch.load(load_path, map_location='cpu')

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
        from collections import OrderedDict
        i_state_dict = OrderedDict()
        for key in state_dict.keys():
            print(key)
            if 'cls' in key:
                tkey = key.replace('cls', 'vqa')
                i_state_dict[tkey] = state_dict[key]
            else:
                i_state_dict[key] = state_dict[key]

    model.load_state_dict(i_state_dict)

    if args.dataset  == 'all':
        for dataset in all_datasets:
            predict_dataset(args, dataset, model, device)
    else:
        predict_dataset(args, args.dataset, model, device)
        
        

    

if __name__ == '__main__':
    main()

