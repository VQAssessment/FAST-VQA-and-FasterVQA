import torch
import cv2
import random
from models import BaseEvaluator
from datasets import VQAInferenceDataset, get_fragments

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


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['LIVE_VQC', 'LSVQ', 'KoNViD', 'CVD2014'], default='LIVE_VQC', help='the inference dataset name')
    parser.add_argument('--pdpath', type=str, default='../datasets/', help='the inference dataset name')
    
    args = parser.parse_args()

    ## adaptively choose the device
    

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    ## defining model and loading checkpoint

    model = BaseEvaluator().to(device)
    load_path = 'pretrained_weights/all_aligned_fragments.pth'
    state_dict = torch.load(load_path, map_location='cpu')

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
        from collections import OrderedDict
        i_state_dict = OrderedDict()
        for key in state_dict.keys():
            if 'cls' in key:
                tkey = key.replace('cls', 'vqa')
                i_state_dict[tkey] = state_dict[key]
            else:
                i_state_dict[key] = state_dict[key]

    model.load_state_dict(i_state_dict)

    ## getting datasets (if you want to load from existing VQA datasets)

    dataset_name = args.dataset
    dataset_path = f'{args.pdpath}/{dataset_name}'

    inference_set = VQAInferenceDataset(f'examplar_data_labels/{dataset_name}/labels.txt', dataset_path)
    
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
    
    #pickle.dump(results)
    

if __name__ == '__main__':
    main()
