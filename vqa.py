
import yaml
import decord
from fastvqa.datasets import get_spatial_fragments, SampleFrames, FragmentSampleFrames
from fastvqa.models import DiViDeAddEvaluator
import torch
import numpy as np
import argparse

def sigmoid_rescale(score, model="FasterVQA"):
    mean, std = mean_stds[model]
    x = (score - mean) / std
    print(f"Inferring with model [{model}]:")
    score = 1 / (1 + np.exp(-x))
    return score

mean_stds = {
    "FasterVQA": (0.14759505, 0.03613452), 
    "FasterVQA-MS": (0.15218826, 0.03230298),
    "FasterVQA-MT": (0.14699507, 0.036453716),
    "FAST-VQA":  (-0.110198185, 0.04178565),
    "FAST-VQA-M": (0.023889644, 0.030781006), 
}

opts = {
    "FasterVQA": "./options/fast/f3dvqa-b.yml", 
    "FasterVQA-MS": "./options/fast/fastervqa-ms.yml", 
    "FasterVQA-MT": "./options/fast/fastervqa-mt.yml", 
    "FAST-VQA": "./options/fast/fast-b.yml", 
    "FAST-VQA-M": "./options/fast/fast-m.yml", 
}

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    ### can choose between
    ### options/fast/f3dvqa-b.yml
    ### options/fast/fast-b.yml
    ### options/fast/fast-m.yml
    
    parser.add_argument(
        "-m", "--model", type=str, 
        default="FasterVQA", 
        help="model type: can choose between FasterVQA, FasterVQA-MS, FasterVQA-MT, FAST-VQA, FAST-VQA-M",
    )
    
    ## can be your own
    parser.add_argument(
        "-v", "--video_path", type=str, 
        default="./demos/10053703034.mp4", 
        help="the input video path"
    )
    
    parser.add_argument(
        "-d", "--device", type=str, 
        default="cuda", 
        help="the running device"
    )
    
    
    args = parser.parse_args()

    video_reader = decord.VideoReader(args.video_path)
    
    opt = opts.get(args.model, opts["FAST-VQA"])
    with open(opt, "r") as f:
        opt = yaml.safe_load(f)

    ### Model Definition
    evaluator = DiViDeAddEvaluator(**opt["model"]["args"]).to(args.device)
    evaluator.load_state_dict(torch.load(opt["test_load_path"], map_location=args.device)["state_dict"])

    ### Data Definition
    vsamples = {}
    t_data_opt = opt["data"]["val-kv1k"]["args"]
    s_data_opt = opt["data"]["val-kv1k"]["args"]["sample_types"]
    for sample_type, sample_args in s_data_opt.items():
        ## Sample Temporally
        if t_data_opt.get("t_frag",1) > 1:
            sampler = FragmentSampleFrames(fsize_t=sample_args["clip_len"] // sample_args.get("t_frag",1),
                                           fragments_t=sample_args.get("t_frag",1),
                                           num_clips=sample_args.get("num_clips",1),
                                          )
        else:
            sampler = SampleFrames(clip_len = sample_args["clip_len"], num_clips = sample_args["num_clips"])
        
        num_clips = sample_args.get("num_clips",1)
        frames = sampler(len(video_reader))
        print("Sampled frames are", frames)
        frame_dict = {idx: video_reader[idx] for idx in np.unique(frames)}
        imgs = [frame_dict[idx] for idx in frames]
        video = torch.stack(imgs, 0)
        video = video.permute(3, 0, 1, 2)

        ## Sample Spatially
        sampled_video = get_spatial_fragments(video, **sample_args)
        mean, std = torch.FloatTensor([123.675, 116.28, 103.53]), torch.FloatTensor([58.395, 57.12, 57.375])
        sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
        
        sampled_video = sampled_video.reshape(sampled_video.shape[0], num_clips, -1, *sampled_video.shape[2:]).transpose(0,1)
        vsamples[sample_type] = sampled_video.to(args.device)
        print(sampled_video.shape)
    result = evaluator(vsamples)
    score = sigmoid_rescale(result.mean().item(), model=args.model)
    print(f"The quality score of the video (range [0,1]) is {score:.5f}.")
