
import yaml
import decord
from fastvqa.datasets import get_spatial_fragments, SampleFrames, FragmentSampleFrames
from fastvqa.models import DiViDeAddEvaluator
import torch
import numpy as np
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    ### can choose between
    ### options/fast/f3dvqa-b.yml
    ### options/fast/fast-b.yml
    ### options/fast/fast-m.yml
    parser.add_argument(
        "-o", "--opt", type=str, 
        default="./options/fast/f3dvqa-b.yml", 
        help="the option file"
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


    with open(args.opt, "r") as f:
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
            sampler = FragmentSampleFrames(fsize_t=t_data_opt["clip_len"] // t_data_opt.get("t_frag",1),
                                           fragments_t=t_data_opt.get("t_frag",1),
                                           num_clips=t_data_opt.get("num_clips",1),
                                          )
        else:
            sampler = SampleFrames(clip_len = t_data_opt["clip_len"], num_clips = t_data_opt["num_clips"])
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
        num_clips = t_data_opt.get("num_clips",1)
        sampled_video = sampled_video.reshape(sampled_video.shape[0], num_clips, -1, *sampled_video.shape[2:]).transpose(0,1)
        vsamples[sample_type] = sampled_video.to(args.device)
    
    result = evaluator(vsamples)
    print(f"The quality score of the video is {result.mean().item():.5f}.")
