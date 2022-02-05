## loading example video
import decord
from decord import VideoReader
from decord import cpu, gpu
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm

import random
random.seed(42)

decord.bridge.set_bridge('torch')

def get_fragments(video, fragments=7, fsize=32, aligned=32, random=False):
    size = fragments * fsize
    if min(video.shape[-2:]) < size:
        ovideo = video
        video = torch.nn.functional.interpolate(video / 255., scale_factor = size / min(video.shape[-2:]), mode='bilinear')
        video = (video * 255.).type_as(ovideo)
    dur_t, res_h, res_w = video.shape[-3:]
    assert dur_t % aligned == 0, 'Please provide match vclip and align index'
    size = (fragments * fsize, fragments * fsize)
    
    hgrids = torch.LongTensor([res_h // fragments * i for i in range(fragments)])
    wgrids = torch.LongTensor([res_w // fragments * i for i in range(fragments)])
    hlength, wlength = res_h // fragments, res_w // fragments
    
    if random:
        if res_h > fsize:
            rnd_h = torch.randint(res_h - fsize, (len(hgrids), len(wgrids), dur_t // aligned))
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if res_w > fsize:
            rnd_w = torch.randint(res_w - fsize, (len(hgrids), len(wgrids), dur_t // aligned))
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
    else:
        if hlength > fsize:
            rnd_h = torch.randint(hlength - fsize, (len(hgrids), len(wgrids), dur_t // aligned))
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if wlength > fsize:
            rnd_w = torch.randint(wlength - fsize, (len(hgrids), len(wgrids), dur_t // aligned))
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
    
    target_video = torch.zeros(video.shape[:-2] + size).to(video.device)
    #target_videos = []
        
    for i, hs in enumerate(hgrids):
        for j, ws in enumerate(wgrids):
            for t in range(dur_t // aligned):
                t_s, t_e = t * aligned, (t+1) * aligned
                h_s, h_e = i * fsize, (i+1) * fsize
                w_s, w_e = j * fsize, (j+1) * fsize
                if random:
                    h_so, h_eo = rnd_h[i][j][t], rnd_h[i][j][t] + fsize
                    w_so, w_eo = rnd_w[i][j][t], rnd_w[i][j][t] + fsize
                else:
                    h_so, h_eo = hs + rnd_h[i][j][t], hs + rnd_h[i][j][t] + fsize
                    w_so, w_eo = ws + rnd_w[i][j][t], ws + rnd_w[i][j][t] + fsize
                target_video[:,t_s:t_e,h_s:h_e,w_s:w_e] = video[:,t_s:t_e,h_so:h_eo,w_so:w_eo]
    #target_videos.append(video[:,t_s:t_e,h_so:h_eo,w_so:w_eo])
    #target_video = torch.stack(target_videos, 0).reshape((dur_t // aligned, fragments, fragments,) + target_videos[0].shape).permute(3,0,4,1,5,2,6)
    #target_video = target_video.reshape((-1, dur_t,) + size) ## Splicing Fragments
    return target_video

class SampleFrames:
    def __init__(self, clip_len, frame_interval=1, num_clips=1):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        
    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips)
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(
                    num_frames - ori_clip_len + 1, size=self.num_clips))
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)

        return clip_offsets

    def _get_test_clips(self, num_frames, start_index=0):
        """Get clip offsets in test mode.

        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int32)
        else:
            clip_offsets = np.zeros((self.num_clips, ), dtype=np.int32)
        return clip_offsets


    def __call__(self, total_frames, train=False, start_index=0):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if train:
            clip_offsets = self._get_train_clips(total_frames)
        else:
            clip_offsets = self._get_test_clips(total_frames)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        frame_inds = np.mod(frame_inds, total_frames)
        frame_inds = np.concatenate(frame_inds) + start_index
        return frame_inds.astype(np.int32)


class VQAInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, ann_file, data_prefix, clip_len=32, frame_interval=2, num_clips=4, aligned=32, fragments=7, fsize=32, nfrags=8, cache_in_memory=False, phase='test'):
        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.fragments = fragments
        self.fsize = fsize
        self.nfrags = nfrags
        self.aligned = aligned
        self.sampler = SampleFrames(clip_len, frame_interval, num_clips)
        self.video_infos = []
        self.phase = phase
        self.mean=torch.FloatTensor([123.675, 116.28, 103.53])
        self.std=torch.FloatTensor([58.395, 57.12, 57.375])
        if isinstance(self.ann_file, list):
            self.video_infos = self.ann_file
        else:
            with open(self.ann_file, 'r') as fin:
                for line in fin:
                    line_split = line.strip().split(',')
                    filename, _, _, label = line_split
                    label = float(label)
                    filename = osp.join(self.data_prefix, filename)
                    self.video_infos.append(dict(filename=filename, label=label))
        if cache_in_memory:
            self.cache = {}
            for i in tqdm(range(len(self)), desc='Caching fragments'):
                self.cache[i] = self.__getitem__(i, tocache=True)
        else:
            self.cache = None
                
    def __getitem__(self, index, fragments=-1, fsize=-1, tocache=False):
        if tocache or self.cache is None:
            if fragments == -1:
                fragments = self.fragments
            if fsize == -1:
                fsize = self.fsize
            video_info = self.video_infos[index]
            filename = video_info['filename']
            label = video_info['label']
            vreader = VideoReader(filename)
            frame_inds = self.sampler(len(vreader), self.phase == 'train')
            frame_dict = {
                idx: vreader[idx]
                for idx in np.unique(frame_inds)
            }
            imgs = [frame_dict[idx] for idx in frame_inds]
            img_shape = imgs[0].shape
            video = torch.stack(imgs, 0)
            video = video.permute(3, 0, 1, 2)
            if self.nfrags == 1:
                vfrag = get_fragments(video, fragments, fsize, aligned=self.aligned)
            else: 
                vfrag = get_fragments(video, fragments, fsize, aligned=self.aligned)
                for i in range(1, self.nfrags):
                    vfrag = torch.cat((vfrag, get_fragments(video, fragments, fsize, aligned=self.aligned)), 1)
            if tocache:
                return (vfrag, frame_inds, label, img_shape)
        else:
            vfrag, frame_inds, label, img_shape = self.cache[index]
        vfrag = ((vfrag.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3,0,1,2)
        return {'video': vfrag.reshape((-1, self.nfrags * self.num_clips, self.clip_len) + vfrag.shape[2:]).transpose(0,1), ## B, V, T, C, H, W
                'frame_inds': frame_inds,
                'gt_label': label,
                'original_shape': img_shape,
               }

    def __len__(self):
        return len(self.video_infos)