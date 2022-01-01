## loading example video
import decord
from decord import VideoReader
from decord import cpu, gpu
import os.path as osp
import numpy as np
import torch
decord.bridge.set_bridge('torch')

def get_fragments(video, fragments, fsize=32, aligned=32):
    
    dur_t, res_h, res_w = video.shape[-3:]
    assert dur_t % aligned == 0, 'Please provide match vclip and align index'
    size = (fragments * fsize, fragments * fsize)
    
    hgrids = torch.LongTensor([res_h // fragments * i for i in range(fragments)])
    wgrids = torch.LongTensor([res_w // fragments * i for i in range(fragments)])
    hlength, wlength = res_h // fragments, res_w // fragments
    
    if hlength > fsize:
        rnd_h = torch.randint(hlength - fsize, (len(hgrids), len(wgrids), dur_t // aligned))
    else:
        rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
    if wlength > fsize:
        rnd_w = torch.randint(wlength - fsize, (len(hgrids), len(wgrids), dur_t // aligned))
    else:
        rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
    
    target_video = torch.zeros(video.shape[:-2] + size).to(video.device)
        
    for i, hs in enumerate(hgrids):
        for j, ws in enumerate(wgrids):
            for t in range(dur_t // aligned):
                t_s, t_e = t * aligned, (t+1) * aligned
                h_s, h_e = i * fsize, (i+1) * fsize
                w_s, w_e = j * fsize, (j+1) * fsize
                h_so, h_eo = hs + rnd_h[i][j][t], hs + rnd_h[i][j][t] + fsize
                w_so, w_eo = ws + rnd_w[i][j][t], ws + rnd_w[i][j][t] + fsize
                target_video[:,t_s:t_e,h_s:h_e,w_s:w_e] = video[:,t_s:t_e,h_so:h_eo,w_so:w_eo]
    return target_video

class SampleFrames:
    def __init__(self, clip_len, frame_interval=1, num_clips=1):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips

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


    def __call__(self, total_frames, start_index=0):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        clip_offsets = self._get_test_clips(total_frames)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        frame_inds = np.mod(frame_inds, total_frames)
        frame_inds = np.concatenate(frame_inds) + start_index
        return frame_inds.astype(np.int32)


class VQAInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, ann_file, data_prefix, clip_len=32, frame_interval=2, num_clips=4, aligned=32):
        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.sampler = SampleFrames(clip_len, frame_interval, num_clips)
        self.video_infos = []
        self.mean=torch.Tensor([123.675, 116.28, 103.53])
        self.std=torch.Tensor([58.395, 57.12, 57.375])
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split(',')
                filename, _, _, label = line_split
                label = float(label)
                filename = osp.join(self.data_prefix, filename)
                self.video_infos.append(dict(filename=filename, label=label))
                
    def __getitem__(self, index):
        video_info = self.video_infos[index]
        filename = video_info['filename']
        label = video_info['label']
        vreader = VideoReader(filename)
        frame_inds = self.sampler(len(vreader))
        frame_dict = {
            idx: vreader[idx]
            for idx in np.unique(frame_inds)
        }
        imgs = [frame_dict[idx] for idx in frame_inds]
        video = torch.stack(imgs, 0)
        video = ((video - self.mean) / self.std).permute(3, 0, 1, 2)
        vfrag = get_fragments(video, 7)
        return {'video': vfrag.reshape((-1, self.num_clips, self.clip_len) + vfrag.shape[2:]).transpose(0,1), 
                'frame_inds': frame_inds,
                'gt_label': label,
               }

    def __len__(self):
        return len(self.video_infos)