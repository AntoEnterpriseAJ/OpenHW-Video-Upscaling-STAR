import os
import os.path as osp
import random
from typing import Any, Dict
import gc

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F

from video_to_video.modules import *
from video_to_video.utils.config import cfg
from video_to_video.diffusion.diffusion_sdedit import GaussianDiffusion
from video_to_video.diffusion.schedules_sdedit import noise_schedule
from video_to_video.utils.logger import get_logger

from diffusers import AutoencoderKLTemporalDecoder

from hip import roctx

logger = get_logger()

class VideoToVideo_sr():
    def __init__(self, opt, device=torch.device(f'cuda:0')):
        self.opt = opt
        self.device = device # torch.device(f'cuda:0')

        # text_encoder
        text_encoder = FrozenOpenCLIPEmbedder(device=self.device, pretrained="laion2b_s32b_b79k")
        text_encoder.model.to(self.device)
        self.text_encoder = text_encoder
        logger.info(f'Build encoder with FrozenOpenCLIPEmbedder')

        # U-Net with ControlNet
        generator = ControlledV2VUNet()
        generator = generator.to(self.device)
        generator.eval()

        cfg.model_path = opt.model_path
        load_dict = torch.load(cfg.model_path, map_location='cpu')
        if 'state_dict' in load_dict:
            load_dict = load_dict['state_dict']
        ret = generator.load_state_dict(load_dict, strict=False)
        
        self.generator = generator.half()
        logger.info('Load model path {}, with local status {}'.format(cfg.model_path, ret))

        # Noise scheduler
        sigmas = noise_schedule(
            schedule='logsnr_cosine_interp',
            n=1000,
            zero_terminal_snr=True,
            scale_min=2.0,
            scale_max=4.0)
        diffusion = GaussianDiffusion(sigmas=sigmas)
        self.diffusion = diffusion
        logger.info('Build diffusion with GaussianDiffusion')

        # Temporal VAE
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="vae", variant="fp16"
        )
        vae.eval()
        vae.requires_grad_(False)
        vae.to(self.device)
        self.vae = vae
        logger.info('Build Temporal VAE')

        torch.cuda.empty_cache()

        self.negative_prompt = cfg.negative_prompt
        self.positive_prompt = cfg.positive_prompt

        negative_y = text_encoder(self.negative_prompt).detach()
        self.negative_y = negative_y


    def test(self, input: Dict[str, Any], total_noise_levels=1000, \
                 steps=50, solver_mode='fast', guide_scale=7.5, max_chunk_len=32):
        roctx.range_push("VideoToVideo_sr::test")
        try:
            video_data = input['video_data']
            y = input['y']
            (target_h, target_w) = input['target_res']

            roctx.range_push("upsample+pad")
            video_data = F.interpolate(video_data, [target_h,target_w], mode='bilinear')
            logger.info(f'video_data shape: {video_data.shape}')
            frames_num, _, h, w = video_data.shape
            padding = pad_to_fit(h, w)
            video_data = F.pad(video_data, padding, 'constant', 1)
            roctx.range_pop()

            video_data = video_data.unsqueeze(0)
            bs = 1
            video_data = video_data.to(self.device)

            roctx.range_push("VAE.encode")
            video_data_feature = self.vae_encode(video_data)
            roctx.range_pop()
            torch.cuda.empty_cache()

            y = self.text_encoder(y).detach()

            with amp.autocast(enabled=True):
                t = torch.LongTensor([total_noise_levels-1]).to(self.device)
                noised_lr = self.diffusion.diffuse(video_data_feature, t)

                model_kwargs = [{'y': y}, {'y': self.negative_y}]
                model_kwargs.append({'hint': video_data_feature})

                chunk_inds = make_chunks(frames_num, interp_f_num=0, max_chunk_len=max_chunk_len) if frames_num > max_chunk_len else None
                solver = 'dpmpp_2m_sde'

                roctx.range_push(f"Diffusion.sample_sr[solver={solver}, steps={steps}]")
                gen_vid = self.diffusion.sample_sr(
                    noise=noised_lr,
                    model=self.generator,
                    model_kwargs=model_kwargs,
                    guide_scale=guide_scale,
                    guide_rescale=0.2,
                    solver=solver,
                    solver_mode=solver_mode,
                    return_intermediate=None,
                    steps=steps,
                    t_max=total_noise_levels - 1,
                    t_min=0,
                    discretization='trailing',
                    chunk_inds=chunk_inds,)
                roctx.range_pop()

                gen_vid = gen_vid.cpu()
                del noised_lr, video_data_feature
                model_kwargs.clear()
                torch.cuda.empty_cache(); gc.collect()

                roctx.range_push("VAE.decode_chunk")
                vid_tensor_gen = self.vae_decode_chunk(gen_vid, chunk_size=1)
                roctx.range_pop()

            roctx.range_push("crop+rearrange")
            w1, w2, h1, h2 = padding
            vid_tensor_gen = vid_tensor_gen[:,:,h1:h+h1,w1:w+w1]
            gen_video = rearrange(vid_tensor_gen, '(b f) c h w -> b c f h w', b=bs)
            roctx.range_pop()

            torch.cuda.empty_cache()
            return gen_video.type(torch.float32).cpu()
        finally:
            roctx.range_pop()
            
    def temporal_vae_decode(self, z, num_f):
        return self.vae.decode(z/self.vae.config.scaling_factor, num_frames=num_f).sample

    def vae_decode_chunk(self, z, chunk_size: int = 3):
        z = rearrange(z, "b c f h w -> (b f) c h w").half()  # ensure fp16
        video_cpu = []
        for start in range(0, z.shape[0], chunk_size):
            chunk = z[start:start + chunk_size].to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                num_f = chunk.shape[0]
                rgb = self.temporal_vae_decode(chunk, num_f)

            video_cpu.append(rgb.detach().cpu()) # off‑load result
            del chunk, rgb
            torch.cuda.empty_cache() # free up GPU memory

        return torch.cat(video_cpu, dim=0) # lives on CPU

    def vae_encode(self, t, chunk_size=1):
        num_f = t.shape[1]
        t = rearrange(t, "b f c h w -> (b f) c h w")
        z_list = []
        for ind in range(0,t.shape[0],chunk_size):
            z_list.append(self.vae.encode(t[ind:ind+chunk_size]).latent_dist.sample())
        z = torch.cat(z_list, dim=0)
        z = rearrange(z, "(b f) c h w -> b c f h w", f=num_f)
        return z * self.vae.config.scaling_factor
    

def pad_to_fit(h, w):
    BEST_H, BEST_W = 720, 1280

    if h < BEST_H:
        h1, h2 = _create_pad(h, BEST_H)
    elif h == BEST_H:
        h1 = h2 = 0
    else: 
        h1 = 0
        h2 = int((h + 48) // 64 * 64) + 64 - 48 - h

    if w < BEST_W:
        w1, w2 = _create_pad(w, BEST_W)
    elif w == BEST_W:
        w1 = w2 = 0
    else:
        w1 = 0
        w2 = int(w // 64 * 64) + 64 - w
    return (w1, w2, h1, h2)

def _create_pad(h, max_len):
    h1 = int((max_len - h) // 2)
    h2 = max_len - h1 - h
    return h1, h2


def make_chunks(f_num, interp_f_num, max_chunk_len, chunk_overlap_ratio=0.5):
    MAX_CHUNK_LEN = max_chunk_len
    MAX_O_LEN = MAX_CHUNK_LEN * chunk_overlap_ratio
    chunk_len = int((MAX_CHUNK_LEN-1)//(1+interp_f_num)*(interp_f_num+1)+1)
    o_len = int((MAX_O_LEN-1)//(1+interp_f_num)*(interp_f_num+1)+1)
    chunk_inds = sliding_windows_1d(f_num, chunk_len, o_len)
    return chunk_inds


def sliding_windows_1d(length, window_size, overlap_size):
    stride = window_size - overlap_size
    ind = 0
    coords = []
    while ind<length:
        if ind+window_size*1.25>=length:
            coords.append((ind,length))
            break
        else:
            coords.append((ind,ind+window_size))
            ind += stride  
    return coords
