# Simplified overview
import sys
import os
import argparse

# Add RAFT module path
raft_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "analysis_model", "RAFT", "core"))
sys.path.append(raft_path)

# Now import RAFT
from raft import RAFT

from utils.utils import InputPadder
import lpips, torch, cv2
import numpy as np
from torchvision import transforms

def load_models(args):

    # Load RAFT
    raft_model = RAFT(args)
    raft_model.load_state_dict(torch.load("analysis_model/RAFT/models/raft-things.pth"))
    raft_model = raft_model.eval().cuda()

    # Load LPIPS
    lpips_model = lpips.LPIPS(net='alex').cuda()
    return raft_model, lpips_model

def tensorize(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transforms.ToTensor()(frame).unsqueeze(0).cuda()
    tensor = tensor * 2 - 1  # LPIPS expects [-1,1]
    return tensor

def warp(img, flow):
    B, C, H, W = img.shape
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2).float().cuda()
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
    flow = flow.permute(0, 2, 3, 1)
    grid = grid + flow
    grid[..., 0] = 2 * grid[..., 0] / (W - 1) - 1
    grid[..., 1] = 2 * grid[..., 1] / (H - 1) - 1
    return torch.nn.functional.grid_sample(img, grid, align_corners=True)

def compute_e_star_warp(video_ref, video_dist):
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    cap_ref = cv2.VideoCapture(video_ref)
    cap_dist = cv2.VideoCapture(video_dist)
    raft, lpips_model = load_models(args)

    total_score = 0
    count = 0

    while True:
        ret1, fr1 = cap_ref.read()
        ret2, fr2 = cap_dist.read()
        if not ret1 or not ret2:
            break

        fr1 = cv2.resize(fr1, (640, 360))
        fr2 = cv2.resize(fr2, (640, 360))

        ref_tensor = tensorize(fr1)
        dist_tensor = tensorize(fr2)

        padder = InputPadder(ref_tensor.shape)
        ref_tensor, dist_tensor = padder.pad(ref_tensor, dist_tensor)

        with torch.no_grad():
            _, flow = raft(ref_tensor, dist_tensor, iters=20, test_mode=True)
            warped_ref = warp(ref_tensor, flow)
            score = lpips_model(warped_ref, dist_tensor)

        total_score += score.item()
        count += 1
        print(f"Frame {count} E*-warp score: {score.item():.4f}")

    cap_ref.release()
    cap_dist.release()
    final_score = total_score / count
    print(f"\nFinal E*-warp (avg LPIPS on warped frames): {final_score:.4f}")
    return final_score

if __name__ == "__main__":
    video_ref = "../input/video/023_kingai_reedit.mp4"
    video_dist = "../results/023_klingai_reedit_20250627_124443.mp4"
    compute_e_star_warp(video_ref, video_dist)