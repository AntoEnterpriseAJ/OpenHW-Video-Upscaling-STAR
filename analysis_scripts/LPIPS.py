import argparse
import cv2
import torch
import lpips
import os
import numpy as np

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB and normalize to [0,1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        frames.append(frame)
    cap.release()
    return frames

def preprocess_frame(frame):
    # frame shape: H x W x 3, values in [0,1]
    tensor = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0)  # 1 x 3 x H x W
    tensor = tensor * 2 - 1  # scale to [-1, 1]
    return tensor.float()

def resize_frame(frame, target_size):
    # Resize frame to target size (H, W)
    resized_frame = cv2.resize(frame, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    return resized_frame

def resize_frames(frames, target_size):
    resized_frames = [resize_frame(frame, target_size) for frame in frames]
    return resized_frames

def get_video_resolution(video_path):
    """
    Get the resolution of a video file.
    
    :param video_path: Path to the video file.
    :return: Tuple (width, height) of the video resolution.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return width, height

def width_height_to_resolution(width, height):
    """
    Convert width and height to a resolution tuple.
    
    :param width: Width of the video.
    :param height: Height of the video.
    :return: Tuple (height, width) representing the resolution.
    """
    return (height, width)

def compute_lpips_video(distorted_path, reference_path):
    # Load LPIPS model (alexnet backbone recommended)
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn.eval()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()

    frames_dist = extract_frames(distorted_path)
    frames_ref = extract_frames(reference_path)

    width, height = get_video_resolution(distorted_path)
    target_size = width_height_to_resolution(width, height)
    frames_dist = resize_frames(frames_dist, target_size)
    frames_ref = resize_frames(frames_ref, target_size)

    if len(frames_dist) != len(frames_ref):
        print(f"Warning: Number of frames differ! Distorted: {len(frames_dist)}, Reference: {len(frames_ref)}")
        min_len = min(len(frames_dist), len(frames_ref))
        frames_dist = frames_dist[:min_len]
        frames_ref = frames_ref[:min_len]

    lpips_scores = []
    for i, (f_dist, f_ref) in enumerate(zip(frames_dist, frames_ref)):
        f_dist_t = preprocess_frame(f_dist)
        f_ref_t = preprocess_frame(f_ref)
        if torch.cuda.is_available():
            f_dist_t = f_dist_t.cuda()
            f_ref_t = f_ref_t.cuda()
        with torch.no_grad():
            dist = loss_fn(f_dist_t, f_ref_t)
        lpips_scores.append(dist.item())
        print(f"Frame {i+1}/{len(frames_dist)} LPIPS: {dist.item():.4f}")

    avg_lpips = np.mean(lpips_scores)
    return avg_lpips

def get_lpips(distorted_file_path, reference_file_path, output_file_name):
    avg_lpips = compute_lpips_video(distorted_file_path, reference_file_path)
    print(f"\nAverage LPIPS over video: {avg_lpips:.4f}")

    with open(output_file_name, "w") as f:
        f.write(f"Average LPIPS: {avg_lpips:.4f}\n")