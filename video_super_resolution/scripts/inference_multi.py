# scripts/inference_multi.py
from inference import STAR

star = STAR(model_path="/opt/openhardware/pretrained_weights/light_deg.pt",
            solver_mode="fast", steps=15, guide_scale=7.5,
            upscale=2, max_chunk_len=12)

jobs = [
    ("./input/video/patient_1_L2.mp4", "This is a grayscale ultrasound video of a patient’s lung. You can see the thin, bright pleural line gently sliding with each breath, faint horizontal A-line echoes beneath it, and occasional vertical B-line streaks extending down from the pleura."),
    ("./input/video/patient_1_L3.mp4", "This is a grayscale ultrasound video of a patient’s lung. You can see the thin, bright pleural line gently sliding with each breath, faint horizontal A-line echoes beneath it, and occasional vertical B-line streaks extending down from the pleura."),
]
for path, prompt in jobs:
    out = star.enhance_a_video(path, prompt)
    print("saved:", out)
