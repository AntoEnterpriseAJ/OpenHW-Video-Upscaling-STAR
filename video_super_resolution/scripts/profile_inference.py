import os, sys, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference import STAR
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler, record_function
import torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", required=True)
    p.add_argument("--model_path", default="/opt/openhardware/pretrained_weights/light_deg.pt")
    p.add_argument("--prompt", default="a good video")
    p.add_argument("--save_dir", default="results")
    p.add_argument("--file_name", default="prof_out.mp4")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--solver_mode", default="fast")
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--upscale", type=int, default=2)
    p.add_argument("--max_chunk_len", type=int, default=12)
    p.add_argument("--log_dir", default="logs/pt-prof")
    args = p.parse_args()

    star = STAR(
        result_dir=args.save_dir,
        file_name=args.file_name,
        model_path=args.model_path,
        solver_mode=args.solver_mode,
        steps=args.steps,
        guide_scale=args.cfg,
        upscale=args.upscale,
        max_chunk_len=args.max_chunk_len,
    )

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    os.makedirs(args.log_dir, exist_ok=True)
    handler = tensorboard_trace_handler(args.log_dir, worker_name="rx7900xtx", use_gzip=True)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_modules=True,
        on_trace_ready=handler,
    ) as prof:
        with record_function("STAR.enhance_a_video"):
            star.enhance_a_video(args.input_path, args.prompt)
        torch.cuda.synchronize()

    print("\n=== Top ops by Self CUDA Time ===")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
    print("\n=== Top ops by CUDA Time (inclusive) ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

if __name__ == "__main__":
    main()
