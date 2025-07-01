import subprocess
import os
import datetime
from analysis_scripts.LPIPS import get_lpips

def get_video_resolution(video_file_path):
    """
    Get the resolution of a video file using ffprobe.
    
    :param video_file_path: Path to the video file.
    :return: A tuple containing width and height of the video.
    """
    if not os.path.exists(video_file_path):
        raise FileNotFoundError(f"Video file not found: {video_file_path}")

    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'csv=p=0',
        video_file_path
    ]

    try:
        output = subprocess.check_output(command).decode('utf-8').strip()
        width, height = map(int, output.split(','))
        return width, height
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get video resolution: {e}")

def analyze_quality(distorted_file_path, reference_file_path, output_file_name, rezolution=(1280, 960)):
    """
    Analyze the quality of a distorted audio file against a reference file using ffmpeg.
    
    :param distorted_file_path: Path to the distorted audio file.
    :param reference_file_path: Path to the reference audio file.
    :param output_file_path: Path where the analysis report will be saved.
    """

    if not os.path.exists(distorted_file_path):
        raise FileNotFoundError(f"Distorted file not found: {distorted_file_path}")
    if not os.path.exists(reference_file_path):
        raise FileNotFoundError(f"Reference file not found: {reference_file_path}")
    if not output_file_name:
        raise ValueError("Output file name must be provided.")

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_path_1 = f"analysis/{current_time}_psnr_{output_file_name}.txt"
    output_file_path_2 = f"analysis/{current_time}_vmaf_{output_file_name}"
    output_file_path_3 = f"analysis/{current_time}_ssim_{output_file_name}.txt"
    output_file_path_4 = f"analysis/{current_time}_lpips_{output_file_name}.txt"

    command1 = [
        'ffmpeg', '-i', distorted_file_path, '-i', reference_file_path,
        '-lavfi', '[0:v]scale=' + rezolution + ':flags=bicubic[scaled];[scaled][1:v]' +
        'psnr=stats_file=' + output_file_path_1, '-f', 'null', '-'
    ]

    command2 = [
        'ffmpeg', '-i', distorted_file_path, '-i', reference_file_path,
        '-lavfi', '[0:v]scale=' + rezolution + ':flags=bicubic[scaled];[scaled][1:v]' +
        'libvmaf=log_path=' + output_file_path_2, '-f', 'null', '-'
    ]
    
    command3 = [
        'ffmpeg', '-i', distorted_file_path, '-i', reference_file_path,
        '-lavfi', '[0:v]scale=' + rezolution + ':flags=bicubic[scaled];[scaled][1:v]' +
        'ssim=stats_file=' + output_file_path_3, '-f', 'null', '-'
    ]
    
    try:
        subprocess.run(command1, check=True)
        print(f"PSNR analysis completed. Report saved to {output_file_path_1}")
        subprocess.run(command2, check=True)
        print(f"VMAF analysis completed. Report saved to {output_file_path_2}")
        subprocess.run(command3, check=True)
        print(f"SSIM analysis completed. Report saved to {output_file_path_3}")
        get_lpips(distorted_file_path, reference_file_path, output_file_path_4)
        print(f"LPIPS analysis completed. Report saved to {output_file_path_4}")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred during quality analysis: {e}")



if __name__ == "__main__":
    # Example usage
    distorted_file = "input/video/023_klingai_reedit.mp4"
    reference_file = "results/023_klingai_reedit_20250627_124443.mp4"
    output_name = "023_klingai_reedit"

    width, height = get_video_resolution(reference_file)
    rezolution = f"{width}:{height}"

    try:
        analyze_quality(distorted_file, reference_file, output_name, rezolution)
    except Exception as e:
        print(f"Error: {e}")