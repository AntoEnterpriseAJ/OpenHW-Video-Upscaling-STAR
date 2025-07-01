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

    if not os.path.exists('analysis/' + output_file_name):
        os.makedirs('analysis/' + output_file_name)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_path_1 = f"analysis/{output_file_name}/{current_time}_psnr_{output_file_name}.txt"
    output_file_path_2 = f"analysis/{output_file_name}/{current_time}_vmaf_{output_file_name}"
    output_file_path_3 = f"analysis/{output_file_name}/{current_time}_ssim_{output_file_name}.txt"
    output_file_path_4 = f"analysis/{output_file_name}/{current_time}_lpips_{output_file_name}.txt"

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

def get_videos_list(directory):
    """
    Get a list of video files in the specified directory.
    
    :param directory: Directory to search for video files.
    :return: List of video file paths.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    video_extensions = ('.mp4', '.avi', '.mkv', '.mov')
    # Ensure the list only contains the file names with the specified video extensions
    return [
        os.path.splitext(f)[0]
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(video_extensions)
    ]

if __name__ == "__main__":
    # Example usage
    output_name = "patient_1_L2"
    distorted_file = f"input/video/{output_name}.mp4"
    reference_file = f"results/{output_name}_20250701_133046.mp4"

    input_list = get_videos_list('input/video')
    input_list = sorted(input_list, key=lambda x: x.lower())
    print("Input video files:")
    for file in input_list:
        print(file)

    output_list = get_videos_list('results')
    for output_file in output_list:
        of = output_file[:-16]  # Remove the timestamp part from the filename
        if of not in input_list:
            print(f"Warning: Output file {output_file} does not have a corresponding input file in 'input/video' directory. Removing it from the output list.")
            output_list.remove(output_file)

    output_list = sorted(output_list, key=lambda x: x.lower())

    print("Output video files:")
    for file in output_list:
        print(file)

    try:
        for index in range(len(input_list)):
            distorted_file = f"input/video/{input_list[index]}.mp4"
            reference_file = f"results/{output_list[index]}.mp4"
            output_name = input_list[index]

            width, height = get_video_resolution(reference_file)
            rezolution = f"{width}:{height}"

            analyze_quality(distorted_file, reference_file, output_name, rezolution)
    except Exception as e:
        print(f"Error: {e}")