# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 00:11:39 2025

@author: yzhao
"""

import os

# from moviepy import VideoFileClip

import subprocess
from imageio_ffmpeg import get_ffmpeg_exe


def make_mp4_clip(
    video_path, start_time, end_time, save_path=None, save_dir="./assets/videos/"
):
    duration = end_time - start_time
    if save_path is None:
        video_name = os.path.basename(video_path).split(".")[0]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        mp4_file = video_name + f"_time_range_{start_time}-{end_time}" + ".mp4"
        save_path = os.path.join(save_dir, mp4_file)

    ff = get_ffmpeg_exe()
    cmd = [
        ff,
        "-y",  # overwrite output if it exists
        "-ss",
        str(start_time),  # seek to start time
        "-i",
        video_path,  # input file
        "-t",
        str(duration),  # clip duration
        # "-c", "copy",  # copy all streams (no re-encode)
        "-c:v",
        "libx264",  # re-encode video to H.264
        "-movflags",
        "+faststart",  # for better MP4 playback start
        "-f",
        "mp4",  # force MP4 container
        save_path,
    ]
    subprocess.run(cmd, check=True)


"""
def avi_to_mp4(
    avi_path, start_time, end_time, save_path=None, save_dir="./assets/videos/"
):
    if save_path is None:
        avi_name = os.path.basename(avi_path).split(".")[0]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        mp4_file = avi_name + f"_time_range_{start_time}-{end_time}" + ".mp4"
        save_path = os.path.join(save_dir, mp4_file)

    clip = (
        VideoFileClip(avi_path)
        .subclipped(start_time=start_time, end_time=end_time)
        .without_audio()
    )
    clip.write_videofile(save_path, audio=False, logger=None)
    clip.close()
    # return save_path, mp4_file
"""

if __name__ == "__main__":
    video_path = "C:/Users/yzhao/python_projects/sleep_scoring/user_test_files/35_ymaze_ymaze_Cam2.avi"
    # avi_to_mp4(avi_path, start_time=1000000, end_time=1000000 + 80)
    make_mp4_clip(video_path, start_time=849, end_time=871)
