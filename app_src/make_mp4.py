# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 00:11:39 2025

@author: yzhao
"""

import os

from moviepy import VideoFileClip


def avi_to_mp4(avi_path, start_time, end_time, save_path=None, save_dir="./assets/videos/"):
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
    #return save_path, mp4_file


if __name__ == "__main__":
    avi_path = "C:/Users/yzhao/matlab_projects/sleep_data/20220914_788_FP_unscored/20220914_788_FP_2022-09-14_13-53-27-322_video_0.avi"
    avi_to_mp4(avi_path, start_time=1000000, end_time=1000000 + 80)
