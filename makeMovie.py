import os
import subprocess

'''
Feb 6, 2025
RVP

This script makes movies from simulation snapshots using ffmpeg.
Input: directory name with snapshots.
Pre-requisite: snapshots and relevant directories must exist in input_folder.
'''

TopDir = "/Users/rahul/City College Dropbox/Rahul Pandare/CUNY/research/bidisperse_project/figures/ang_vel/"
OutDir = "/Users/rahul/City College Dropbox/Rahul Pandare/CUNY/research/bidisperse_project/figures/ang_vel/"

# Input file(s) name
filename  = ['phi_0.77_ar_1.4_vr_0.5_angV_transV2'] 
# Example: ['phi_0.77_ar_2.0_vr_0.5_int', 'phi_0.765_ar_1.4_vr_0.5_int', 'phi_0.795_ar_4.0_vr_0.5_int']

framerate = 8
codec     = 'libx264'
pix_fmt   = 'yuv420p'

for i in range(len(filename)):
    input_folder = TopDir + filename[i]
    output_file  = filename[i] + ".mp4"

    if os.path.exists(input_folder):
        if os.path.exists(os.path.join(OutDir, output_file)):
            print(f'\nMovie already exists - {output_file}\n')
        else:
            frame_numbers = sorted([int(f.split('.')[0]) for f in os.listdir(input_folder) if f.endswith('.png')])
            start_number = frame_numbers[0] if frame_numbers else 0

            command = [
                'ffmpeg',                            # command to invoke ffmpeg
                '-framerate', str(framerate),         # input framerate of the movie
                '-start_number', str(start_number),   # first frame number to start with
                '-i', os.path.join(input_folder, '%d.png'),  # pattern for input image files (numbered)
                '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # scaling filter: force width and height divisible by 2
                '-r', str(framerate),                 # output framerate (usually match input framerate)
                '-c:v', codec,                        # codec for encoding the video stream (libx264 for mp4)
                '-b:v', '15M',                        # video bitrate (controls quality vs file size)
                '-preset', 'slow',                    # encoding preset (speed vs compression trade-off)
                '-pix_fmt', pix_fmt,                  # pixel format (yuv420p for compatibility)
                os.path.join(OutDir, output_file)     # output movie file path
            ]

            try:
                subprocess.run(command, check=True)
                print(f"\nVideo created successfully: {output_file}\n")
            except subprocess.CalledProcessError as e:
                print(f"\nError during video creation: {e}\n")
    else:
        print(f"\nInput folder not found: {input_folder}\n")

###############                                
# preset options for ffmpeg encoding:

# ultrafast: Prioritizes encoding speed (larger files, lower quality).
# veryslow: Prioritizes compression efficiency (smaller files, higher quality, slower encoding).
# fast: Good balance between speed and quality.
# medium: Default preset; reasonable compromise between speed and quality.
# slow: Better compression with slower encoding (used in this script).