import os
import subprocess

'''
Feb 6, 2025
RVP

This script to makes movies from simulation snapshots using ffmpeg
Input: directory name with snapshots
pre-requiste: need to have the snapshots and relevant directories in the input_folder
'''

TopDir = "/Users/rahul/City College Dropbox/Rahul Pandare/CUNY/research/bidisperse_project/figures/new_data/movies/system/"
OutDir = "/Users/rahul/City College Dropbox/Rahul Pandare/CUNY/research/bidisperse_project/figures/new_data/movies/system/"

filename  = ['phi_0.795_ar_4.0_vr_0.25']

framerate = 24
codec     = 'libx264'
pix_fmt   = 'yuv420p'

for i in range(len(filename)):
    input_folder = TopDir + filename[i]
    output_file  = filename[i] + ".mp4"

    if os.path.exists(input_folder):
        if os.path.exists(os.path.join(OutDir, output_file)):
            print(f'\nMovie already exits - {output_file}\n')
        else:
            command = [
                'ffmpeg', # the operating command
                '-framerate', str(framerate), # input framerate of the movie
                '-start_number', '200',    # start of frame
                #'-pattern_type', 'glob',  # pattern of the image files to read. 'sequence' - image files have sequential naming,  'glob' - 
                '-i', os.path.join(input_folder, '%d.png'), # image file name structure
                '-r', str(framerate), # output framerate of the movie (ideally 24 or 30 fps)
                '-c:v', codec,        # compressing-decompression format (ideally H.264 for mp4)
                '-b:v', '10M',        # bitrate of the video stream- 5 megabits/s here; i.e. 5M data used to represent each second of the video
                                      # higher bitrate meaning better quality video but larger output movie size. (default is 5M) 
                '-preset', 'slow',    # specifying a preset for video encoding process. it determines the trade-off between encoding speed and compression efficiency. See below for preset options.
                '-pix_fmt', pix_fmt,  # pixel color format. options: yuv444p, rgb24, bgr24
                os.path.join(OutDir, output_file)
            ]
            try:
                subprocess.run(command, check=True)
                print(f"\nVideo created successfully: {output_file}\n")
            except subprocess.CalledProcessError as e:
                print(f"\nError during video creation: {e}\n")

###############                                
# preset options for ffmpeg encoding-

# ultrafast: This preset prioritizes encoding speed over compression efficiency. It produces videos quickly but may result in larger file sizes and lower overall quality.
# veryslow: This preset prioritizes compression efficiency over speed. It produces higher quality videos with potentially smaller file sizes, but the encoding process is slower.
#fast: This preset is a middle ground between speed and quality.
# medium: This preset is a balanced option and is often a good choice for general use.
# slow: This preset prioritizes quality over speed and may produce smaller file sizes with improved visual quality.