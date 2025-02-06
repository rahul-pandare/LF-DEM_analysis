# Script to makes movies from simulation snapshots using ffmpeg
# user input: phi
# pre-requiste: need to have the snapshots and relevant directories in the output_path

# command to run in terminal: python -c "from makeMovie import makeMovie; makeMovie(phi)"

import os
import subprocess

NP_array  = [1000]

ar_array = [1.0]

run_dict = {500:8, 1000:1, 2000:2, 4000:1}

TopDir = "/Users/rahul/Downloads"
OutDir= "/Users/rahul/Downloads/"

allFiles=["clusters","frictParts", "interactions", "superposition"]

def makeMovie(phi,framerate=24, codec='libx264', pix_fmt='yuv420p'):
    
    for j in range(len(NP_array)):
        NP = NP_array[j]
        for l in range(len(ar_array)):
            ar = ar_array[l]
            for m in range(run_dict[NP_array[j]]):
                run = m+1;
                for n in range(len(allFiles)):
                    output_file = "NP_" + str(NP) + "_phi_0." + str(int(phi*100)) + "_ar_" + str(ar) + "_Vr_0.5_run_" + str(run) +".mp4"
                    input_folder= TopDir + "/NP_" + str(NP) + "/phi_0." + str(int(phi*100)) + "/ar_" + str(ar) + "/Vr_0.5/run_" + str(run) +"/snapshots/"+allFiles[n]
                    output_path = OutDir + allFiles[n]

                    if os.path.exists(input_folder):
                        if os.path.exists(os.path.join(output_path, output_file)):
                            print(f'\nMovie already exits - {output_file}\n')
                        else:
                            command = [
                                'ffmpeg', # the operating command
                                '-framerate', str(framerate), # input framerate of the movie
                                '-pattern_type', 'sequence', # pattern of the image files to read. Sequence - image files have sequential naming
                                '-i', os.path.join(input_folder, '%d.svg'), # image file name structure
                                '-r', str(framerate), # output framerate of the movie (ideally 24 or 30 fps)
                                '-c:v', codec, # compressing-decompression format (ideally H.264 for mp4)
                                '-b:v', '5M', # bitrate of the video stream- 5 megabits/s here; i.e. 5M data used to represent each second of the video
                                              # higher bitrate meaning better quality video but larger output movie size. (default is 5M) 
                                '-preset', 'medium', # specifying a preset for video encoding process. it determines the trade-off between encoding speed and compression efficiency. See below for preset options.
                                '-pix_fmt', pix_fmt, # pixel color format. options: yuv444p, rgb24, bgr24
                                os.path.join(output_path, output_file)
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