import os
import subprocess

'''
May 24, 2025
RVP

This script makes GIFs from simulation snapshots using ffmpeg.
Input: directory name with snapshots.
Pre-requisite: snapshots and relevant directories must exist in input_folder.
'''

TopDir = "/Users/rahul/City College Dropbox/Rahul Pandare/CUNY/research/bidisperse_project/miscelleneous/DEM_codes/figures/"
OutDir = TopDir  # Output is saved in same top directory

# Input file(s) name
filename = ['NP_100_phi_0.5_Cundall-Strack', 'NP_100_phi_0.5_Hertzian-Mindlin']

framerate = 10

for name in filename:
    input_folder = os.path.join(TopDir, name)
    output_file  = name + ".gif"
    palette_file = os.path.join(input_folder, "palette.png")  # Temporary palette

    if os.path.exists(input_folder):
        # Step 1: Generate color palette (only first 300 frames)
        palette_command = [
            'ffmpeg',
            '-y',
            '-framerate', str(framerate),
            '-i', os.path.join(input_folder, 'frame_%03d.png'),
            '-vf', 'select=\'lte(n\\,299)\',palettegen', # Generate palette from first 300 frames
            palette_file
        ]


        # Step 2: Create GIF using palette (only first 300 frames)
        gif_command = [
            'ffmpeg',
            '-y',
            '-framerate', str(framerate),
            '-i', os.path.join(input_folder, 'frame_%03d.png'),
            '-i', palette_file,
            '-lavfi', 'select=\'lte(n\\,299)\',paletteuse', # Use palette for GIF
            os.path.join(OutDir, output_file)
        ]

        try:
            subprocess.run(palette_command, check=True)
            subprocess.run(gif_command, check=True)
            print(f"\nGIF created successfully: {output_file}\n")
        except subprocess.CalledProcessError as e:
            print(f"\nError during GIF creation: {e}\n")
    else:
        print(f"\nInput folder not found: {input_folder}\n")
