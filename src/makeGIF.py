import os
import subprocess

"""
03 Oct 2025
RVP

High-quality GIF creation from simulation snapshots using ffmpeg.
- Uses palettegen + paletteuse to avoid color shifts (fixes yellowing text)
- Automatically scales/pads frames to uniform size
- Works with numbered frames (N.png, N+1.png, ...). Do the math and mention num_frames accordingly.
"""

TopDir = "/Users/rahul/City College Dropbox/Rahul Pandare/CUNY/gitHub/LF_DEM_analysis/"
OutDir = TopDir  # GIF output directory

# Input folder names (relative to TopDir)
folders = ['phi_0.76_ar_1.4_vr_0.5_angV_transV2']

framerate = 8
start_num = 700  # first frame number
num_frames = 151  # total frames: 850 - 700 + 1

for name in folders:
    input_folder = os.path.join(TopDir, name)
    output_file  = os.path.join(OutDir, f"{name}.gif")
    palette_file = os.path.join(input_folder, "palette.png")

    if os.path.exists(input_folder):
        # Step 1: Generate palette
        palette_command = [
            'ffmpeg',
            '-y',
            '-framerate', str(framerate),
            '-start_number', str(start_num),
            '-i', os.path.join(input_folder, '%d.png'),
            '-vf', "scale=iw:ih:force_original_aspect_ratio=decrease,"
                   "pad=ceil(iw/2)*2:ceil(ih/2)*2,palettegen",
            '-frames:v', str(num_frames),
            palette_file
        ]

        # Step 2: Create GIF using palette
        gif_command = [
            'ffmpeg',
            '-y',
            '-framerate', str(framerate),
            '-start_number', str(start_num),
            '-i', os.path.join(input_folder, '%d.png'),
            '-i', palette_file,
            '-lavfi', "scale=iw:ih:force_original_aspect_ratio=decrease,"
                      "pad=ceil(iw/2)*2:ceil(ih/2)*2,paletteuse",
            '-frames:v', str(num_frames),
            output_file
        ]

        try:
            print(f"\nGenerating palette for {name}...")
            subprocess.run(palette_command, check=True, capture_output=True, text=True)

            print(f"Creating GIF for {name}...")
            subprocess.run(gif_command, check=True, capture_output=True, text=True)

            print(f"\nGIF created successfully: {output_file}\n")

        except subprocess.CalledProcessError as e:
            print(f"\nError during GIF creation for {name}:\n{e.stderr}\n")

    else:
        print(f"\nInput folder not found: {input_folder}\n")