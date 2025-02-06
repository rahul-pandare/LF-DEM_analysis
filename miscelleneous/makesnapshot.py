#!/usr/bin/env python3
import os
import matplotlib                   # type: ignore
import numpy             as     np  # type: ignore
import matplotlib.pyplot as     plt # type: ignore
import colorsys
from matplotlib import colors       # type: ignore
matplotlib.use('TkAgg')

'''
July 30, 2024
RVP

This is a universal script. To be run from the particular case condition.
This script produces a snapshot in the PWD. For a certain timestep (frame = 300).

Command to execute in terminal at the location:
makesnapshot.py
'''

# Particles data file.
parFile = 'par_random_seed_params_stress100r_shear.dat'


cmap           = matplotlib.colormaps['gist_rainbow'] # type: ignore
alpha          = 0.75

hls            = np.array([colorsys.rgb_to_hls(*c) for c in cmap(np.arange(cmap.N))[:,:3]])
hls[:,1]      *= alpha
rgb            = np.clip(np.array([colorsys.hls_to_rgb(*c) for c in hls]), 0,1)
cmap           = colors.LinearSegmentedColormap.from_list("", rgb)

"====================================================================================================================================="

def readParFile(particleFile):
    particleFile.seek(0)
    hashCounter   = 0
    temp          = []
    particlesList = []

    fileLines = particleFile.readlines()[22:] # skipping the comment lines

    for line in fileLines:
        if not line.split()[0] == '#':
            lineList = [float(value) for value in line.split()]
            temp.append(lineList)
        else:
            # Checking if counter reaches 7 (7 lines of comments after every timestep data).
            hashCounter += 1 
            if hashCounter == 7: 
                particlesList.append(np.array(temp))
                temp        = []
                hashCounter = 0
    particleFile.close()
    return particlesList

"====================================================================================================================================="

frame = 300

dataname = os.getcwd() # current working directory
      
with open(f'{dataname}/{parFile}', 'r') as particleFile:
    lines = particleFile.readlines()
    particlesList = readParFile(particleFile)

# Box dimensions.
Lx = float(lines[3].split()[2]) 
Lz = float(lines[3].split()[2])

# Particle sizes and radii.
px =particlesList[frame][:,2]
pz =particlesList[frame][:,3]
pr =particlesList[frame][:,1]

# Setting up axis and box walls.
_, ax = plt.subplots(1, 1, figsize=(5,5), dpi = 500)
newLx = Lx + 2*np.max(pr)
newLz = Lz + 2*np.max(pr)

ax.clear()
for i in range(len(particlesList[frame])):
    if pr[i] == 1:
        circle = plt.Circle((px[i],pz[i]), pr[i], facecolor='#323232', fill=True, edgecolor='None')
    else:
        circle = plt.Circle((px[i],pz[i]), pr[i], facecolor='#ADD8E6', fill=True, edgecolor='None')
    ax.add_artist(circle)

ax.set_xlim([-(newLx/2+0.2),(newLx/2+0.2)])
ax.set_ylim([-(newLz/2+0.2),(newLz/2+0.2)])
ax.axis('off')
ax.set_aspect('equal')

# Saving figure.
figFormat     = ".png"
plt.savefig(dataname + '/snapshot_frame_' + str(frame) + figFormat, bbox_inches = "tight", dpi = 500)

matplotlib.pyplot.close()
print(f'\n Snapshot generated. \n')