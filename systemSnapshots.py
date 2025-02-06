import os
import matplotlib                   # type: ignore
import numpy             as     np  # type: ignore
import matplotlib.pyplot as     plt # type: ignore
import colorsys
from matplotlib import colors       # type: ignore
matplotlib.use('TkAgg')

'''
July 22, 2024
RVP

This script produces snapshots for a given range of strain units.
Snapshots are produced for all the phi and ar values mentioned.
NOTE: snapshots produced for just one run (run = 1)

Command to execute in terminal:
python3 -c systemSnapshots.py
'''

# Input and output paths.
topDir        = '/media/rahul/Rahul_2TB/high_bidispersity/'
fig_save_path = '/home/rahul/Dropbox (City College)/CUNY/Research/Bidisperse Project/SOR_meeting/figures/snapshots/movie/phi_0.78_ar_4.0/'

# Path errors.
print(f"Error: Path '{topDir}' not found. Check mount point") if not os.path.exists(topDir) else None
print(f"Error: Path '{fig_save_path}' not found. Check mount point") if not os.path.exists(fig_save_path) else None

# Simulation parameters.
npp = 1000
phi = [0.78] #, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76]
ar  = [4.0]  #, 1.4, 1.8, 2.0, 4.0]

cmap      = matplotlib.colormaps['gist_rainbow'] # type: ignore
hls       = np.array([colorsys.rgb_to_hls(*c) for c in cmap(np.arange(cmap.N))[:,:3]])
alpha     = 0.75
hls[:,1] *= alpha
rgb       = np.clip(np.array([colorsys.hls_to_rgb(*c) for c in hls]), 0,1)
cmap      = colors.LinearSegmentedColormap.from_list("", rgb)

# Particles data file.
parFile = 'par_random_seed_params_stress100r_shear.dat'

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

endFrame = 10*100
for j in range(len(phi)):
    phii = phi[j]
    phii = '{:.3f}'.format(phii) if len(str(phii).split('.')[1]) > 2 else '{:.2f}'.format(phii)
    for k in range(len(ar)):
        dataname = topDir + 'NP_' + str(npp) + '/phi_' + phii + '/ar_' + str(ar[k]) + '/Vr_0.5'
        if os.path.exists(dataname):
            for l in range(1):
                with open(f'{dataname}/run_{l+1}/{parFile}', 'r') as particleFile:
                    lines = particleFile.readlines()
                    particlesList = readParFile(particleFile)

                # Box dimensions.
                Lx = float(lines[3].split()[2]) 
                Lz = float(lines[3].split()[2])

                for kk in range(endFrame):
                    # Particle sizes and radii.
                    px =particlesList[kk][:,2]
                    pz =particlesList[kk][:,3]
                    pr =particlesList[kk][:,1]

                    # Setting up axis and box walls.
                    _, ax = plt.subplots(1, 1, figsize=(5,5), dpi = 500)
                    newLx = Lx + 2*np.max(pr)
                    newLz = Lz + 2*np.max(pr)

                    ax.clear()
                    for i in range(len(particlesList[kk])):
                        if pr[i] == 1:
                            circle = plt.Circle((px[i],pz[i]), pr[i], facecolor='#fc8d59', fill=True, edgecolor='None')
                        else:
                            circle = plt.Circle((px[i],pz[i]), pr[i], facecolor='#91bfdb', fill=True, edgecolor='None')
                        ax.add_artist(circle)

                    ax.set_xlim([-(newLx/2+0.2),(newLx/2+0.2)])
                    ax.set_ylim([-(newLz/2+0.2),(newLz/2+0.2)])
                    ax.axis('off')
                    ax.set_aspect('equal')

                    # Saving figure.
                    figFormat     = ".png"
                    plt.savefig(fig_save_path + str(kk) + figFormat, bbox_inches = "tight", dpi = 800)

                    matplotlib.pyplot.close()
        else:
            print(f"{dataname} - Not Found")