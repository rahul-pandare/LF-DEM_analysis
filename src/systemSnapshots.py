import os
import matplotlib                   # type: ignore
import numpy             as     np  # type: ignore
import matplotlib.pyplot as     plt # type: ignore
import colorsys
from matplotlib import colors       # type: ignore
matplotlib.use('TkAgg')

'''
Feb 12, 2025
RVP

This script produces snapshots for a given range of frames (strain units)
NOTE: Script creates a directory to store snapshots if it does not exist already
'''

# Input and output paths.
topDir        = '/Volumes/rahul_2TB/high_bidispersity/new_data/'
fig_save_path = '/Users/rahul/City College Dropbox/Rahul Pandare/CUNY/research/bidisperse_project/figures/new_data/movies/system/'

# Path errors.
print(f"Error: Path '{topDir}' not found. Check mount point") if not os.path.exists(topDir) else None
print(f"Error: Path '{fig_save_path}' not found. Check mount point") if not os.path.exists(fig_save_path) else None

# Simulation parameters.
npp    = 1000
phi    = [0.795] #, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76]
ar     = [4.0]  #, 1.4, 1.8, 2.0, 4.0]
vr     = '0.25'
numRun = 1

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

"==========================================================================================="

startFrame = 1 * 100
endFrame   = 5 * 100

for j in range(len(phi)):
    phii = phi[j]
    phii = '{:.3f}'.format(phii) if len(str(phii).split('.')[1]) > 2 else '{:.2f}'.format(phii)
    for k in range(len(ar)):
        dataname = f"{topDir}NP_{npp}/phi_{phii}/ar_{ar[k]}/Vr_{vr}"
        if os.path.exists(dataname):
            with open(f'{dataname}/run_{numRun}/{parFile}', 'r') as particleFile:
                lines = particleFile.readlines()
                particlesList = readParFile(particleFile)

            # Box dimensions.
            Lx = float(lines[3].split()[2]) 
            Lz = float(lines[3].split()[2])

            newLx = Lx + 2*ar[k]
            newLz = Lz + 2*ar[k]
            
            for kk in range(startFrame, endFrame):
                # Particle sizes and radii.
                px =particlesList[kk][:,2]
                pz =particlesList[kk][:,3]
                pr =particlesList[kk][:,1]

                # Setting up axis and box walls.
                fig, ax = plt.subplots(1, 1, figsize=(5,5), dpi = 500)
                
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
                
                directory = f'{fig_save_path}phi_{phii}_ar_{ar[k]}_vr_{vr}'
                os.makedirs(directory, exist_ok=True)
                fig.savefig(f'{directory}/{kk}.png', dpi=400)
                print(f'>     Processed frame: {kk}/{endFrame-1}      ')
                plt.close()
        else:
            print(f"{dataname} - Not Found")