import os
import matplotlib                   # type: ignore
import numpy             as     np  # type: ignore
import matplotlib.pyplot as     plt # type: ignore
import colorsys
from matplotlib import colors       # type: ignore
import readFiles

matplotlib.use('TkAgg')
'''
May 13
RVP

This code makes snapshots for only LL contact persistance with a trail of last 10 timesteps of parcticle positions.
'''

# Input and output paths.
topDir        = '/media/rahul/rahul_2TB/high_bidispersity/new_data/'
fig_save_path = '/media/Linux_1TB/City College Dropbox/Rahul Pandare/CUNY/research/bidisperse_project/figures/paper/LL_perse_snaps/'

# Path errors.
print(f"Error: Path '{topDir}' not found. Check mount point") if not os.path.exists(topDir) else None
print(f"Error: Path '{fig_save_path}' not found. Check mount point") if not os.path.exists(fig_save_path) else None

# Simulation parameters.
npp = 1000
phi = [0.78] #, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76]
ar  = [4.0] #, 1.4, 1.8, 2.0, 4.0]
vr  = '0.75'
run = 1

# Particles data file.
parFile     = 'par_random_seed_params_stress100r_shear.dat'
intFile     = 'int_random_seed_params_stress100r_shear.dat'
ranSeedFile = "random_seed.dat"
cmap        = matplotlib.colormaps['gist_rainbow'] # type: ignore
alpha       = 0.75
hls         = np.array([colorsys.rgb_to_hls(*c) for c in cmap(np.arange(cmap.N))[:,:3]])
hls[:,1]   *= alpha
rgb         = np.clip(np.array([colorsys.hls_to_rgb(*c) for c in hls]), 0,1)
cmap        = colors.LinearSegmentedColormap.from_list("", rgb)

frames = [200, 1500]
for j in range(len(phi)):
    phii = phi[j]
    phii = '{:.3f}'.format(phii) if len(str(phii).split('.')[1]) > 2 else '{:.2f}'.format(phii)
    for k in range(len(ar)):
        dataname = f"{topDir}NP_{npp}/phi_{phii}/ar_{ar[k]}/Vr_{vr}"
        pr = ar[k]
        if os.path.exists(dataname):
            with open(f'{dataname}/run_{run}/{parFile}', 'r') as particleFile:
                lines = particleFile.readlines()
                particlesList = readFiles.readParFile(particleFile)

            interFile   = open(f'{dataname}/run_{run}/{intFile}', 'r')
            contactList = readFiles.interactionsList(interFile) 

            Lx = float(lines[3].split()[2]) 
            Lz = float(lines[3].split()[2])
            newLx = Lx + 2 * np.max(pr)
            newLz = Lz + 2 * np.max(pr)

            trajHistory = 20

            for kk in range(frames[0], frames[1]):
                print(f'Generating snapshot {kk} out of {frames[1]}')
                
                _, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=500)
                ax.clear()

                s       = trajHistory
                opacity = 0
                for frame in range(kk - trajHistory, kk):
                    s -= 1 
                    opacity += 1/trajHistory
                    for i in range(len(contactList[frame])):
                        contState = int(contactList[frame][i][10])
                        partIndex1 = int(contactList[frame][i][0])
                        partIndex2 = int(contactList[frame][i][1])
                        
                        pr1 = particlesList[frame][partIndex1][1]
                        pr2 = particlesList[frame][partIndex2][1]
                        
                        if contState == 2 and (pr1 == pr2 > 1):
                            px1 = particlesList[frame][partIndex1][2]
                            pz1 = particlesList[frame][partIndex1][3]

                            px2 = particlesList[frame][partIndex2][2]
                            pz2 = particlesList[frame][partIndex2][3]

                            if s == 0: #current particle
                                circle1 = plt.Circle((px1, pz1), pr, facecolor='#8B0000', fill=True, edgecolor='None')
                                circle2 = plt.Circle((px2, pz2), pr, facecolor='#8B0000', fill=True, edgecolor='None')
                                ax.add_artist(circle1)
                                ax.add_artist(circle2)
                            else:
                                circle1 = plt.Circle((px1, pz1), pr, facecolor='#91bfdb', fill=True, edgecolor='None', alpha=opacity)
                                circle2 = plt.Circle((px2, pz2), pr, facecolor='#91bfdb', fill=True, edgecolor='None', alpha=opacity)
                                ax.add_artist(circle1)
                                ax.add_artist(circle2)

                # Set limits and aspect for the plot
                ax.set_xlim([-(newLx / 2), (newLx / 2)])
                ax.set_ylim([-(newLz / 2), (newLz / 2)])
                ax.axis('off')
                ax.set_aspect('equal')

                # Saving figure for the current batch of frames
                directory = f'{fig_save_path}phi_{phii}_ar_{ar[k]}_vr_{vr}_LL_perse'
                os.makedirs(directory, exist_ok=True)
                figFormat = ".png"
                plt.savefig(f'{directory}/{str(kk)}{figFormat}', bbox_inches="tight", dpi=800)
                print(f'>     Processed frame: {frame}/{frames[1]-1}      ')
                plt.close()
        else:
            print(f"{dataname} - Not Found")