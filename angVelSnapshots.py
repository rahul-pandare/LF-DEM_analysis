
import os
import glob
import platform
from   tqdm              import tqdm  # type: ignore
from   pathlib           import Path
import numpy             as np        # type: ignore
import matplotlib.pyplot as plt       # type: ignore
import matplotlib.colors as mcolors   # type: ignore
import readFiles

'''
Jun 2, 2025
RVP

This script produces snapshots Snapshots for angular velocity and 
traslational velocity with normalised scale for angular velocity wrt shear rate
NOTE: snapshots produced for just one run (run = 1)

Command to execute in terminal:
python3 angVelSnapshots.py
'''

plt.rcParams["text.usetex"]         = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"  
plt.rcParams["figure.autolayout"]   = True

system_platform = platform.system()
if system_platform == 'Darwin':  # macOS
    topDir = Path("/Volumes/rahul_2TB/high_bidispersity/new_data/")
    fig_save_path = Path("/Users/rahul/City College Dropbox/Rahul Pandare/CUNY/research/bidisperse_project/figures/ang_vel/")
elif system_platform == 'Linux':
    topDir = Path("/media/rahul/rahul_2TB/high_bidispersity/new_data/")
    fig_save_path = Path("/media/Linux_1TB/City College Dropbox/Rahul Pandare/CUNY/research/bidisperse_project/figures/ang_vel/")
else:
    raise OSError(f"Unsupported OS: {system_platform}")

# Simulation parameters
npp    = 1000
phi    = [0.76] #, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76]
ar     = [1.4]  #, 1.4, 1.8, 2.0, 4.0]
vr     = ['0.5']
numRun = 1

# Particles data file
particleFile    = 'par_*.dat' 
interactionFile = 'int_*.dat' 
dataFile        = 'data_*.dat'

# Frame details
startFrame  = 99
endFrame    = 101
#or
frames     = [101, 224, 228, 229, 231, 256, 274, 463, 474, 505, 516, 538, 570, 575, 630, 703, 757, 760, 773, 863, 952, 962, 1055, 1072, 1077, 1084, 1157, 1310, 1358, 1611, 1612, 1617, 1643, 1664, 1782, 1986, 1994]

# anvel limit for visualization
angVelRange = [-20, 20]

for i, phii in enumerate(phi):
    phii = '{:.3f}'.format(phii) if len(str(phii).split('.')[1]) > 2 else '{:.2f}'.format(phii)
    for j, arj in enumerate(ar):
        for k, vrk in enumerate(vr):
            dataname = f"{topDir}/NP_{npp}/phi_{phii}/ar_{arj}/Vr_{vrk}/run_{numRun}"
            if os.path.exists(dataname):
                parPath  = open(glob.glob(f'{dataname}/{particleFile}')[0])
                dat_file = glob.glob(f'{dataname}/{dataFile}')[0]
                
                parLines = parPath.readlines()
                parList  = readFiles.readParFile(parPath)
                data     = np.loadtxt(dat_file)
                srate    = data[:, 2]

                # Box size & colormap
                Lx, Lz       = float(parLines[3].split()[2]), float(parLines[3].split()[2])
                newLx, newLz = Lx + 2*arj, Lz + 2*arj
                minAngVelres = -0.5
                maxAngVelres =  0.5
                colorNorm    = mcolors.Normalize(vmin=minAngVelres, vmax=maxAngVelres)

                directory = f'{fig_save_path}/phi_{phii}_ar_{arj}_vr_{vrk}_angV_transV2'
                os.makedirs(directory, exist_ok=True)

                #for frame in tqdm(range(startFrame, endFrame), desc="Outer loop"):
                for frame in tqdm(frames, desc="Inner loop", leave=False):
                    #frame = 1500
                    frameList = parList[frame]
    
                    pIndex  = frameList[:,0]
                    pr      = frameList[:,1]
                    px      = frameList[:,2]
                    pz      = frameList[:,3]
                    velx    = frameList[:,4]
                    velz    = frameList[:,6]
                    angVely = frameList[:,8]
                    
                    angVelyNorm = angVely/srate[frame]
                    #colorNorm   = mcolors.Normalize(vmin =   angVelRange[0], vmax = angVelRange[1])
                    colorNorm   = mcolors.Normalize(vmin = min(angVelyNorm), vmax = max(angVelyNorm))
                    colors      = plt.cm.coolwarm(colorNorm(angVelyNorm))
                    fig, ax     = plt.subplots(1, 1, figsize=(3,3), dpi=300)
    
                    # plotting all particles 
                    for i in range(npp):
                        circle = plt.Circle((px[i],pz[i]), pr[i], facecolor=colors[i], fill=True, edgecolor='none')
                        ax.add_artist(circle)
                        #ax.arrow(px[i], pz[i], velx[i], velz[i], head_width=0.05, head_length=0.05, fc='k', ec='k') 
                        ax.quiver(px[i], pz[i], velx[i], velz[i], angles='xy', scale_units='xy', scale=.6, 
                                  color     = 'k',  width     = 0.0018,                    # Makes arrow shaft very thin
                                  headwidth = 3,   headlength = 3,      headaxislength=2,  # Makes arrowhead smaller
                                  linewidth = 0.1, zorder     = 10)
    
                    ax.set_xlim([-(newLx/2+0.2),(newLx/2+0.2)])
                    ax.set_ylim([-(newLz/2+0.2),(newLz/2+0.2)])
                    ax.set_title(fr'$\phi = {phii}, \;\delta = {arj}, \; \zeta = {float(vrk):.2f},\; \gamma = {frame/100:.2f}$', 
                                 fontsize=7, pad=6, fontweight='bold', x=0.5)
                    ax.axis('off')
                    ax.set_aspect('equal')
    
                    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=colorNorm)  # Using 'RdYlBu' colormap
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02, shrink=1.0, aspect=30)
                    cbar.set_label(r"$\omega/ \dot \gamma$", fontsize=5, labelpad=-2)
                    cbar.outline.set_linewidth(0.4)
                    #cbar.set_ticks(np.linspace(-0.5, 0.5, 7))
                    #cbar.set_ticklabels([r'$-0.5$', r'$-0.3$', r'$-0.1$', r'$0$', r'$0.1$', r'$0.3$', r'$0.5$'])
                    cbar.ax.tick_params(labelsize=5, width=0.4)

                    fig.tight_layout()
                    #fig.savefig(f'{fig_save_path}/{frame}.svg', dpi=400, bbox_inches="tight")
                    fig.savefig(f'{directory}/{frame}.png', dpi=500, bbox_inches="tight")
                    print(f'>     Processed frame: {frame}/{endFrame-1}      ')
                    plt.close(fig)
                    #gc.collect()