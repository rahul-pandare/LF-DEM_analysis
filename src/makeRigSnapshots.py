import os
import sys
import glob
import platform
from   tqdm              import tqdm # type: ignore
from   pathlib           import Path
import matplotlib                    # type: ignore
import numpy             as     np   # type: ignore
import matplotlib.pyplot as     plt  # type: ignore
import readFiles

'''
Feb 3, 2025
RVP

This script produces snapshots with rigid clusters for a given range of strain units.
Snapshots are produced for all the phi and ar values mentioned.
NOTE: snapshots produced for just one run (run = 1)

Command to execute in terminal:
python3 makeRigSnapshots.py
'''
system_platform = platform.system()

if system_platform == 'Darwin':  # macOS
    topDir = Path("/Volumes/rahul_2TB/high_bidispersity/new_data/")
    fig_save_path = Path("/Users/rahul/City College Dropbox/Rahul Pandare/CUNY/research/bidisperse_project/figures/ang_vel/")
elif system_platform == 'Linux':
    topDir = Path("/media/rahul/rahul_2TB/high_bidispersity/new_data/")
    fig_save_path = Path("/media/Linux_1TB/City College Dropbox/Rahul Pandare/CUNY/research/bidisperse_project/figures/ang_vel/")
else:
    raise OSError(f"Unsupported OS: {system_platform}")

# Validate paths
for path in [topDir, fig_save_path]:
    if not path.exists():
        print(f"Error: Path '{path}' not found. Check mount point.")

# Simulation parameters.
npp    = 1000
phi    = [0.76] #, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76]
ar     = [1.4]  #, 1.4, 1.8, 2.0, 4.0]
vr     = ['0.5']
numRum = 1

# Particles data file.
parFile = 'par_*.dat'
rigFile = 'rig_*.dat'

plt.rcParams.update({
    "figure.max_open_warning": 0,
    "text.usetex": True,
    "figure.autolayout": True,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size":        10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "patch.linewidth":  .2,
    "lines.markersize":  5,
    "hatch.linewidth":  .2
})
plt.rcParams['text.latex.preamble']= r"\usepackage{amsmath}"
matplotlib.use('Agg')

# Frame details
startFrame = 100
endFrame   = 1650
#or
frames     = [101, 224, 228, 229, 231, 256, 274, 463, 474, 505, 516, 538, 570, 575, 630, 703, 757, 760, 773, 863, 952, 962, 1055, 1072, 1077, 1084, 1157, 1310, 1358, 1611, 1612, 1617, 1643, 1664, 1782, 1986, 1994]

for j in range(len(phi)):
    phii = phi[j]
    phii = '{:.3f}'.format(phii) if len(str(phii).split('.')[1]) > 2 else '{:.2f}'.format(phii)
    for k in range(len(ar)):
        for l in range(len(vr)):
            dataname = topDir / f"NP_{npp}/phi_{phii}/ar_{ar[k]}/Vr_{vr[l]}"
            if os.path.exists(dataname):
                particleFile  = open(glob.glob(f'{dataname}/run_{numRum}/{parFile}')[0])
                parLines      = particleFile.readlines()
                particlesList = readFiles.readParFile(particleFile)

                rigFilePath   = glob.glob(f'{dataname}/run_{numRum}/{rigFile}')
                if not rigFilePath:
                    print(f"Error: {rigFile} not found at {dataname}/run_{numRum}")
                    sys.exit(1)
                rigidFile     = open(rigFilePath[0])
                rigClusterIDs = readFiles.rigList(rigidFile)
                clusterIDs    = [[np.nan] if len(samplelist[0]) < 2 else list({int(num) for sublist in samplelist for num in sublist}) for samplelist in rigClusterIDs]

                # Box dimensions.
                Lx = float(parLines[3].split()[2]) 
                Lz = float(parLines[3].split()[2])

                directory = f'{fig_save_path}/phi_{phii}_ar_{ar[k]}_vr_{vr[l]}_rig'
                os.makedirs(directory, exist_ok=True)
                
                #for frame in tqdm(range(startFrame, endFrame), desc="Outer loop"):
                for frame in tqdm(frames, desc="Inner loop", leave=False):
                    # Particle sizes and radii.
                    px = particlesList[frame][:,2]
                    pz = particlesList[frame][:,3]
                    pr = particlesList[frame][:,1]

                    # Setting up axis and box walls.
                    fig, ax = plt.subplots(1, 1, figsize=(5,5), dpi = 500)
                    newLx   = Lx + 2*np.max(pr)
                    newLz   = Lz + 2*np.max(pr)

                    allPart      = particlesList[frame][:,0]
                    rigidPart    = clusterIDs[frame]
                    notRigidPart = allPart[np.isin(allPart, rigidPart) == False]

                    ax.clear()
                    for index in notRigidPart:
                        index  = int(index)
                        circle = plt.Circle((px[index],pz[index]), pr[index], facecolor='w', edgecolor='k', linewidth=0.5, zorder=1)
                        ax.add_artist(circle)
                    
                    for index in rigidPart:
                        circle = plt.Circle((px[index],pz[index]), pr[index], facecolor='#A00000', edgecolor=None, zorder=2)
                        ax.add_artist(circle)

                    ax.set_xlim([-(newLx/2+0.2),(newLx/2+0.2)])
                    ax.set_ylim([-(newLz/2+0.2),(newLz/2+0.2)])
                    ax.axis('off')
                    ax.set_aspect('equal')
                    ax.set_title(fr'$\phi = {phii}, \;\delta = {ar[k]}, \; \zeta = {float(vr[l]):.2f},\; \gamma = {frame/100:.2f}$', 
                                 fontsize=10, pad=6, fontweight='bold', x=0.5)

                    # Saving figure
                    fig.savefig(f'{directory}/{frame}.png', dpi=400)
                    #print(f'>     Processed frame: {frame}/{endFrame-1}      ')
                    plt.close(fig)
            else:
                print(f"{dataname} - Not Found")