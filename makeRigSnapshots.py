import os
import sys
import glob
import matplotlib                       # type: ignore
import numpy             as     np      # type: ignore
import matplotlib.pyplot as     plt     # type: ignore
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

# mac paths
topDir        = '/Volumes/rahul_2TB/high_bidispersity/new_data/'
fig_save_path = '/Users/rahul/City College Dropbox/Rahul Pandare/CUNY/research/bidisperse_project/figures/ang_vel/'

# linux paths
#topDir        = '/media/rahul/rahul_2TB/high_bidispersity/new_data/'
#fig_save_path = '/media/Linux_1TB/City College Dropbox/Rahul Pandare/CUNY/research/bidisperse_project/figures/ang_vel/'

# Path errors.
print(f"Error: Path '{topDir}' not found. Check mount point") if not os.path.exists(topDir) else None
print(f"Error: Path '{fig_save_path}' not found. Check mount point") if not os.path.exists(fig_save_path) else None

# Simulation parameters.
npp    = 1000
phi    = [0.77] #, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76]
ar     = [2.0]  #, 1.4, 1.8, 2.0, 4.0]
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
startFrame = 900
endFrame   = 1000

for j in range(len(phi)):
    phii = phi[j]
    phii = '{:.3f}'.format(phii) if len(str(phii).split('.')[1]) > 2 else '{:.2f}'.format(phii)
    for k in range(len(ar)):
        for l in range(len(vr)):
            dataname = topDir + 'NP_' + str(npp) + '/phi_' + phii + '/ar_' + str(ar[k]) + '/Vr_' + vr[l]
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

                directory = f'{fig_save_path}phi_{phii}_ar_{ar[k]}_vr_{vr[l]}_rig'
                os.makedirs(directory, exist_ok=True)
                
                for frame in range(startFrame, endFrame):
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
                    #ax.set_title(rf"$\boldsymbol{{\gamma}} = \mathbf{{{frame/100:.2f}}}$", fontsize=16, pad=8, color=tColor)

                    # Saving figure
                    fig.savefig(f'{directory}/{frame}.png', dpi=400)
                    print(f'>     Processed frame: {frame}/{endFrame-1}      ')
                    plt.close(fig)
            else:
                print(f"{dataname} - Not Found")