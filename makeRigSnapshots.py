import os
import sys
import glob
import matplotlib                       # type: ignore
import numpy             as     np      # type: ignore
import matplotlib.pyplot as     plt     # type: ignore

'''
Feb 3, 2025
RVP

This script produces snapshots with rigid clusters for a given range of strain units.
Snapshots are produced for all the phi and ar values mentioned.
NOTE: snapshots produced for just one run (run = 1)

Command to execute in terminal:
python3 -c makeRigSnapshots.py
'''

# Input and output paths.
topDir        = '/media/Linux_1TB/new_Data/'
fig_save_path = '/media/Linux_1TB/figures/'

# Path errors.
print(f"Error: Path '{topDir}' not found. Check mount point") if not os.path.exists(topDir) else None
print(f"Error: Path '{fig_save_path}' not found. Check mount point") if not os.path.exists(fig_save_path) else None

# Simulation parameters.
npp    = 1000
phi    = [0.77] #, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76]
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

"====================================================================================================================================="

def ParList(particleFile):
    '''
    Function to read parameters file (par*.dat). We read this file to get 
    particle positions
    '''
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

def rigList(rigidFile):
    '''
    Function to read rigid file (rig*.dat). We read this file to get 
    particle IDs of rigid particles in each timestep.
    '''
    hashCounter = -4
    clusterIDs  = []
    temp = []
    for line in rigidFile:
        if line[0] == '#':
            hashCounter += 1
            if len(temp) > 0:
                clusterIDs.append(temp)
                temp = []
        elif hashCounter >= 0:
            temp.append(line.strip())
            
    rigClusterIDsList = []
    for _, sampleList in enumerate(clusterIDs):
        tempList = []
        for kk in range(len(sampleList)):
            tempList.append([int(indx) for indx in sampleList[kk].split(',')])
        rigClusterIDsList.append(tempList)
    return rigClusterIDsList
"====================================================================================================================================="

# Frame details
startFrame = 1000
endFrame   = 1010

for j in range(len(phi)):
    phii = phi[j]
    phii = '{:.3f}'.format(phii) if len(str(phii).split('.')[1]) > 2 else '{:.2f}'.format(phii)
    for k in range(len(ar)):
        for l in range(len(vr)):
            dataname = topDir + 'NP_' + str(npp) + '/phi_' + phii + '/ar_' + str(ar[k]) + '/Vr_' + vr[l]
            if os.path.exists(dataname):
                particleFile  = open(glob.glob(f'{dataname}/run_{numRum}/{parFile}')[0])
                parLines      = particleFile.readlines()
                particlesList = ParList(particleFile)

                rigFilePath   = glob.glob(f'{dataname}/run_{numRum}/{rigFile}')
                if not rigFilePath:
                    print(f"Error: {rigFile} not found at {dataname}/run_{numRum}")
                    sys.exit(1)
                rigidFile     = open(rigFilePath[0])
                rigClusterIDs = rigList(rigidFile)
                clusterIDs    = [[np.nan] if len(samplelist[0]) < 2 else list({int(num) for sublist in samplelist for num in sublist}) for samplelist in rigClusterIDs]

                # Box dimensions.
                Lx = float(parLines[3].split()[2]) 
                Lz = float(parLines[3].split()[2])

                for kk in range(startFrame, endFrame):
                    # Particle sizes and radii.
                    px = particlesList[kk][:,2]
                    pz = particlesList[kk][:,3]
                    pr = particlesList[kk][:,1]

                    # Setting up axis and box walls.
                    _, ax = plt.subplots(1, 1, figsize=(5,5), dpi = 500)
                    newLx = Lx + 2*np.max(pr)
                    newLz = Lz + 2*np.max(pr)

                    allPart      = particlesList[kk][:,0]
                    rigidPart    = clusterIDs[kk]
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
                    #ax.set_title(rf"$\gamma = {kk/100:.2f}$", fontsize=12, pad=5)

                    # Saving figure
                    figFormat     = ".png"
                    plt.savefig(fig_save_path + str(kk) + figFormat, bbox_inches = "tight", dpi = 300)

                    print(f'Gamma: {kk/100:.2f}')
                    matplotlib.pyplot.close()
            else:
                print(f"{dataname} - Not Found")