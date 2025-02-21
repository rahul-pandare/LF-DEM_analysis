import os
import glob
import matplotlib               # type: ignore
import numpy             as np  # type: ignore
import matplotlib.pyplot as plt # type: ignore

'''
Feb 5, 2025
RVP

This script produces snapshots of normal force interactions for a given range of strain units.
NOTE: Script creates a directory to store snapshots if it does not exist already

pre-requisite - need specific directories to store the images
'''

# Input and output paths
topDir        = '/Volumes/rahul_2TB/high_bidispersity/new_data/'
fig_save_path = '/Users/rahul/City College Dropbox/Rahul Pandare/CUNY/research/bidisperse_project/conferences/GSOE_poster/movies/'

# Path errors.
print(f"Error: Path '{topDir}' not found. Check mount point") if not os.path.exists(topDir) else None
print(f"Error: Path '{fig_save_path}' not found. Check mount point") if not os.path.exists(fig_save_path) else None

# Simulation parameters
npp    = 1000
phi    = [0.77] #, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76]
ar     = [1.4]  #, 1.4, 1.8, 2.0, 4.0]
vr     = ['0.5']
numRun = 1

# Particles data file
particleFile    = 'par_*.dat' 
interactionFile = 'int_*.dat' 

plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
plt.rcParams["figure.autolayout"]   = True
matplotlib.use('Agg')

"====================================================================================================================================="

def parList(particleFile):
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

def interactionsList(interactionFile):
    '''
    This function reads the interaction file and creates a nested-list,
    each list inside contains the array of all interaction parameters for
    that timestep.

    Input: interactionFile - the location of the interaction data file
    '''

    hashCounter = 0
    temp        = []
    contactList = [] # list with interaction parameters for each element at each timestep

    fileLines = interactionFile.readlines()[27:] # skipping the comment lines
    for line in fileLines:
        if not line.split()[0] == '#':
            lineList = [float(value) for value in line.split()]
            temp.append(lineList)
        else:
            hashCounter += 1 # checking if counter reaches 7 (7 lines of comments after every timestep data)
            if hashCounter == 7: 
                contactList.append(np.array(temp))
                temp        = []
                hashCounter = 0
    interactionFile.close()
    return contactList

"====================================================================================================================================="

# Frame details
startFrame = 600
endFrame   = 700

maxLineWidth = 5

for j in range(len(phi)):
    phii = phi[j]
    phii = '{:.3f}'.format(phi[j]) if len(str(phi[j]).split('.')[1]) > 2 else '{:.2f}'.format(phi[j])
    for k in range(len(ar)):
        for l in range(len(vr)):
            dataname = f"{topDir}NP_{npp}/phi_{phii}/ar_{ar[k]}/Vr_{vr[l]}/run_{numRun}"
            if os.path.exists(dataname):
                intPath = open(glob.glob(f'{dataname}/{interactionFile}')[0])
                intList = interactionsList(intPath)

                parPath  = open(glob.glob(f'{dataname}/{particleFile}')[0])
                parLines = parPath.readlines()
                parList1 = parList(parPath)

                # box dimensions
                Lx = float(parLines[3].split()[2]) 
                Lz = float(parLines[3].split()[2])

                # setting up axis and box walls
                _, ax = plt.subplots(1, 1, figsize=(5,5), dpi = 500)
                newLx = Lx + 2*ar[k]
                newLz = Lz + 2*ar[k]
                
                # getting the max force from the given range of frames
                maxForces = []
                for frame in range(startFrame, endFrame):
                    frameList = intList[frame]
                    lubNorm   = frameList[:,6]
                    contNorm  = frameList[:,11]
                    repulNorm = frameList[:,16]
                    normInts  = lubNorm + contNorm + repulNorm

                    maxForces.append(np.max(normInts))
                
                maxForce = np.max(maxForces)
                
                print("\n")
                # plotting frames
                for frame in range(startFrame, endFrame):
                    # position and radius data from par_*.dat
                    px = parList1[frame][:,2]
                    pz = parList1[frame][:,3]
                    pr = parList1[frame][:,1]
                    NP = len(pr)
                    
                    # interaction data from int_*.dat
                    frameList = intList[frame]
                    pi        = np.array([int(x) for x in frameList[:,0]])
                    pj        = np.array([int(x) for x in frameList[:,1]])
                    normxij   = frameList[:,2]
                    normzij   = frameList[:,4]
                    gapij     = frameList[:,5]
                    lubNorm   = frameList[:,6]
                    contState = frameList[:,10]
                    contNorm  = frameList[:,11]
                    repulNorm = frameList[:,16]
                    normInts  = lubNorm + contNorm + repulNorm

                    # plot parameters
                    numInts       = len(contState)
                    contLineWidth = np.array(normInts) * maxLineWidth / maxForce
                    intColor      = np.array(['r']*numInts, dtype=object)
                    contLess      = np.array(contState == 0, dtype=bool)
                    fricLess      = np.array(contState == 1, dtype=bool)
                    if contLess.size > 0: intColor[contLess] = '#80EF80'
                    if fricLess.size > 0: intColor[fricLess] = 'tab:cyan'

                    fig, ax = plt.subplots(1, 1, figsize=(5,5))
                    
                    # plotting all particles 
                    for i in range(NP):
                        if pr[i] == 1:
                            circle = plt.Circle((px[i],pz[i]), pr[i], facecolor='#9FB5C4', fill=True, edgecolor='none')
                        else:
                            circle = plt.Circle((px[i],pz[i]), pr[i], facecolor='#8A9BA8', fill=True, edgecolor='none')
                        ax.add_artist(circle)
                    
                    # plotting interactions
                    for i in range(numInts):
                        p1  = np.array([px[pi[i]], pz[pi[i]]])    # particle 1 position
                        p2  = np.array([px[pj[i]], pz[pj[i]]])    # particle 1 position
                        nij = np.array([normxij[i], normzij[i]]) # normal vector 1 to 2
                        pir = pr[pi[i]]
                        pjr = pr[pj[i]]
                        rij = nij * (gapij[i] + 2.) * (pir + pjr) * 0.5 # vector length between partcle 1 and 2
                        p11 = p1 + rij
                        ax.plot([p1[0], p11[0]],[p1[1], p11[1]], color=intColor[i], linewidth=contLineWidth[i], solid_capstyle='round', solid_joinstyle='round')
                       
                        if (np.sign(nij[0]) != np.sign((p2 - p1)[0])) or (np.sign(nij[1]) != np.sign((p2 - p1)[1])):
                            p22 = p2 - rij
                            ax.plot([p2[0], p22[0]], [p2[1], p22[1]], color=intColor[0], linewidth=contLineWidth[0], solid_capstyle='round', solid_joinstyle='round')
                    
                    ax.set_xlim([-(newLx/2+0.2),(newLx/2+0.2)])
                    ax.set_ylim([-(newLz/2+0.2),(newLz/2+0.2)])
                    #ax.set_title(fr'$\gamma = {frame/100:.2f}$', pad=10, fontweight='bold')
                    ax.axis('off')
                    ax.set_aspect('equal')
                    
                    directory = f'{fig_save_path}phi_{phii}_ar_{ar[k]}_vr_{vr[l]}_int'
                    os.makedirs(directory, exist_ok=True)
                    fig.savefig(f'{directory}/{frame}.png', dpi=500, transparent=True)
                    print(f'>     Processed frame: {frame}/{endFrame-1}      ')
                    plt.close()
                    
                plt.close('all')