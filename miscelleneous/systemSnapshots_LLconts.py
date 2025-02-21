import os
import glob
import matplotlib                   # type: ignore
import numpy             as     np  # type: ignore
import matplotlib.pyplot as     plt # type: ignore
import colorsys
from matplotlib import colors       # type: ignore
matplotlib.use('TkAgg')

'''
Feb 21, 2025
RVP

This script produces snapshots for a given range of frames (strain units)with contact lines between LL contacts.
NOTE: Script creates a directory to store snapshots if it does not exist already
'''

# Input and output paths.
topDir        = '/Volumes/rahul_2TB/high_bidispersity/new_data/'
fig_save_path = '/Users/rahul/City College Dropbox/Rahul Pandare/CUNY/research/bidisperse_project/conferences/GSOE_poster/movies/'

# Path errors.
print(f"Error: Path '{topDir}' not found. Check mount point") if not os.path.exists(topDir) else None
print(f"Error: Path '{fig_save_path}' not found. Check mount point") if not os.path.exists(fig_save_path) else None

# Simulation parameters.
npp    = 1000
phi    = [0.80] #, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76]
ar     = [4.0]  #, 1.4, 1.8, 2.0, 4.0]
vr     = '0.5'
numRun = 1

cmap      = matplotlib.colormaps['gist_rainbow'] # type: ignore
hls       = np.array([colorsys.rgb_to_hls(*c) for c in cmap(np.arange(cmap.N))[:,:3]])
alpha     = 0.75
hls[:,1] *= alpha
rgb       = np.clip(np.array([colorsys.hls_to_rgb(*c) for c in hls]), 0,1)
cmap      = colors.LinearSegmentedColormap.from_list("", rgb)

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

startFrame = 1700
endFrame   = 1900

for j in range(len(phi)):
    phii = phi[j]
    phii = '{:.3f}'.format(phii) if len(str(phii).split('.')[1]) > 2 else '{:.2f}'.format(phii)
    for k in range(len(ar)):
        dataname = f"{topDir}NP_{npp}/phi_{phii}/ar_{ar[k]}/Vr_{vr}/run_{numRun}"
        if os.path.exists(dataname):
            intPath = open(glob.glob(f'{dataname}/{interactionFile}')[0])
            intList = interactionsList(intPath)
            
            parPath  = open(glob.glob(f'{dataname}/{particleFile}')[0])
            parLines = parPath.readlines()
            parList1 = parList(parPath)

            # Box dimensions.
            Lx = float(parLines[3].split()[2]) 
            Lz = float(parLines[3].split()[2])

            newLx = Lx + 2*ar[k]
            newLz = Lz + 2*ar[k]
            
            for frame in range(startFrame, endFrame):
                # Particle sizes and radii
                px =parList1[frame][:,2]
                pz =parList1[frame][:,3]
                pr =parList1[frame][:,1]

                frameList = intList[frame]
                pi        = np.array([int(x) for x in frameList[:,0]])
                pj        = np.array([int(x) for x in frameList[:,1]])
                normxij   = frameList[:,2]
                normzij   = frameList[:,4]
                gapij     = frameList[:,5]
                contState = frameList[:,10]
                numInts   = len(contState)

                # Setting up axis and box walls.
                fig, ax = plt.subplots(1, 1, figsize=(5,5), dpi = 500)
                
                # colors
                allSmall  = '#ADD8E6'   # light blue
                allLarge  = '#ADD8E6'   # blue
                contLarge = '#FF964F'   # orange
                bondLL    = '#8B6B4E' #'#A68B6D'   # brown

                ax.clear()
                for i in range(len(parList1[frame])):
                    if pr[i] == 1:
                        circle = plt.Circle((px[i], pz[i]), pr[i], facecolor=allSmall, fill=True, edgecolor='None', alpha=0.3)
                    else:
                        circle = plt.Circle((px[i],pz[i]), pr[i], facecolor=allLarge, fill=True, edgecolor='None', alpha=0.5)
                    ax.add_artist(circle)

                for i in range(numInts):
                    p1  = np.array([px[pi[i]], pz[pi[i]]])    # particle 1 position
                    p2  = np.array([px[pj[i]], pz[pj[i]]])    # particle 1 position
                    nij = np.array([normxij[i], normzij[i]])  # normal vector 1 to 2
                    pir = pr[pi[i]]
                    pjr = pr[pj[i]]
                    rij = nij * (gapij[i] + 2.) * (pir + pjr) * 0.5 # vector length between partcle 1 and 2
                    p11 = p1 + rij
                    pairContState = contState[i]
                    if pairContState == 2 and (pir == pjr > 1):
                        circle1 = plt.Circle(p1, pir, facecolor=contLarge, fill=True, edgecolor='None')
                        circle2 = plt.Circle(p2, pjr, facecolor=contLarge, fill=True, edgecolor='None')
                        ax.add_artist(circle1)
                        ax.add_artist(circle2)
                        ax.plot([p1[0], p11[0]],[p1[1], p11[1]], color=bondLL, linewidth=3, solid_capstyle='round', solid_joinstyle='round')
                        if (np.sign(nij[0]) != np.sign((p2 - p1)[0])) or (np.sign(nij[1]) != np.sign((p2 - p1)[1])):
                            p22 = p2 - rij
                            ax.plot([p2[0], p22[0]], [p2[1], p22[1]], color=bondLL, linewidth=3, solid_capstyle='round', solid_joinstyle='round')

                ax.set_xlim([-(newLx/2+0.2),(newLx/2+0.2)])
                ax.set_ylim([-(newLz/2+0.2),(newLz/2+0.2)])
                ax.axis('off')
                ax.set_aspect('equal')
                
                directory = f'{fig_save_path}phi_{phii}_ar_{ar[k]}_vr_{vr}'
                os.makedirs(directory, exist_ok=True)
                fig.savefig(f'{directory}/{frame}.png', dpi=600, transparent=True)
                print(f'>     Processed frame: {frame}/{endFrame-1}      ')
                plt.close()
        else:
            print(f"{dataname} - Not Found")