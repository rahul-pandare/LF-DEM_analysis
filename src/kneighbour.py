import os
import glob
import numpy             as np  # type: ignore
import matplotlib.pyplot as plt # type: ignore
import src.readFiles as readFiles # type: ignore
import sys
'''
Jan 29, 2026 RVP
Functions for analysing kneighbors and spanning k-neighbor clusters.
'''

# Particles data file
parFile = 'par_*.dat' 
intFile = 'int_*.dat'
rigFile = 'rig_*.dat'

def kneighborfig(dataname, frame, k=3):
    '''
    Make snapshots highighting particles with at least k contacts.
    global variables: phi, stress, npp, ar
    '''
    _, ax  = plt.subplots(1, 1, figsize=(5,5))
    
    parPath  = glob.glob(f'{dataname}/{parFile}')[0]
    parLines = open(parPath).readlines()

    # box dimensions
    Lx = float(parLines[3].split()[2]) 
    Lz = float(parLines[3].split()[2])
    
    newLx = Lx + 2*ar
    newLz = Lz + 2*ar
    
    parList   = readFiles.readParFile(open(parPath))
    intPath   = open(glob.glob(f'{dataname}/{intFile}')[0])
    intList   = readFiles.interactionsList(intPath)
    frameList = intList[frame]

    p1 = frameList[:,0].astype(int)
    p2 = frameList[:,1].astype(int)
    contState = frameList[:,10]

    contMat = np.zeros((npp, npp), dtype=np.uint8)
    mask    = contState == 2
    contMat[p1[mask], p2[mask]] = 1
    contMat[p2[mask], p1[mask]] = 1
    
    px = parList[frame][:, 2]
    pz = parList[frame][:, 3]
    pr = parList[frame][:, 1]
    
    totContacts = contMat.sum(axis=0)
    
    # plotting all particles 
    for i in range(len(px)):
        if totContacts[i] >= k:
            circle = plt.Circle((px[i],pz[i]), pr[i], facecolor='#A00000', fill=True, edgecolor='none') #083d5f
        else:
            circle = plt.Circle((px[i],pz[i]), pr[i], facecolor='#8A9BA8', fill=True, edgecolor='none')
        ax.add_artist(circle)
        
    ax.set_xlim([-(newLx/2+0.2),(newLx/2+0.2)])
    ax.set_ylim([-(newLz/2+0.2),(newLz/2+0.2)])
    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_title(fr'$\mathbf{{\phi = {phi},\; \sigma/\sigma_0 = {stress},\; \gamma = {frame/100:.2f},\; k \geq 3}}$',
                 fontsize=16, pad=1, fontweight='bold', x=0.5)
    
def makerigsnapshot(dataname, frame):
    '''
    Make snapshots highighting rigid particles.
    global variables: phi, stress, npp
    '''
    _, ax  = plt.subplots(1, 1, figsize=(5,5))
    
    if os.path.exists(dataname):
        parPath  = glob.glob(f'{dataname}/{parFile}')[0]
        parLines = open(parPath).readlines()
        parList  = readFiles.readParFile(open(parPath))

        rigFilePath   = glob.glob(f'{dataname}/{rigFile}')
        if not rigFilePath:
            print(f"Error: {rigFile} not found at {dataname}")
            sys.exit(1)
        rigidFile     = open(rigFilePath[0])
        rigClusterIDs = readFiles.rigList(rigidFile)
        clusterIDs    = [[np.nan] if len(samplelist[0]) < 2 else list({int(num) for sublist in samplelist for num in sublist}) for samplelist in rigClusterIDs]

        # Box dimensions.
        Lx = float(parLines[3].split()[2]) 
        Lz = float(parLines[3].split()[2])

        px = parList[frame][:,2]
        pz = parList[frame][:,3]
        pr = parList[frame][:,1]

        # Setting up axis and box walls.
        newLx   = Lx + 2*np.max(pr)
        newLz   = Lz + 2*np.max(pr)

        allPart      = parList[frame][:,0]
        rigidPart    = clusterIDs[frame]
        notRigidPart = allPart[np.isin(allPart, rigidPart) == False]

        ax.clear()
        for index in notRigidPart:
            index  = int(index)
            circle = plt.Circle((px[index],pz[index]), pr[index], facecolor='w', edgecolor='k', linewidth=0.25, zorder=1)
            ax.add_artist(circle)
        
        for index in rigidPart:
            circle = plt.Circle((px[index],pz[index]), pr[index], facecolor='#A00000', edgecolor=None, zorder=2)
            ax.add_artist(circle)

        ax.set_xlim([-(newLx/2),(newLx/2)])
        ax.set_ylim([-(newLz/2),(newLz/2)])
        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_title(fr'$\mathbf{{\phi = {phi},\; \sigma/\sigma_0 = {stress},\; \gamma = {frame/100:.2f}}}$', 
                     fontsize=17, pad=1, fontweight='bold', x=0.5)

        #plt.tight_layout()
        
def kclusters(contmatrix, frame, k=3):
    '''
    Identifys k-neighbor clusters from contact matrix.
    Returns list of sets, each set is a k-neighbor cluster.
    Input: contact matrix (numpy array: NxN), frame number (int), k (int)
    '''
    
    cordination = contMat.sum(axis=1)
    #neighbors   = {i: np.where(contMat[i] == 1)[0] for i in range(contMat.shape[0])}
    knodes      = set(np.where(cordination >= k)[0])
    neighbors   = {i: [j for j in np.where(contMat[i] == 1)[0] 
                   if j in knodes] for i in knodes}
    
    visited  = set()
    clusters = []

    for node in knodes:
        if node not in visited:
            stack = [node]
            cluster = set()

            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    cluster.add(current)
                    stack.extend(neighbors[current])

            clusters.append(cluster)
            
    return clusters

def clusterSpan(positions, box_length):
    '''
    Identifies the span of a cluster in periodic boundaries.
    '''
    ref = positions[0]
    shifted = (positions - ref + box_length/2) % box_length - box_length/2
    return shifted.max() - shifted.min()