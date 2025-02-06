import os
import matplotlib                   # type: ignore
import numpy             as     np  # type: ignore
import matplotlib.pyplot as     plt # type: ignore
import random
matplotlib.use('TkAgg')

"""
July 16, 2024
RVP

This script is to plot the force histogram wrt angular contact for different simulation parameters.
The functions here can plot angular force histograms for different size-pairs in the system

commands to run in terminal: 
All angular forces for a phi for all delta: 
python3 -c "from angularForce import angularForces; angularForces(0.70, True)"

Angular forces for all different size pairs for a phi and delta value: 
python3 -c "from angularForce import angularForcesAllSizePair; angularForcesAllSizePair(0.75, 1.4, True)"

Angular forces for a specific size pair for a phi value and all delta:
python3 -c "from angularForce import angularForcesSizePair; angularForcesSizePair(0.75, 'ss', True)"

Angular forces on a specific size for a phi value and all delta:
python3 -c "from angularForce import angularForcesOnSize; angularForcesOnSize(0.75, 'onsmall', True)"
"""

# Matplotlib rc parameters modification.
plt.rcParams.update({
  "figure.max_open_warning" : 0,
  "text.usetex"             : True,
  "text.latex.preamble"     : r"\usepackage{type1cm}",
  "figure.autolayout"       : True,
  "font.family"             : "STIXGeneral",
  "mathtext.fontset"        : "stix",
  "font.size"               : 8,
  "xtick.labelsize"         : 12,
  "ytick.labelsize"         : 12,
  "lines.linewidth"         : 1,
  "lines.markersize"        : 5
})

plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"

# Simulation data mount point and figure save path.
topDir        = "/media/rahul/Rahul_2TB/high_bidispersity/"
fig_save_path = "/home/rahul/Dropbox (City College)/CUNY/Research/Bidisperse Project/analysis/figures/angular_Force"

# Relevant file names to read.
ranSeedFile = "random_seed.dat"
intFile     = "int_random_seed_params_stress100r_shear.dat"
contFile    = 'contacts.txt'

# Some simulation parameters.
run         = {500:8, 1000:4, 2000:2, 4000:1}

ar          = [1.0, 1.4, 1.8, 2.0, 4.0]

def angularForces(phii, figsave = False):
    '''
    This function plots the sum of forces in a particular angular bin for all particles in the system and all ar values.
    Plots is fc/f0 vs theta, with one curve for each ar value for one phi value.

    input: phii - value for one single 
    '''

    plt.clf()
    cmap = matplotlib.colormaps['viridis_r'] #color scheme

    npp = 1000
    off = 100

    phii = '{:.3f}'.format(phii) if len(str(phii).split('.')[1]) > 2 else '{:.2f}'.format(phii)

    angleBins  = np.linspace(-np.pi, np.pi,72)
    binCenters = (angleBins[:-1] + angleBins[1:])/2
    
    contCount = 0
    for k in range(len(ar)):
        forceBin_Sum = [0]*len(binCenters)
        dataname     = topDir+'NP_'+str(npp)+'/phi_'+phii+'/ar_'+str(ar[k])+'/Vr_0.5'
        if os.path.exists(dataname):
            for l in range (run[npp]):
                interFile   = open(f'{dataname}/run_{l+1}/{intFile}', 'r')
                contactList = interactionsList(interFile)
                for sampleList in contactList[off:]:
                    for i in range (sampleList.shape[0]):
                        if int(sampleList[i,10]) == 2:
                            contForce = sampleList[i,11]
                            thetaRad  = np.arctan2(sampleList[i,4],sampleList[i,2])
                            updateContForce(thetaRad, binCenters, forceBin_Sum, contForce)
                            contCount += 2

        forceAvg     = np.mean(forceBin_Sum)
        forceDensity = [i/forceAvg for i in forceBin_Sum]
        plt.plot(binCenters, forceDensity, 'o', markersize = 3, label = r'$\delta$' + f' = {ar[k]}',color = cmap((k+1)/len(ar))) 

    xticks       = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    xtick_labels = [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$']
    plt.xticks(xticks, xtick_labels)
    plt.title(fr"$\phi = {phii}$", fontsize = 18)
    plt.xlabel(r'$\theta$',  fontsize = 14, fontstyle = 'italic')
    plt.ylabel(r'$f_c/f_0$', fontsize = 14, fontstyle = 'italic')

    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.legend(fontsize = 14)
    plt.legend(fontsize = 10, labelspacing = 1, borderpad = 1)

    plt.grid(which = 'Both', alpha = 0.2)

    plt.tight_layout()
    if figsave:
        figFormat     = ".pdf"
        plt.savefig(fig_save_path + '/angularForces_NP_' + str(npp) + '_phi_' + str(phii) + figFormat, bbox_inches = "tight", dpi = 500)

    #plt.show()

"====================================================================================================================================="

def angularForcesAllSizePair(phii, arr, figsave = False):
    '''
    This function plots angular forces for one phi value and ar value for all size pairs.
    The plots would be fc/f0 vs theta, it will have 4 curves ( 3 for different size pairs and 1 total).

    Inputs:
    phii - phi value
    arr  - ar value
    '''

    plt.clf()
    cmap = matplotlib.colormaps['viridis_r'] #color scheme

    npp = 1000 # no. of particles
    off = 100  # steady state cut off

    angleBins  = np.linspace(-np.pi, np.pi,72)
    binCenters = (angleBins[:-1] + angleBins[1:])/2

    phii = '{:.3f}'.format(phii) if len(str(phii).split('.')[1]) > 2 else '{:.2f}'.format(phii)

    forceBin_Sum = [[0]*len(binCenters) for _ in range(run[npp])] # empty nested list for all four runs 
    contCount    = [0]*run[npp]
    sizePair     = ['Total', 'SS', 'SL', 'LL']
    
    dataname     = topDir + 'NP_' + str(npp) + '/phi_' + phii + '/ar_' + str(arr) + '/Vr_0.5'
    if os.path.exists(dataname):
        for l in range (run[npp]):
            # List of particle size. List index is particle index.
            ranFile      = open(f'{dataname}/run_{l+1}/{ranSeedFile}', 'r')
            particleSize = particleSizeList(ranFile, sizeRatio = arr) 

            # Nested list of all interaction params, each list is one timestep.
            interFile   = open(f'{dataname}/run_{l+1}/{intFile}', 'r')
            contactList = interactionsList(interFile) 
           
            for sampleList in contactList[off:]:
                for i in range (sampleList.shape[0]):
                    particleSize1 = particleSize[int(sampleList[i,0])]
                    particleSize2 = particleSize[int(sampleList[i,1])]
                    contState     = int(sampleList[i,10]) # Contact state
                    contForce     = sampleList[i,11] # Norm of normal contact force
                    thetaRad      = np.arctan2(sampleList[i,4], sampleList[i,2])

                    if contState == 2:
                        # For all particles.
                        updateContForce(thetaRad, binCenters, forceBin_Sum[0], contForce)
                        contCount[0] += 2
                        
                        # Small-Small.
                        if (particleSize1 == particleSize2 == 1):
                            updateContForce(thetaRad, binCenters, forceBin_Sum[1], contForce)
                            contCount[1] += 2

                        # Small-Large
                        elif (particleSize1 != particleSize2):
                            updateContForce(thetaRad, binCenters, forceBin_Sum[2], contForce)
                            contCount[2] += 2

                        # Large-Large
                        elif (particleSize1 == particleSize2 > 1):
                            updateContForce(thetaRad, binCenters, forceBin_Sum[3], contForce)
                            contCount[3] += 2

    # Loop for plotting each size pair 
    for ii in range(len(forceBin_Sum)):
        forceAvg     = np.mean(forceBin_Sum[ii])
        forceDensity = [i/forceAvg for i in forceBin_Sum[ii]]
        #forceDensity = [i/contCount[ii]/binWidth for i in forceBin_Sum[ii]]
        plt.plot(binCenters, forceDensity, 'o', markersize = 3, label = str(sizePair[ii]), color = cmap((ii+1)/len(ar))) 

    xticks       = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    xtick_labels = [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$']
    plt.xticks(xticks, xtick_labels)
    plt.title(fr"$\phi = {phii},  \delta = {arr}$", fontsize = 18)
    plt.xlabel(r'$\theta$',  fontsize = 14, fontstyle = 'italic')
    plt.ylabel(r'$f_c/f_0$', fontsize = 14, fontstyle = 'italic')

    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.legend(fontsize = 10, labelspacing = 1, borderpad = 1)

    plt.grid(which = 'Both', alpha = 0.2)

    plt.tight_layout()
    if figsave:
        figFormat     = ".pdf"
        plt.savefig(fig_save_path + '/angularForcesAllSizePair_NP_' + str(npp) + '_phi_' + str(phii) + '_ar_' + str(arr) + figFormat, bbox_inches = "tight", dpi = 500)

    #plt.show()

"====================================================================================================================================="

# (angularContactsSizePair) script for histogram of angular contacts for a size pair small-small/ small-large/ large-large for one phi value
def angularForcesSizePair(phii, sizePair, figsave = False):
    '''
    This function is for plotting histogram of angular contacts for a size pair small-small/ small-large/ large-large 
    for one phi value.
    Plots would be fc/f0 vs theta, will contain curves for all ar values for a single size pair

    Inputs:
    phii     - phi value
    sizePair - can be one of the following strings: 'total', 'ss', 'sl' or 'll'
    '''

    plt.clf()
    cmap = matplotlib.colormaps['viridis_r'] # Color scheme

    npp = 1000 # No. of particles
    off = 100  # Steady state cut off

    angleBins  = np.linspace(-np.pi, np.pi,72)
    binCenters = (angleBins[:-1] + angleBins[1:])/2

    phii = '{:.3f}'.format(phii) if len(str(phii).split('.')[1]) > 2 else '{:.2f}'.format(phii)

    for k in range(len(ar)):
        dataname = topDir + 'NP_' + str(npp) + '/phi_' + phii + '/ar_' + str(ar[k]) + '/Vr_0.5'
        if os.path.exists(dataname):
            for l in range (run[npp]):
                ranFile      = open(f'{dataname}/run_{l+1}/{ranSeedFile}', 'r')
                particleSize = particleSizeList(ranFile, sizeRatio = ar[k])

                interFile   = open(f'{dataname}/run_{l+1}/{intFile}', 'r')
                contactList = interactionsList(interFile)

                forceBin_Sum = [0]*len(binCenters) # Total number of particles in frictional contacts
                contCount    = 0
                for sampleList in contactList[off:]:
                    for i in range(sampleList.shape[0]):
                        particleSize1 = particleSize[int(sampleList[i, 0])]
                        particleSize2 = particleSize[int(sampleList[i, 1])]
                        contForce     = sampleList[i,11] # Norm of normal contact force
                        contState     = int(sampleList[i, 10]) # Contact state

                        if contState == 2:
                            thetaRad = np.arctan2(sampleList[i, 4], sampleList[i, 2])
                            if sizePair   == 'total':
                                updateContForce(thetaRad, binCenters, forceBin_Sum, contForce)
                                contCount += 2
                                plotTitle = 'Total Contacts'

                            elif sizePair == 'ss' and particleSize1 == particleSize2 == 1:
                                updateContForce(thetaRad, binCenters, forceBin_Sum, contForce)
                                contCount += 2
                                plotTitle = 'Small - Small Contacts'

                            elif sizePair == 'sl' and particleSize1 != particleSize2:
                                updateContForce(thetaRad, binCenters, forceBin_Sum, contForce)
                                contCount += 2
                                plotTitle = 'Small - Large Contacts'

                            elif sizePair == 'll' and particleSize1 == particleSize2 > 1:
                                updateContForce(thetaRad, binCenters, forceBin_Sum, contForce)
                                contCount += 2
                                plotTitle = 'Large - Large Contacts'      

        # Histogram plotting
        forceAvg     = np.mean(forceBin_Sum)
        forceDensity = [i/forceAvg for i in forceBin_Sum]
        plt.plot(binCenters, forceDensity, 'o', markersize = 3, label = r'$\delta$' + f' = {ar[k]}', color = cmap((k+1)/len(ar))) 

    xticks       = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    xtick_labels = [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$']
    plt.xticks(xticks, xtick_labels)
    plt.title(fr"$\phi = {phii}$, {plotTitle}", fontsize = 18)
    plt.xlabel(r'$\theta$',  fontsize = 14, fontstyle = 'italic')
    plt.ylabel(r'$f_c/f_0$', fontsize = 14, fontstyle = 'italic')

    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.legend(fontsize = 10, labelspacing = 1, borderpad = 1)

    plt.grid(which = 'Both', alpha = 0.2)

    plt.tight_layout()
    if figsave:
        figFormat     = ".pdf"
        plt.savefig(fig_save_path+'/angularForcesSizePair_' + sizePair + '_NP_' + str(npp) + '_phi_' + str(phii) + figFormat, bbox_inches = "tight", dpi = 500)

    #plt.show()

"====================================================================================================================================="

def angularForcesOnSize(phii, onSize, figsave = False):
    '''
    This function is for plotting histogram of angular contact forces on a particular size particle - either small or large 
    for one phi value.
    Plots would be fc/f0 vs theta, will contain curves for all ar values for a single size.

    Inputs:
    phii     - phi value
    onSize - can be one of the following strings: 'onsmall' or 'onlarge'
    '''
    plt.clf()
    cmap = matplotlib.colormaps['viridis_r'] #color scheme

    npp = 1000
    off = 100

    angleBins  = np.linspace(-np.pi, np.pi,72)
    binCenters = (angleBins[:-1] + angleBins[1:])/2
    
    phii = '{:.3f}'.format(phii) if len(str(phii).split('.')[1])>2 else '{:.2f}'.format(phii)
    
    for k in range(len(ar)):
        dataname = topDir + 'NP_' + str(npp) + '/phi_' + phii + '/ar_' + str(ar[k]) + '/Vr_0.5'
        if os.path.exists(dataname):
            for l in range (run[npp]):
                ranFile      = open(f'{dataname}/run_{l+1}/{ranSeedFile}', 'r')
                particleSize = particleSizeList(ranFile, sizeRatio = ar[k])

                interFile   = open(f'{dataname}/run_{l+1}/{intFile}', 'r')
                contactList = interactionsList(interFile)

                forceBin_Sum = [0]*len(binCenters) # Total number of particles in frictional contacts
                contCount    = 0
                for sampleList in contactList[off:]:
                    for i in range(sampleList.shape[0]):
                        particleSize1 = particleSize[int(sampleList[i, 0])]
                        particleSize2 = particleSize[int(sampleList[i, 1])]
                        contForce     = sampleList[i,11] # Norm of normal contact force
                        contState     = int(sampleList[i, 10]) # Contact state

                        if contState == 2:
                            thetaRad = np.arctan2(sampleList[i, 4], sampleList[i, 2])
                            if onSize   == 'onsmall' and (particleSize1 == 1 or particleSize2 == 1):
                                updateContForce(thetaRad, binCenters, forceBin_Sum, contForce)
                                contCount += 2
                                plotTitle = 'On Small Contacts'

                            elif onSize == 'onlarge' and (particleSize1 > 1 or particleSize2 > 1):
                                updateContForce(thetaRad, binCenters, forceBin_Sum, contForce)
                                contCount += 2
                                plotTitle = 'On Large Contacts'

        # Histogram plotting
        forceAvg     = np.mean(forceBin_Sum)
        forceDensity = [i/forceAvg for i in forceBin_Sum]
        plt.plot(binCenters, forceDensity, 'o', markersize = 3, label = r'$\delta$' + f' = {ar[k]}', color = cmap((k+1)/len(ar))) 

    xticks       = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    xtick_labels = [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$']
    plt.xticks(xticks, xtick_labels)
    plt.title(fr"$\phi = {phii}$, {plotTitle}", fontsize = 18)
    plt.xlabel(r'$\theta$',  fontsize = 14, fontstyle = 'italic')
    plt.ylabel(r'$f_c/f_0$', fontsize = 14, fontstyle = 'italic')

    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.legend(fontsize = 10, labelspacing = 1, borderpad = 1)

    plt.grid(which = 'Both', alpha = 0.2)

    plt.tight_layout()
    if figsave:
        figFormat     = ".pdf"
        plt.savefig(fig_save_path+'/angularForcesOnSize_' + onSize + '_NP_' + str(1000) + '_phi_' + str(phii) + figFormat, bbox_inches = "tight", dpi = 500)

    #plt.show()

"====================================================================================================================================="

# Below are the functions used in the above histogram functions. Esessentially to shorten it.

def updateContForce(theta, binCentersList, forceList, forceVal):
    '''
    This function is used to calulated the complementary contact angle and 
    sum-append the force for that contact to relevant list

    Inputs:
    theta          - primary contact angle. Calculated by arctan2(nz,nx)
    binCentersList - centers of bins
    forceList      - list of total forces for each bin. The force appends here at the relevant bin
    forceVal       - the value of norm of the normal contact force to be sum-appended to 'forceList'
    '''
    
    bin_Center = int(np.floor(len(binCentersList)/2))
    bin_Index = np.argmin(np.abs(binCentersList - theta))
    forceList[bin_Index] += forceVal

    if bin_Index != bin_Center:
        bin_Index2 = (bin_Center + 1) + bin_Index if bin_Index < bin_Center else bin_Index - (bin_Center + 1)
        forceList[bin_Index2] += forceVal

def particleSizeList(randomSeedFile, sizeRatio, npp = 1000):
    '''
    This function reads the random seed file and creates
    a list of particle sizes. The list index is the particle index.

    Inputs:
    randomSeedFile - location for the random seed file. It contains the particle index and sizes
    sizeRatio      - delta or ar for the case
    npp            - system size
    '''

    if sizeRatio == 1:
        # Painting particles randomly in two colours for monodisperse case.
        particleSize = [1]*(int(npp/2)) + [2]*(int(npp/2))
        random.shuffle(particleSize)
    else:
        particleSize = np.loadtxt(randomSeedFile, usecols = 3) # reading only column 3 which has particle size
        randomSeedFile.close()
    return particleSize

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