# This script is for plotting the histogram of angular contacts
# the contact angles go from -pi to pi
# histogram of each set contains contacts from all the 4 runs (in case NP = 1000) and not average

import os
import matplotlib                   # type: ignore
import numpy             as     np  # type: ignore
import matplotlib.pyplot as     plt # type: ignore
import random

# commands to run in terminal: 
# All contacts for a phi for all delta: 
# python3 -c "from angularContacts import angularContacts; angularContacts(0.70, True)"

# Angular contacts for all different size pairs for a phi and delta value: 
# python3 -c "from angularContacts import angularContactsAllSizePair; angularContactsAllSizePair(0.75, 1.4, True)"

# Angular contacts for a specific size pair for a phi value and all delta:
# python3 -c "from angularContacts import angularContactsSizePair; angularContactsSizePair(0.75, 'ss', True)"

# Matplotlib rc parameters modification
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

plt.rcParams['text.latex.preamble']= r"\usepackage{amsmath}"

# script for histogram of angular contacts
topDir      = "/media/rahul/Rahul_2TB/high_bidispersity/"

ranSeedFile = "random_seed.dat"
intFile     = "int_random_seed_params_stress100r_shear.dat"
contFile    = 'contacts.txt'

NP          = [1000]

run         = {500:8,1000:4,2000:2,4000:1}

ar          = [1.0, 1.4, 1.8, 2.0, 4.0]


# angularContacts plot the histogram for total contacts for all deltas for one phi value
def angularContacts(phii, figsave = False):
    plt.clf()
    cmap = matplotlib.colormaps['viridis_r'] #color scheme
    
    npp = 1000
    off = 100

    phii = '{:.3f}'.format(phii) if len(str(phii).split('.')[1])>2 else '{:.2f}'.format(phii)
    for k in range(len(ar)):
        contactAngleAllRuns = []
        dataname            = topDir+'NP_'+str(npp)+'/phi_'+phii+'/ar_'+str(ar[k])+'/Vr_0.5'
        if os.path.exists(dataname):
            for l in range (run[npp]):
                interFile   = open(f'{dataname}/run_{l+1}/{intFile}', 'r')
                contactList = interactionsList(interFile)

                # identifying every frictional contact and computing the contact angle

                contactAngle = [] # all contact angles for a case and one run
                for sampleList in contactList:
                    timeStepContactAngle = [] # all contact angles at each timestep
                    for i in range (sampleList.shape[0]):
                        if int(sampleList[i,10]) == 2:
                            thetaRad = np.arctan2(sampleList[i,4],sampleList[i,2]) # contact angle = tan^-1(nz/nx) ; nz, nx: unit vector in z and x
                            timeStepContactAngle.append(thetaRad)
                            if thetaRad < 0:
                                timeStepContactAngle.append(np.pi - abs(thetaRad))
                            else:
                                timeStepContactAngle.append(thetaRad - np.pi)

                    contactAngle.append(timeStepContactAngle) # nested list of all contact angles

                contactAngleList = [i for sub in contactAngle[off:] for i in sub] # unzipping the nested list into a long single list
                contactAngleAllRuns.append(contactAngleList)

            allContactAngles  = [i for sub in contactAngleAllRuns for i in sub] # unzipping nested list for all 4 runs
            counts, bin_edges = np.histogram(allContactAngles, bins = 72, density = True)
            bin_centers       = (bin_edges[:-1] + bin_edges[1:]) / 2
            plt.plot(bin_centers, counts, 'o', markersize = 3, label = r'$\delta$' + f' = {ar[k]}', color = cmap((k+1)/len(ar))) 

    xticks       = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    xtick_labels = [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$']
    plt.xticks(xticks, xtick_labels)
    plt.title(fr"$\phi = {phii}$", fontsize = 18)
    plt.xlabel(r'$\theta$', fontsize = 14,fontstyle = 'italic')
    plt.ylabel(r'$E_c$', fontsize = 14,fontstyle = 'italic')

    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    #plt.legend(fontsize = 14)
    plt.legend(fontsize = 10, labelspacing = 1, borderpad = 1)

    plt.grid(which = 'Both', alpha = 0.2)

    plt.tight_layout()
    if figsave:
        figFormat     =".pdf"
        #fig_save_path = "/home/rahul/Dropbox (City College)/CUNY/Research/Bidisperse Project/analysis/figures/angular_Contacts"
        fig_save_path = "/media/Linux_1TB/Dropbox (City College)/CUNY/Research/Bidisperse Project/analysis/figures/angular_Contacts"
        plt.savefig(fig_save_path + '/angularContacts_NP_' + str(1000) + '_phi_' + str(phii) + figFormat, bbox_inches = "tight", dpi = 500)
    #plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Function below (angularContactsAllSizePair) plots for histogram of 
# angular contacts for all size pairs -- small-small, small-large and large-large for one phi and one delta value

def angularContactsAllSizePair(phii, arr, figsave = False):

    plt.clf()
    cmap = matplotlib.colormaps['viridis_r'] #color scheme

    npp = 1000 # no. of particles
    off = 100  # steady state cut off

    phii = '{:.3f}'.format(phii) if len(str(phii).split('.')[1])>2 else '{:.2f}'.format(phii)

    contactAngleAllRuns   = [[] for _ in range(run[npp])] # empty nested list for all four runs 
    sizePair              = ['Total', 'SS', 'SL', 'LL']
    
    dataname = topDir+'NP_'+str(npp)+'/phi_'+phii+'/ar_'+str(arr)+'/Vr_0.5'
    if os.path.exists(dataname):
        for l in range (run[npp]):
            ranFile = open(f'{dataname}/run_{l+1}/{ranSeedFile}', 'r')

            if arr == 1:
                particleSize = [1]*(int(npp/2)) + [2]*(int(npp/2))
                random.shuffle(particleSize)
                #painting particles randomly in two colours for monodisperse case
            else:
                particleSize = np.loadtxt(ranFile,usecols = 3) # reading only column 3 which has particle size
                ranFile.close()

            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            # reading interaction file
            # storing the interactions for each timestep in a list
            hashCounter = 0
            temp        = []
            contactList = [] # list with interaction parameters for each element at each timestep

            interFile = open(f'{dataname}/run_{l+1}/{intFile}', 'r')
            fileLines = interFile.readlines()[27:] # skipping the comment lines
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
            interFile.close()

            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            # total angluar contacts

            contactAngleTotal = [] # all contact angles for a case and one run
            for sampleList in contactList:
                timeStepContactAngle = [] # all contact angles at each timestep
                for i in range (sampleList.shape[0]):
                    if int(sampleList[i,10]) == 2:
                        thetaRad = np.arctan2(sampleList[i,4], sampleList[i,2]) # contact angle = tan^-1(nz/nx) ; nz, nx: unit vector in z and x
                        timeStepContactAngle.append(thetaRad)
                        if thetaRad < 0:
                            timeStepContactAngle.append(np.pi - abs(thetaRad))
                        else:
                            timeStepContactAngle.append(thetaRad - np.pi)

                contactAngleTotal.append(timeStepContactAngle) # nested list of all contact angles

            contactAngleList = [i for sub in contactAngleTotal[off:] for i in sub]
            contactAngleAllRuns[0].append(contactAngleList)

            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            # small-small angluar contacts

            contactAngleTotal = []
            for sampleList in contactList:
                timeStepContactAngle = [] 
                for i in range (sampleList.shape[0]):
                    if (particleSize[int(sampleList[i,0])] == particleSize[int(sampleList[i,1])] == 1) and int(sampleList[i,10]) == 2:
                        thetaRad = np.arctan2(sampleList[i,4], sampleList[i,2])
                        timeStepContactAngle.append(thetaRad)
                        if thetaRad < 0:
                            timeStepContactAngle.append(np.pi - abs(thetaRad))
                        else:
                            timeStepContactAngle.append(thetaRad - np.pi)

                contactAngleTotal.append(timeStepContactAngle) # nested list of all contact angles

            contactAngleList = [i for sub in contactAngleTotal[off:] for i in sub]
            contactAngleAllRuns[1].append(contactAngleList)

            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            # small-large angluar contacts

            contactAngleTotal = [] 
            for sampleList in contactList:
                timeStepContactAngle = [] 
                for i in range (sampleList.shape[0]):
                    if (particleSize[int(sampleList[i, 0])] != particleSize[int(sampleList[i, 1])]) and int(sampleList[i, 10]) == 2:
                        thetaRad = np.arctan2(sampleList[i,4], sampleList[i,2])
                        timeStepContactAngle.append(thetaRad)
                        if thetaRad < 0:
                            timeStepContactAngle.append(np.pi - abs(thetaRad))
                        else:
                            timeStepContactAngle.append(thetaRad - np.pi)

                contactAngleTotal.append(timeStepContactAngle) # nested list of all contact angles

            contactAngleList = [i for sub in contactAngleTotal[off:] for i in sub]
            contactAngleAllRuns[2].append(contactAngleList)

            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            # large-large angluar contacts

            contactAngleTotal = [] 
            for sampleList in contactList:
                timeStepContactAngle = [] 
                for i in range (sampleList.shape[0]):
                    if (particleSize[int(sampleList[i,0])] == particleSize[int(sampleList[i,1])] > 1) and int(sampleList[i,10]) == 2:
                        thetaRad = np.arctan2(sampleList[i,4], sampleList[i,2])
                        timeStepContactAngle.append(thetaRad)
                        if thetaRad < 0:
                            timeStepContactAngle.append(np.pi - abs(thetaRad))
                        else:
                            timeStepContactAngle.append(thetaRad - np.pi)

                contactAngleTotal.append(timeStepContactAngle) # nested list of all contact angles

            contactAngleList = [i for sub in contactAngleTotal[off:] for i in sub]
            contactAngleAllRuns[3].append(contactAngleList)

    # Loop for plotting each size pair 
    for ii in range(len(contactAngleAllRuns)):
        allContactAngles   = [i for sub in contactAngleAllRuns[ii] for i in sub]
        counts, bin_edges  = np.histogram(allContactAngles, bins = 72, density = True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(bin_centers, counts, 'o', markersize = 3, label = str(sizePair[ii]), color = cmap((ii+1)/len(sizePair)))

    xticks       = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    xtick_labels = [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$']
    plt.xticks(xticks, xtick_labels)
    plt.title(fr"$\phi = {phii},  \delta = {arr}$", fontsize = 18)
    plt.xlabel(r'$\theta$', fontsize = 14, fontstyle = 'italic')
    plt.ylabel(r'$E_c$', fontsize = 14, fontstyle = 'italic')

    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    #plt.legend(fontsize = 14)
    plt.legend(fontsize = 10, labelspacing = 1, borderpad = 1)

    plt.grid(which = 'Both', alpha = 0.2)

    plt.tight_layout()
    if figsave:
        figFormat     = ".pdf"
        #fig_save_path = "/home/rahul/Dropbox (City College)/CUNY/Research/Bidisperse Project/analysis/figures/angular_Contacts"
        fig_save_path = "/media/Linux_1TB/Dropbox (City College)/CUNY/Research/Bidisperse Project/analysis/figures/angular_Contacts"
        plt.savefig(fig_save_path + '/angularContactsAllSizePair_NP_' + str(1000) + '_phi_' + str(phii) + '_ar_' + str(arr) + figFormat, bbox_inches = "tight", dpi = 500)

    #plt.show()

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ##



# (angularContactsSizePair) script for histogram of angular contacts for a size pair small-small/ small-large/ large-large for one phi value
def angularContactsSizePair(phii, sizePair, figsave = False):

    '''
    The input sizePair can be one of the following strings: 'total', 'ss', 'sl' or 'll'
    '''
    plt.clf()
    cmap = matplotlib.colormaps['viridis_r'] #color scheme

    npp = 1000
    off = 100

    phii = '{:.3f}'.format(phii) if len(str(phii).split('.')[1])>2 else '{:.2f}'.format(phii)
    
    for k in range(len(ar)):
        contactAngleAllRuns = []
        dataname = topDir+'NP_'+str(npp)+'/phi_'+phii+'/ar_'+str(ar[k])+'/Vr_0.5'
        if os.path.exists(dataname):
            for l in range (run[npp]):
                ranFile = open(f'{dataname}/run_{l+1}/{ranSeedFile}', 'r')

                if ar[k] == 1:
                    particleSize = [1]*(int(npp/2)) + [2]*(int(npp/2))
                    random.shuffle(particleSize)
                    #painting particles randomly in two colours for monodisperse case
                else:
                    particleSize = np.loadtxt(ranFile, usecols = 3) # reading only column 3 which has particle size
                    ranFile.close()

                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                # reading interaction file
                # storing the interactions for each timestep in a list
                hashCounter = 0
                temp        = []
                contactList = [] # list with interaction parameters for each element at each timestep

                interFile = open(f'{dataname}/run_{l+1}/{intFile}', 'r')
                fileLines = interFile.readlines()[27:] # skipping the comment lines
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
                interFile.close()

                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                contactAngleTotal = [] # total number of particles in frictional contacts
                for sampleList in contactList:
                    timeStepContactAngle = []  # index of particles in contact

                    for i in range(sampleList.shape[0]):
                        particleSize1 = particleSize[int(sampleList[i, 0])]
                        particleSize2 = particleSize[int(sampleList[i, 1])] # 
                        contState     = int(sampleList[i, 10]) # contact state

                        if contState == 2:
                            thetaRad = np.arctan2(sampleList[i, 4], sampleList[i, 2])
                            if sizePair   == 'total':
                                updateContAngle(thetaRad, timeStepContactAngle)
                                plotTitle = 'Total Contacts'

                            elif sizePair == 'ss' and particleSize1 == particleSize2 == 1:
                                updateContAngle(thetaRad, timeStepContactAngle)
                                plotTitle = 'Small - Small Contacts'

                            elif sizePair == 'sl' and particleSize1 != particleSize2:
                                updateContAngle(thetaRad, timeStepContactAngle)
                                plotTitle = 'Small - Large Contacts'

                            elif sizePair == 'll' and particleSize1 == particleSize2 > 1:
                                updateContAngle(thetaRad, timeStepContactAngle)
                                plotTitle = 'Large - Large Contacts'      

                    contactAngleTotal.append(timeStepContactAngle) # nested list of all contact angles

                contactAngleList = [i for sub in contactAngleTotal[off:] for i in sub]
                contactAngleAllRuns.append(contactAngleList)

        # Histogram plotting
        allContactAngles   = [i for sub in contactAngleAllRuns for i in sub]
        counts, bin_edges  = np.histogram(allContactAngles, bins = 72, density = True)
        bin_centers        = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(bin_centers, counts, 'o', markersize = 3, label = r'$\delta$' + f' = {ar[k]}',color = cmap((k+1)/len(ar)))

    xticks       = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    xtick_labels = [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$']
    plt.xticks(xticks, xtick_labels)
    plt.title(fr"$\phi = {phii}$, {plotTitle}", fontsize = 18)
    plt.xlabel(r'$\theta$', fontsize = 14, fontstyle = 'italic')
    plt.ylabel(r'$E_c$', fontsize = 14, fontstyle = 'italic')

    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    #plt.legend(fontsize = 14)
    plt.legend(fontsize = 10, labelspacing = 1, borderpad = 1)

    plt.grid(which = 'Both', alpha = 0.2)

    plt.tight_layout()
    if figsave:
        figFormat     = ".pdf"
        #fig_save_path = "/home/rahul/Dropbox (City College)/CUNY/Research/Bidisperse Project/analysis/figures/angular_Contacts"
        fig_save_path = "/media/Linux_1TB/Dropbox (City College)/CUNY/Research/Bidisperse Project/analysis/figures/angular_Contacts"
        plt.savefig(fig_save_path+'/angularContactsSizePair_' + sizePair + '_NP_' + str(1000) + '_phi_' + str(phii) + figFormat, bbox_inches = "tight", dpi = 500)

    #plt.show()


"====================================================================================================================================="

# Below are the functions used in the above histogram functions. Esessentially to shorten it.

# function to calulated the complementary contact angle and append to relevant list
def updateContAngle(theta, contList):
    '''
    This function to calulated the complementary contact angle and append to 
    relevant list (contList)

    Inputs:
    theta    - primary contact angle. Calculated by arctan2(nz,nx)
    contList - list where the angle appends. This list contains all the contact angle for one case, all runs
    '''
    contList.append(theta)
    if theta < 0:
        contList.append(np.pi - abs(theta))
    else:
        contList.append(theta - np.pi)

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