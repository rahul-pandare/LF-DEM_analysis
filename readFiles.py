import numpy as np # type: ignore
import random
import ctypes
'''
Mar 9 , 2025 RVP - File created
May 30, 2025 RVP - Major error correction.
interaction, rigid and particle list did not append the last strain
as a result the total snapshots were one less in dimensions.

NOTE:Script contains functions to read LF-DEM simulation files

'''

def rigClusterList(rigFile):
    '''
    This function reads the rig_*.dat and creates a list of particle index 
    in rigid cluster. Each element in a list is a str with rigid particle index
    from a cluster. 

    NOTE: The list elements may have repeated index numbers. Filter before
    processing.

    Inputs: rigFile - open(path/to/rig_* file)
    Output: len(output) = no. of clusters
            output sample: ['0','0','0',..., '210,600,550,600', '224,224,948,775,948']
    '''
    hashCounter = -4
    clusterIDs  = []
    for line in rigFile:
        if line[0] == '#':
            hashCounter += 1
        elif hashCounter >= 0:
            clusterIDs.append(line.strip())
    return clusterIDs

def rigList(rigFile):
    '''
    This function reads the rig_*.dat and creates a nested-list of particle index 
    in rigid cluster. Each element in a list (for each timestep) is a nested 
    list (for each cluster) with rigid particle index. 

    Input : open(path/to/rig_*.dat file)
    Output: [[[0]], [[97, 235, 97], [174, 201, 488]], ...,[[381, 235, 381]]]
            len(output) = no. of timesteps

    NOTE: The list elements may have repeated index numbers. Filter before processing.
    '''
    hashCounter = -4
    clusterIDs  = []
    temp = []
    for line in rigFile:
        if line[0] == '#':
            hashCounter += 1
            if len(temp) > 0:
                clusterIDs.append(temp)
                temp = []
        elif hashCounter >= 0:
            temp.append(line.strip())
    if temp:
        clusterIDs.append(temp)

    rigClusterIDsList = []
    for sampleList in clusterIDs:
        tempList = []
        for kk in range(len(sampleList)):
            tempList.append([int(indx) for indx in sampleList[kk].split(',')])
        rigClusterIDsList.append(tempList)
    return rigClusterIDsList

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
    This function reads the interaction file and creates a list of arrays.
    Each array contains interaction parameters for a timestep

    Input : interactionFile - open(path/to/int_* file)
    Output: len(output) = no. of timesteps
            output sample: [arr(t0), arr(t1)...., arr(tn)]
            arr(tn).shape = (n, 17) where n is the no. of interactions in a timestep
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

    if temp:
        contactList.append(np.array(temp))

    interactionFile.close()
    return contactList

def readParFile(particleFile):
    '''
    This function reads the particle file (contains particle wise information) and creates a list of arrays,
    each array contains all individual particle parameters for that timestep.

    NOTE: this function is same as readParFile2() function. Structed differently
    to set move the internal file pointer back to the required file.

    Input : ParametersFile - open(path/to/particle file)
    Output: output = [arr(t0), arr(t1)...., arr(tn)]
            len(output) = total snapshots 
            arr(tn).shape = (npp, 11) where npp is the no. of particles in system
           
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

    # to append last the strain data
    if temp:
        particlesList.append(np.array(temp))

    particleFile.close()
    return particlesList

def readParFile2(particleFile):
    ## formerly parametersList()
    '''
    This function reads the particle file (contains particle wise information) and creates a list of arrays,
    each array contains all individual particle parameters for that timestep.

    NOTE: This version of reading the particle file sets the pointer and opens the file automatically

    Input : ParametersFile - path/to/particle file
    Output: output = [arr(t0), arr(t1)...., arr(tn)]
            len(output) = total snapshots 
            arr(tn).shape = (npp, 11) where npp is the no. of particles in system
    '''

    parFile = open(particleFile, 'r')

    hashCounter = 0
    temp        = []
    parList     = [] # list with parameters parameters for each element at each timestep

    fileLines = parFile.readlines()[22:] # skipping the comment lines
    for line in fileLines:
        if not line.split()[0] == '#':
            lineList = [float(value) for value in line.split()]
            temp.append(lineList)
        else:
            hashCounter += 1 # checking if counter reaches 7 (7 lines of comments after every timestep data)
            if hashCounter == 7: 
                parList.append(np.array(temp))
                temp        = []
                hashCounter = 0

    if temp:
        parList.append(np.array(temp))

    parFile.close()
    return parList

def readParFile3(particleFile, final_strain = 20):
    '''
    This function reads the particle file (contains particle wise information) and creates a list of arrays,
    each array contains all individual particle parameters for that timestep.

    Input: path/to/particle file
    Output: output = [arr(t0), arr(t1)...., arr(tn)]
            len(output) = total snapshots 
            arr(tn).shape = (npp, 11) where npp is the no. of particles in system
    '''
    particleFile.seek(0)
    data    = np.loadtxt(particleFile, comments='#')    # reading the file excluding comments
    parList = np.split(data, int((final_strain)*100)+1) # splitting the data into the snapshots
    particleFile.close()

    return parList


def free_mem(*arrays):
    '''
    To free up memory instead of just using del
    efficacy not tested yet
    '''
    for arr in arrays:
        if isinstance(arr, np.ndarray):
            ctypes.memset(arr.ctypes.data, 0, arr.nbytes)