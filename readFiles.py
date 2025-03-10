import numpy as np # type: ignore
import random
'''
Mar 9, 2025
RVP

Script contains functions to read LF-DEM simulation files
'''

def rigList(rigFile):
    '''
    This function reads the rig_*.dat and creates a list of particle index 
    in rigid cluster. Each element in a list is a str with rigid particle index
    from a cluster. 
    len(output) = no. of clusters
    output sample: ['0','0','0',..., '210,600,550,600', '224,224,948,775,948']

    NOTE: The list elements may have repeated index numbers. Filter before
    processing.

    Inputs:
    rigFile - rig_*.dat path
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
    This function reads the rig_*.dat and creates a list of particle index 
    in rigid cluster. Each element in a list (for each timestep) is a nested 
    list (for each cluster) with rigid particle index. 
    len(output) = no. of timesteps
    output sample: [[[0]], [[97, 235, 97], [174, 201, 488]], [[381, 235, 381]]]

    NOTE: The list elements may have repeated index numbers. Filter before
    processing.

    Inputs:
    rigFile - rig_*.dat path
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
            
    rigClusterIDsList = []
    for ii, sampleList in enumerate(clusterIDs):
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

def parametersList(ParametersFile):
    '''
    This function reads the parameters file and creates a nested-list,
    each list inside contains the array of all interaction parameters for
    that timestep.

    Input: ParametersFile - the location of the parameters data file
    '''

    parFile = open(ParametersFile, 'r')

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
    parFile.close()
    return parList

def readParFile(particleFile):
    '''
    This function reads the parameters file and creates a nested-list,
    each list inside contains the array of all interaction parameters for
    that timestep.

    NOTE: this function is same as parametersList() function. Structed differently
    to set move the internal file pointer back to the required file.

    Input: ParametersFile - the location of the parameters data file
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