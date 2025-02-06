import os
import numpy as np # type: ignore
import random

'''
June 11, 2024
RVP

This script generated a text file 'contacts.txt' for any specific case.
The text file contains data on total number of contacts and number of particles in contact
for different size pair or on a particular size contact.

commands to run in terminal:
python3 contatcts.py 
'''

# Simulation data mount point
#topDir      = "/Volumes/Rahul_2TB/high_bidispersity/"
topDir      = "/media/rahul/Rahul_2TB/high_bidispersity/"

# Relevant file names to read.
ranSeedFile = "random_seed.dat"
intFile     = "int_random_seed_params_stress100r_shear.dat"

# Some simulation parameters.
NP          = [1000]

run         = {500:8,1000:4,2000:2,4000:1}

phi         = [0.78]

ar          = [4.0] #[1.0, 1.4, 1.8, 2.0, 4.0]


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

for ii in range(len(NP)):
    for j in range(len(phi)):
        phir = '{:.3f}'.format(phi[j]) if len(str(phi[j]).split('.')[1])>2 else '{:.2f}'.format(phi[j])
        for k in range(len(ar)):
            dataname=topDir+'NP_'+str(NP[ii])+'/phi_'+phir+'/ar_'+str(ar[k])+'/Vr_0.5'
            if os.path.exists(dataname):
                for l in range (run[NP[ii]]):
                    ranFile = open(f'{dataname}/run_{l+1}/{ranSeedFile}', 'r')

                    if ar[k] == 1:
                        # Painting particles randomly in two colours for monodisperse case.
                        particleSize = [1]*(int(NP[ii]/2)) + [2]*(int(NP[ii]/2))
                        random.shuffle(particleSize)
                    else:
                        particleSize = np.loadtxt(ranFile,usecols=3) # reading only column 3 which has particle size
                        ranFile.close()
                    
                    # Counting all the small and large particles in system.
                    countSmall = 0
                    countLarge = 0

                    for m in range(NP[ii]):
                        if particleSize[m] > 1:
                            countLarge += 1
                        else:
                            countSmall += 1

                    # reading interaction file
                    interFile   = open(f'{dataname}/run_{l+1}/{intFile}', 'r')
                    contactList = interactionsList(interFile)

                    contType = ['Total', 'Small - Small', 'Small - Large', 'Large - Large', 'On Small', 'On Large']

                    totContList    = [[] for _ in range(len(contType))]
                    particleList   = [[] for _ in range(len(contType))]

                    #TotalParticlesinContact = [] # total number of particles in frictional contacts
                    for sampleList in contactList:
                        countContList  = [0  for _ in range(len(contType))]
                        particleIndex  = [[] for _ in range(len(contType))]
                        for i in range (sampleList.shape[0]):
                            particleIndex1, particleIndex2 = int(sampleList[i,0]), int(sampleList[i,1])
                            particleSize1, particleSize2   = particleSize[particleIndex1], particleSize[particleIndex2]
                            contState                      = int(sampleList[i,10])

                            if contState == 2:
                                # Total
                                countContList[0] += 2
                                particleIndex[0].append(particleIndex1)
                                particleIndex[0].append(particleIndex2)

                                # Small-Small
                                if particleSize1 == particleSize2 == 1:
                                    countContList[1] += 2
                                    particleIndex[1].append(particleIndex1)
                                    particleIndex[1].append(particleIndex2)

                                # Small-Large
                                elif particleSize1 != particleSize2:
                                    countContList[2] += 2
                                    particleIndex[2].append(particleIndex1)
                                    particleIndex[2].append(particleIndex2)

                                # Large-Large
                                elif particleSize1 == particleSize2 > 1:
                                    countContList[3] += 2
                                    particleIndex[3].append(particleIndex1)
                                    particleIndex[3].append(particleIndex2)

                                # On Small
                                if particleSize1 == 1 or particleSize2 == 1:
                                    countContList[4] += 2
                                    if particleSize1 == 1:
                                        particleIndex[4].append(particleIndex1)
                                    if particleSize2 == 1:
                                        particleIndex[4].append(particleIndex2)

                                # On Large
                                if particleSize1 > 1 or particleSize2 > 1:
                                    countContList[5] += 2
                                    if particleSize1 > 1:
                                        particleIndex[5].append(particleIndex1)
                                    if particleSize2 > 1:
                                        particleIndex[5].append(particleIndex2)

                        for ik in range(len(contType)):
                            totContList[ik].append(countContList[ik])
                            particleList[ik].append(len(list(set(particleIndex[ik]))))
                            
                    # writing data onto text file
                    txtFile = open(f'{dataname}/run_{l+1}/contacts.txt', 'w')
                    txtFile.write(f"# nps = {countSmall}\n") # number of small particles
                    txtFile.write(f"# npl = {countLarge}\n\n") # number of large particles
                    
                    # column info
                    txtFile.write("# 0 : Total non-sliding frictional contacts \n")
                    txtFile.write("# 1 : Total number of particles with non-sliding frictional contact \n\n")

                    txtFile.write("# 2 : Total non-sliding frictional contacts between small - small particles\n")
                    txtFile.write("# 3 : Total number of particles with non-sliding frictional contact between small - small particles\n\n")

                    txtFile.write("# 4 : Total non-sliding frictional contacts between small - large particles \n")
                    txtFile.write("# 5 : Total number of particles with non-sliding frictional contact between small - large particles \n\n")
                    
                    txtFile.write("# 6 : Total non-sliding frictional contacts between large - large particles \n")
                    txtFile.write("# 7 : Total number of particles with non-sliding frictional contact between large - large particles \n\n")

                    txtFile.write("# 8 : Total non-sliding frictional contacts on small particles \n")
                    txtFile.write("# 9 : Total number of small particles with non-sliding frictional contact on small particles\n\n")
                    
                    txtFile.write("# 10 : Total non-sliding frictional contacts on large particles \n")
                    txtFile.write("# 11 : Total number of large particles with non-sliding frictional contact on large particles\n\n")
                      
                    for items in zip(totContList[0], particleList[0], totContList[1], particleList[1], 
                                     totContList[2], particleList[2], totContList[3], particleList[3], 
                                     totContList[4], particleList[4], totContList[5], particleList[5]):
                        txtFile.write("\t".join(map(str, items)) + "\n")
                    txtFile.close()
                
                print(f'Done - {dataname}')

            else:
                print(f"path '{dataname}' does not exist")