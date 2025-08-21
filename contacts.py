import os
import numpy as np # type: ignore
import random
import readFiles
import glob

'''
Aug 21, 2025 RVP - Included readFiles.py.
Jun 11, 2024 RVP - Initial version of the script.

This script generated a text file 'contacts.txt' for any specific case.
The text file contains data on total number of contacts and number of particles in contact
for different size pair or on a particular size contact.

commands to run in terminal:
python3 contatcts.py 
'''

# Simulation data mount point
topDir      = "/Volumes/rahul_2TB/high_bidispersity/new_data"
#topDir      = "/media/rahul/Rahul_2TB/high_bidispersity/"

# Relevant file names to read.
ranSeedFile = "random_seed.dat"
intFile     = "int_*.dat"

# Some simulation parameters.
npp  = 1000
phi  = [0.72, 0.74, 0.75, 0.76, 0.765, 0.77, 0.78, 0.785, 0.79, 0.795, 0.8]
ar   = [1.4] #[1.0, 1.4, 2.0, 4.0]
vr   = ['0.5']
runs =  2

for i in range(len(phi)):
    phir = '{:.3f}'.format(phi[i]) if len(str(phi[i]).split('.')[1])>2 else '{:.2f}'.format(phi[i])
    for j in range(len(ar)):
        for k in range(len(vr)):
            dataname = f"{topDir}/NP_{npp}/phi_{phir}/ar_{ar[j]}/Vr_{vr[k]}"
            if os.path.exists(dataname):
                for l in range (runs):
                    ranFile = open(f'{dataname}/run_{l+1}/{ranSeedFile}', 'r')

                    if ar[k] == 1:
                        # Painting particles randomly in two colours for monodisperse case.
                        particleSize = [1]*(int(npp/2)) + [2]*(int(npp/2))
                        random.shuffle(particleSize)
                    else:
                        particleSize = np.loadtxt(ranFile,usecols=3) # reading only column 3 which has particle size
                        ranFile.close()
                    
                    # Counting all the small and large particles in system.
                    countSmall = 0
                    countLarge = 0

                    for m in range(npp):
                        if particleSize[m] > 1:
                            countLarge += 1
                        else:
                            countSmall += 1

                    # reading interaction file
                    interFile    = open(glob.glob(f'{dataname}/run_{l+1}/{intFile}')[0], 'r')
                    contactList  = readFiles.interactionsList(interFile)

                    contType     = ['Total', 'Small - Small', 'Small - Large', 'Large - Large', 'On Small', 'On Large']

                    totContList  = [[] for _ in range(len(contType))]
                    particleList = [[] for _ in range(len(contType))]

                    #TotalParticlesinContact = [] # total number of particles in frictional contacts
                    for sampleList in contactList:
                        countContList  = [0  for _ in range(len(contType))]
                        particleIndex  = [[] for _ in range(len(contType))]
                        for ii in range (sampleList.shape[0]):
                            particleIndex1, particleIndex2 = int(sampleList[ii,0]), int(sampleList[ii,1])
                            particleSize1, particleSize2   = particleSize[particleIndex1], particleSize[particleIndex2]
                            contState                      = int(sampleList[ii,10])

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