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
topDir      = "/Volumes/Rahul_2TB/high_bidispersity/"

# Relevant file names to read.
ranSeedFile = "random_seed.dat"
intFile     = "int_random_seed_params_stress100r_shear.dat"

# Some simulation parameters.
NP          = [1000]

run         = {500:8,1000:4,2000:2,4000:1}

phi         = [0.70,0.71,0.72,0.73,0.74,0.76]

ar          = [1.0, 1.4, 1.8, 2.0, 4.0]

for ii in range(len(NP)):
    for j in range(len(phi)):
        phir = '{:.3f}'.format(phi[j]) if len(str(phi[j]).split('.')[1])>2 else '{:.2f}'.format(phi[j])
        for k in range(len(ar)):
            dataname=topDir+'NP_'+str(NP[ii])+'/phi_'+phir+'/ar_'+str(ar[k])+'/Vr_0.5'
            if os.path.exists(dataname):
                for l in range (run[NP[ii]]):
                    ranFile = open(f'{dataname}/run_{l+1}/{ranSeedFile}', 'r')

                    if ar[k] == 1:
                        particleSize = [1]*(int(NP[ii]/2)) + [2]*(int(NP[ii]/2))
                        random.shuffle(particleSize)
                        #painting particles randomly in two colours for monodisperse case
                    else:
                        particleSize = np.loadtxt(ranFile,usecols=3) # reading only column 3 which has particle size
                        ranFile.close()
                    
                    #%% counting all the small and large particles in system
                    countSmall = 0
                    countLarge = 0

                    for m in range(1000):
                        if particleSize[m] > 1:
                            countLarge += 1
                        else:
                            countSmall += 1

                    #%% reading interaction file
                    # storing the interactions for each timestep in a list

                    hashCounter=0
                    temp=[]
                    contactList=[] # list with interaction parameters for each element at each timestep

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
                                temp = []
                                hashCounter = 0
                    interFile.close()

                    #%% Total contacts
                    # list of tuples. Each tuple for each timestep
                    # tuple: (total no. of frictional contact, number of particles in frictional contact)

                    # counting total number for frictional contacts at a timestep. (contact state = 2)
                    totCont = [2*list(sampleList[:, 10]).count(2) for sampleList in contactList] 
                    # multiplying the number contacts by two because we count number of contacts per particle 
                    # and one contact in system means two contacts in total for the interacting pair

                    TotalParticlesinContact = [] # total number of particles in frictional contacts
                    for sampleList in contactList:
                        particleIndex = [] # index of particles in contact
                        for i in range (sampleList.shape[0]):
                            if int(sampleList[i,10]) == 2:
                                particleIndex.append(sampleList[i,0])
                                particleIndex.append(sampleList[i,1])
                        TotalParticlesinContact.append(len(list(set(particleIndex))))

                    TotalContacts = list(zip(totCont,TotalParticlesinContact))

                    #%% Total contacts between small-small particles
                    totalSSContacts  = []
                    totalSSParticles = []

                    for sampleList in contactList:
                        countCont = 0
                        particleIndex = []
                        for i in range (sampleList.shape[0]):
                            if (particleSize[int(sampleList[i,0])] == 1 and particleSize[int(sampleList[i,1])] == 1) and int(sampleList[i,10]) == 2:
                                countCont += 2 # counting two total contacts per pair for each contact
                                particleIndex.append(sampleList[i,0])
                                particleIndex.append(sampleList[i,1])
                                
                        totalSSContacts.append(countCont)
                        totalSSParticles.append(len(list(set(particleIndex))))
                        
                    totalSmallSmall = list(zip(totalSSContacts,totalSSParticles))

                    #%% Total contacts between large-large particles
                    totalLLContacts  = []
                    totalLLParticles = []

                    for sampleList in contactList:
                        countCont = 0
                        particleIndex = []
                        for i in range (sampleList.shape[0]):
                            if (particleSize[int(sampleList[i,0])] > 1 and particleSize[int(sampleList[i,1])] > 1) and int(sampleList[i,10]) == 2:
                                countCont += 2
                                particleIndex.append(sampleList[i,0])
                                particleIndex.append(sampleList[i,1])
                                
                        totalLLContacts.append(countCont)
                        totalLLParticles.append(len(list(set(particleIndex))))
                        
                    totalLargeLarge = list(zip(totalLLContacts,totalLLParticles))

                    #%% Total contacts between small-large particles
                    totalSLContacts  = []
                    totalSLParticles = []

                    for sampleList in contactList:
                        countCont = 0
                        particleIndex = []
                        for i in range (sampleList.shape[0]):
                            if (particleSize[int(sampleList[i, 0])] != particleSize[int(sampleList[i, 1])]) and int(sampleList[i, 10]) == 2:
                                countCont += 2
                                particleIndex.append(sampleList[i,0])
                                particleIndex.append(sampleList[i,1])

                        totalSLContacts.append(countCont)   
                        totalSLParticles.append(len(list(set(particleIndex))))
                        
                    totalSmallLarge = list(zip(totalSLContacts,totalSLParticles))

                    #################################################################################################################################

                    #%% Total contacts on small particles only
                    totCont = []
                    totalParticlesinContactS = []
                    for sampleList in contactList:
                        countCont = 0
                        particleIndex = []
                        for i in range (sampleList.shape[0]):
                            if (particleSize[int(sampleList[i,0])] == 1 or particleSize[int(sampleList[i,1])] == 1) and int(sampleList[i,10]) == 2:
                                countCont += 2
                                if particleSize[int(sampleList[i,0])] == 1 and int(sampleList[i,0]) not in particleIndex:
                                    particleIndex.append(sampleList[i,0])
                                if particleSize[int(sampleList[i,1])] == 1 and int(sampleList[i,1]) not in particleIndex:
                                    particleIndex.append(sampleList[i,1])
                        
                        totCont.append(countCont)
                        totalParticlesinContactS.append(len(particleIndex))
                        
                    totalContactOnSmall = list(zip(totCont,totalParticlesinContactS))

                    #%% Total contacts on large particles only
                    totCont = []
                    totalParticlesinContactL = []
                    for sampleList in contactList:
                        countCont = 0
                        particleIndex = []
                        for i in range (sampleList.shape[0]):
                            if (particleSize[int(sampleList[i,0])] > 1 or particleSize[int(sampleList[i,1])] > 1) and int(sampleList[i,10]) == 2:
                                countCont += 2
                                if particleSize[int(sampleList[i,0])] > 1 and int(sampleList[i,0]) not in particleIndex:
                                    particleIndex.append(sampleList[i,0])
                                if particleSize[int(sampleList[i,1])] > 1 and int(sampleList[i,1]) not in particleIndex:
                                    particleIndex.append(sampleList[i,1])
                        
                        totCont.append(countCont)
                        totalParticlesinContactL.append(len(particleIndex))
                        
                    totalContactOnLarge = list(zip(totCont,totalParticlesinContactL))

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
                    txtFile.write("# 9 : Total number of particles with non-sliding frictional contact on small particles\n\n")
                    
                    txtFile.write("# 10 : Total non-sliding frictional contacts on large particles \n")
                    txtFile.write("# 11 : Total number of particles with non-sliding frictional contact on large particles\n\n")
                      
                    # Write the data from each list side by side
                    for n in range(len(TotalContacts)):
                        row = TotalContacts[n] + totalSmallSmall[n] + totalSmallLarge[n] + totalLargeLarge[n] + totalContactOnSmall[n] + totalContactOnLarge[n]
                        txtFile.write(" ".join(map(str, row)) + "\n")

                    txtFile.close()
                
                print(f'Done - {dataname}')

            else:
                print(f"path '{dataname}' does not exist")