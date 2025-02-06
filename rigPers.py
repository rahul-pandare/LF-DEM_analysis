import numpy as np
import os
import glob

"""
November 18, 2024
RVP

This script is used to calculate rigidity persistence in dense suspensions using auto correlation function
"""

def rigList(rigFile):
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

NP      = [1000]
phi     = [0.74] #[0.70, 0.72, 0.75, 0.76, 0.79]
ar      = [4.0] #[1.0, 1.4, 1.8, 2.0, 4.0]
numRuns = 1
topDir  = "/Volumes/Rahul_2TB/high_bidispersity"
#topDir  = "/media/rahul/Rahul_2TB/high_bidispersity"
off     = 100

print(phi)
print(ar)

for i, npp in enumerate(NP):
    for j, arj in enumerate(ar):
        for k, phij in enumerate(phi):
            phir = '{:.3f}'.format(phij) if len(str(phij).split('.')[1]) > 2 else '{:.2f}'.format(phij)
            for ll in range(numRuns):
                workDir = f'{topDir}/NP_{npp}/phi_{phir}/ar_{arj}/Vr_0.5/run_{ll+1}'
                if os.path.exists(workDir):
                    rigFile = open(glob.glob(f'{workDir}/rig_*.dat')[0])
                    rigClusterIDs = rigList(rigFile)
                    
                    clusterIDs = [[np.nan] if len(samplelist[0]) < 2 else list({int(num) for sublist in samplelist for num in sublist}) for samplelist in rigClusterIDs]
                    rigMatrix  = np.zeros((len(clusterIDs),npp), dtype=bool)
                    ntaus      = len(clusterIDs[off:])

                    for ii,samplelist in enumerate(clusterIDs):
                        if not np.isnan(samplelist)[0]:
                            rigMatrix[ii][samplelist] = True
                    
                    ctau = []
                    for tau in range(ntaus):
                        ctauNP = 0
                        for NPi in range(npp):
                            uncorr = (1/(ntaus-tau)) * np.sum(rigMatrix[off:ntaus-tau, NPi])
                            corr   = 0
                            for t in range(off, ntaus-tau):
                                corr += rigMatrix[t, NPi] * rigMatrix[t+tau, NPi]
                            corr   *= 1/(ntaus-tau)
                            ctauNP += corr - uncorr**2.
                        ctau.append(ctauNP/npp)
                        print(f'{tau}/{ntaus}')

                    ctau_norm = [x/ctau[0] for x in ctau]   
                    rigPersFile = open(workDir+"/rigPers.txt", "w")
                    rigPersFile.write('Delta gamma       C' + '\n')
                    for k in range(ntaus):
                        rigPersFile.write(str(round((k)/100,2)) + '      ' +
                                          str(ctau_norm[k])       + '\n')
                    rigPersFile.close()
                    print(f'Done - {workDir}')