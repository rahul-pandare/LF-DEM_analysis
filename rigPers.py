import numpy as np # type: ignore
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
    for _, sampleList in enumerate(clusterIDs):
        tempList = []
        for kk in range(len(sampleList)):
            tempList.append([int(indx) for indx in sampleList[kk].split(',')])
        rigClusterIDsList.append(tempList)
    return rigClusterIDsList

npp     = 1000
phi     = [0.77]#[0.72, 0.74, 0.75, 0.76, 0.765, 0.77, 0.78, 0.785, 0.79, 0.795, 0.80]
ar      = [2.0]    #, 1.4, 2.0, 4.0]
vr      = ['0.75'] #['0.25', '0.5', '0.75']
numRuns = 1
off     = 100

topDir      = "/Volumes/rahul_2TB/high_bidispersity/new_data"
rigPersFile = "rigPers.txt"

for j, arj in enumerate(ar):
    for k, phik in enumerate(phi):
        phir = '{:.3f}'.format(phik) if len(str(phik).split('.')[1]) > 2 else '{:.2f}'.format(phik)
        for l, vrl in enumerate(vr):
            for m in range(numRuns):
                workDir = f'{topDir}/NP_{npp}/phi_{phir}/ar_{arj}/vr_{vrl}/run_{m+1}'
                if os.path.exists(workDir):
                    if not os.path.exists(f'{workDir}/{rigPersFile}'):
                        print(f'Operating on - phi: {phik}, ar: {arj}, vr: {vrl}')
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
                        rigPers = open(f'{workDir}/{rigPersFile}', "w")
                        rigPers.write('Delta gamma       C' + '\n')
                        for k in range(ntaus):
                            rigPers.write(str(round((k)/100,2)) + '      ' +
                                                str(ctau_norm[k])       + '\n')
                        rigPers.close()
                        print(f'Done - phi_{phir}/ar_{arj}/vr_{vrl}/run_{m+1}')
                    else:
                        print(f'  >>> Rigidity persistance (random) file already exists - phi_{phir}/ar_{arj}/vr_{vrl}/run_{m+1}')