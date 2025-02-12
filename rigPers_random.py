import numpy as np #type: ignore
import os
import glob

"""
November 18, 2024
RVP

This script generates rigidity persistence data for a random configuration of rigid particles in space and time.
The purpose of this script is to empasize the existance of a correlation observed in the actual simulation data.
"""
npp     = 1000
phi     = [0.72, 0.74, 0.75, 0.76, 0.765, 0.77, 0.78, 0.785, 0.79, 0.795, 0.80]
ar      = [1.0, 1.4, 2.0, 4.0]
vr      = ['0.25', '0.5', '0.75']
numRuns = 1
off     = 100
topDir  = "/Volumes/Rahul_2TB/high_bidispersity"

for j, arj in enumerate(ar):
    for k, phik in enumerate(phi):
        phir = '{:.3f}'.format(phik) if len(str(phik).split('.')[1]) > 2 else '{:.2f}'.format(phik)
        for l, vrl in enumerate(vr):
            for m in range(numRuns):
                workDir = f'{topDir}/NP_{npp}/phi_{phir}/ar_{arj}/vr_{vrl}/run_{m+1}'
                if os.path.exists(workDir):
                    print(f'Operating on - phi: {phik}, ar: {arj}, vr: {vrl}')
                    dataFile  = glob.glob(f'{workDir}/data_*.dat')[0]
                    data      = np.loadtxt(dataFile)
                    gammat    = data[-1][1]

                    frigFile  = f'{workDir}/F_rig.txt'
                    data1     = np.loadtxt(frigFile)
                    frig      = np.mean(data1[off:])/npp
                    timesteps = int(gammat*100)

                    Nelements          = int(timesteps * npp) # total elements in matrix
                    Nones              = int(Nelements * frig)
                    array_flat         = np.zeros(Nelements, dtype=bool)
                    array_flat[:Nones] = True

                    np.random.shuffle(array_flat)
                    rigMatrix = array_flat.reshape(timesteps, npp)
                    ntaus     = timesteps - off # total number of steady state timesteps
                    
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
                    rigPersFile = open(workDir+"/rigPers_random.txt", "w")
                    rigPersFile.write('Delta gamma       C' + '\n')
                    for k in range(ntaus):
                        rigPersFile.write(str(round((k)/100,2)) + '      ' +
                                            str(ctau_norm[k])       + '\n')
                    rigPersFile.close()
                    print(f'Done - {workDir}')