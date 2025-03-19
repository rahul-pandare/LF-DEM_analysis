import sys
# path where supporting files exist
sys.path.append('/Users/rahul/City College Dropbox/Rahul Pandare/CUNY/research/bidisperse_project/myLibrary/rigCalc')

import myFunctions    # type: ignore
import FilesGenerator # type: ignore
import os
import glob

'''
Feb 6 2025
RVP

This script creates Frig, rig_*.dat files if missing and also rigPrime files if missing
The above mentioned files are required for snapshots
'''
outputVar = 't'   # it can either be 't' or 'gamma' (check it in your parameters file)

npp     = 1000
phi     = [0.785, 0.795]
ar      = [4.0]
vr      = ['0.25', '0.5', '0.75']
numRuns = 1

topDir   = '/Volumes/rahul_2TB/high_bidispersity/new_data'
frigFile = 'F_rig.txt' 
rigFile  = 'rig_*.dat' 
    
for i, phii in enumerate(phi):
    phir = '{:.3f}'.format(phii) if len(str(phii).split('.')[1]) > 2 else '{:.2f}'.format(phii)
    for j, arj in enumerate(ar):
        for k, vrk in enumerate(vr):
            for m in range(numRuns):
                run = m+1
                datname  = f"{topDir}/NP_{npp}/phi_{phir}/ar_{arj:.1f}/Vr_{vrk}/run_{run}/"
                if os.path.exists(datname):
                    rigPath  = glob.glob(f'{datname}{rigFile}')
                    frigPath = f'{datname}{frigFile}'
                    if len(rigPath) == 0  or not os.path.exists(frigPath):
                        print('')
                        print(f'NP = {npp}, phi = {phir}, ar = {arj}, vr = {vrk}, run = {run}')
                        workingFileName = datname + '00_OngoingFileGenerator.txt'
                        # if os.path.exists(workingFileName):
                        #     print('  >> The files are being generated for this case  >>  SKIPPING')
                        # else:
                        workingFile = open(workingFileName, "w")       
                        workingFile.write('This is just a file to indicate that the some work is going on in this directory.\n')
                        workingFile.close()
                        t_SS = 0 #steadyStateTime.t_SS(i,j,k)
                        FilesGenerator.filesGeneratorOneRun(npp, phii, datname, t_SS, 't', makeMovies=False)
                        os.remove(workingFileName)
                    else:
                        print('  >> The rigdity files already exist for this case  >>  SKIPPING')
        
                    # rigPrimeFile = f'{datname}rigPrime.txt'
                    # if not os.path.exists(rigPrimeFile):
                    #     myFunctions.myPrimeRigidClusters(datname)
                else:
                    print(f'directory not found - {topDir}/NP_{npp}/phi_{phir}/ar_{arj:.1f}/Vr_{vrk}/run_{run}/')