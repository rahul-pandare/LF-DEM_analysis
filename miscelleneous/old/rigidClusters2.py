""""
function script to find F_rig for specified systems
it uses pebble game to calculate rigid clusters

user input: phi, ar

NOTE: command to run in terminal- python3 -c "from rigidClusters2 import rigidClusters; rigidClusters(0.784,1.4)"

earlier named as doAllRuns.py
"""

import os
import FilesGenerator

outputVar = 't'   # it can either be 't' or 'gamma' (check it in your parameters file)

NP_array  = [1000]

#phi_array = [0.70,0.71,0.72,0.74]
phi_array = [0.781]

#ar_array = [1.0]

#run_dict = {500:8, 1000:4, 2000:2, 4000:1}
run_dict = {500:8, 1000:1, 2000:2, 4000:1}

#TopDir = "/media/Linux_1TB/simulations/"
TopDir = "/media/rahul/Rahul_2TB/high_bidispersity/new_params"

def rigidClusters(phi,ar):
    for j in range(len(NP_array)):
        NP = NP_array[j]
        for m in range(run_dict[NP_array[j]]):
            run = m+1;
            Dir = TopDir + "/NP_" + str(NP) + "/phi_0." + str(int(phi*1000)) + "/ar_" + str(ar) + "/Vr_0.5/run_" + str(run) +"/"
            print('')
            print(" NP  = " + str(NP), ",   phi = 0." + str(int(phi*1000)), " , ar = " + str(ar) ," , run = " + str(run))
            workingFileName = Dir + '00_OngoingFileGenerator.txt'
            if os.path.exists(workingFileName):
                print('  >> The files are being generated for this case  >>  SKIPPING')
            else:
                workingFile = open(workingFileName, "w")       
                workingFile.write('This is just a file to indicate that the some work is going on in this directory.\n')
                workingFile.close()
                t_SS = 0 #steadyStateTime.t_SS(i,j,k)
                FilesGenerator.filesGeneratorOneRun(NP, phi, Dir, t_SS, outputVar, makeMovies=False)
                os.remove(workingFileName)
    print('')


'''
dependencies: 
rigidClusters.py (RVP) --> filesgenerator.py --> myFunctions.py (Michel) --> myRigidClusterProcessor.py (Mike van der Naald)

'''
