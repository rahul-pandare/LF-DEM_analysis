import os
import numpy as np
import FilesGenerator
# import steadyStateTime

outputVar = 't'   # it can either be 't' or 'gamma' (check it in your parameters file)

#sigmas    = np.array([20, 100])

NP_array  = [1000]

#phi_array = [0.70,0.71,0.72,0.74]
phi_array = [0.74]

#ar_array = [1.0,1.4,1.8,2.0,4.0]
ar_array = [1.8]

run_dict = {500:8, 1000:4, 2000:2, 4000:1}


TopDir = "/home/rahul/Documents/Bidisperse_project"

for j in range(len(NP_array)):
    NP = NP_array[j]
    for k in range(len(phi_array)):
        phi = phi_array[k]
        for l in range(len(ar_array)):
            ar = ar_array[l]
            for m in range(run_dict[NP_array[j]]):
                run = m+1;
                Dir = TopDir + "/NP_" + str(NP) + "/phi_0." + str(int(phi*100)) + "/ar_" + str(ar) + "/Vr_0.5/run_" + str(run) +"/"
                print('')
                print(" NP  = " + str(NP), ",   phi = 0." + str(int(phi*100)), " , ar = " + str(ar) ," , run = " + str(run))
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
