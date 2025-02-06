# Script to find F_rig of systems specified below

import os
import FilesGenerator

outputVar = 't'   # it can either be 't' or 'gamma' (check it in your parameters file)

NP_array  = [1000]

phi_array = [0.765, 0.766, 0.767, 0.768, 0.769, 0.77]

ar_array = [1.4]

#run_dict = {500:8, 1000:4, 2000:2, 4000:1}
run_dict = {500:8, 1000:1, 2000:2, 4000:1}

TopDir = "/media/rahul/Rahul_2TB/high_bidispersity"

for j in range(len(NP_array)):
    NP = NP_array[j]
    for k in range(len(phi_array)):
        phi = phi_array[k]
        for l in range(len(ar_array)):
            ar = ar_array[l]
            for m in range(run_dict[NP_array[j]]):
                run = m+1;
                phir = '{:.3f}'.format(phi) if len(str(phi).split('.')[1]) > 2 else '{:.2f}'.format(phi)
                Dir = TopDir + "/NP_" + str(NP) + '/phi_' + phir + "/ar_" + str(ar) + "/Vr_0.5/run_" + str(run) +"/"
                print('')
                print(" NP  = " + str(NP), ",   phi = " + phir, " , ar = " + str(ar) ," , run = " + str(run))
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
