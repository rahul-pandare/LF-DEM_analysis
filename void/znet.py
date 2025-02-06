# Script to Z_net
# user input: phi

# command to run in terminal: python -c "from znet import znet; znet(phi)"

import myFunctions
import os

TopDir = "/home/rahul/Documents/Bidisperse_project"

def znet(phi):
    NP_array  = [1000]

    #ar_array = [1.0,1.4,1.8,2.0,4.0]
    ar_array = [1.4,4.0]

    run_dict = {500:8, 1000:4, 2000:2, 4000:1}

    for j in range(len(NP_array)):
        NP = NP_array[j]
        for l in range(len(ar_array)):
            ar = ar_array[l]
            for m in range(run_dict[NP_array[j]]):
                run = m+1;
                Dir = TopDir + "/NP_" + str(NP) + "/phi_0." + str(int(phi*100)) + "/ar_" + str(ar) + "/Vr_0.5/run_" + str(run) +"/"
                workingFileName = Dir + "Z_Znet.txt"
                if os.path.exists(workingFileName):
                    print( ">> Z_net already exists -  phi_0." + str(int(phi*100)) + "/ar_" + str(ar) + "/run_" + str(run))
                else:
                    myFunctions.Z_Znet(Dir)
                    print("Done - phi_0." + str(int(phi*100)) + "/ar_" + str(ar) + "/run_" + str(run))