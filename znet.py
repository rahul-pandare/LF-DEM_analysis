# Script to Z_net
# user input: phi & ar

# command to run in terminal: python -c "from znet2 import znet2; znet2(phi,ar)"

import myFunctions
import os

TopDir = "/media/Linux_1TB/simulations/"

NP_array  = [1000]

run_dict = {500:8, 1000:4, 2000:2, 4000:1}

def znet2(phi,ar):
    for j in range(len(NP_array)):
        NP = NP_array[j]
        for m in range(run_dict[NP_array[j]]):
            run = m+1;
            
            Dir = TopDir + "/NP_" + str(NP) + "/phi_0." + str(int(phi*1000)) + "/ar_" + str(ar) + "/Vr_0.5/run_" + str(run) +"/"
            workingFileName = Dir + "Z_Znet.txt"
            if os.path.exists(workingFileName):
                print( ">> Z_net already exists -  phi_0." + str(int(phi*1000)) + "/ar_" + str(ar) + "/run_" + str(run))
            else:
                myFunctions.Z_Znet(Dir)
                print("Done - phi_0." + str(int(phi*100)) + "/ar_" + str(ar) + "/run_" + str(run))