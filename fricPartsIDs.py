# To find IDs of particles with non-sliding frictional contacts at each timestep

# command to run in terminal: python3 -c "from fricParts2 import frictPartsIDs2; frictPartsIDs2(0.784,1.4)"

import myFunctions
import os

#TopDir = "/media/Linux_1TB/simulations/"
#TopDir =  "/Users/rahul/Documents/Simulations/aws_c7i/"
TopDir = "/media/rahul/Rahul_2TB/high_bidispersity/new_params/"

def frictPartsIDs2(phi,ar):
    NP_array  = [1000]

    #ar_array = [1.0,1.4,1.8,2.0,4.0]

    run_dict = {500:8, 1000:1, 2000:2, 4000:1}

    for j in range(len(NP_array)):
        NP = NP_array[j]
        for m in range(run_dict[NP_array[j]]):
            run = m+1;
            Dir = TopDir + "/NP_" + str(NP) + "/phi_0." + str(int(phi*1000)) + "/ar_" + str(ar) + "/Vr_0.5/run_" + str(run) +"/"
            workingFileName = Dir + "fric_parts.txt"
            if os.path.exists(workingFileName):
                print( ">> fric_parts already exists -  phi_0." + str(int(phi*1000)) + "/ar_" + str(ar) + "/run_" + str(run))
            else:
                myFunctions.frict_parts_IDs(Dir)
                print("Done - phi_0." + str(int(phi*1000)) + "/ar_" + str(ar) + "/run_" + str(run))
