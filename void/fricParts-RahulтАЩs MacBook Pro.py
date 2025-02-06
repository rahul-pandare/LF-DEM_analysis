import myFunctions
import os

TopDir="/Users/rahul/Library/CloudStorage/OneDrive-CUNY/CUNY/Research/Bidisperse_project/high_bidispersity"

def frictPartsIDs(phi):
    NP_array  = [1000]

    ar_array = [1.0,1.4,1.8,2.0,4.0]

    run_dict = {500:8, 1000:4, 2000:2, 4000:1}

    for j in range(len(NP_array)):
        NP = NP_array[j]
        for l in range(len(ar_array)):
            ar = ar_array[l]
            for m in range(run_dict[NP_array[j]]):
                run = m+1;
                Dir = TopDir + "/NP_" + str(NP) + "/phi_0." + str(int(phi*100)) + "/ar_" + str(ar) + "/Vr_0.5/run_" + str(run) +"/"
                workingFileName = Dir + "frictPartsIDs.txt"
                if os.path.exists(workingFileName):
                    print( ">> frictPartsIDs already exists -  phi_0." + str(int(phi*100)) + "/ar_" + str(ar) + "/run_" + str(run))
                else:
                    myFunctions.frict_parts_IDs(Dir)
                    print("Done - phi_0." + str(int(phi*100)) + "/ar_" + str(ar) + "/run_" + str(run))