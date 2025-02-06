# Script to makes simulation snapshots
# user input: phi
# pre-requiste: need to have IDs of particles in cluster (fricParts.py)

# command to run in terminal: python -c "from snapshots import snapshots; snapshots(phi)"

import myFunctions

TopDir = "/home/rahul/Documents/Bidisperse_project"

def snapshots(phi):
    NP_array  = [1000]

    ar_array = [1.0,1.4,1.8,2.0,4.0]

    run_dict = {500:8, 1000:1, 2000:2, 4000:1}

    for j in range(len(NP_array)):
        NP = NP_array[j]
        for l in range(len(ar_array)):
            ar = ar_array[l]
            for m in range(run_dict[NP_array[j]]):
                run = m+1;
                Dir = TopDir + "/NP_" + str(NP) + "/phi_0." + str(int(phi*100)) + "/ar_" + str(ar) + "/Vr_0.5/run_" + str(run) +"/"
                myFunctions.make_snapshots(Dir)
                print("Done - phi_0." + str(int(phi*100)) + "/ar_" + str(ar) + "/run_" + str(run))