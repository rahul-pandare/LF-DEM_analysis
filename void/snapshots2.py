# Script to makes simulation snapshots
# user input: phi & ar
# pre-requiste: need to have IDs of particles in cluster (fricParts.py)

# command to run in terminal: python -c "from snapshots2 import snapshots; snapshots(phi,ar)"
import myFunctions

TopDir = "/Users/rahul/Downloads"

def snapshots(phi,ar):
    NP_array  = [1000]

    run_dict = {500:8, 1000:1, 2000:2, 4000:1}

    for j in range(len(NP_array)):
        NP = NP_array[j]
        for m in range(run_dict[NP_array[j]]):
            run = m+1;
            Dir = TopDir + "/NP_" + str(NP) + "/phi_0." + str(int(phi*100)) + "/ar_" + str(ar) + "/Vr_0.5/run_" + str(run) +"/"
            myFunctions.make_snapshots(Dir)
            print("Done - phi_0." + str(int(phi*100)) + "/ar_" + str(ar) + "/run_" + str(run))