import myFunctions
import os

'''
Feb 6, 2025
RVP

This script to get Z_net
NOTE: phi and ar are user inputs

Command to execute in terminal:
python -c "from znet import znet; znet(phi,ar)"
'''

TopDir = "/media/Linux_1TB/simulations/"

NP = 1000
numRuns = 2

def znet(phi, ar):
    for m in range(numRuns):
        run = m+1
        Dir = TopDir + "/NP_" + str(NP) + "/phi_0." + str(int(phi*1000)) + "/ar_" + str(ar) + "/Vr_0.5/run_" + str(run) +"/"
        workingFileName = Dir + "Z_Znet.txt"
        if os.path.exists(workingFileName):
            print( ">> Z_net already exists -  phi_0." + str(int(phi*1000)) + "/ar_" + str(ar) + "/run_" + str(run))
        else:
            myFunctions.Z_Znet(Dir)
            print("Done - phi_0." + str(int(phi*100)) + "/ar_" + str(ar) + "/run_" + str(run))