import sys
# path where supporting files exist
sys.path.append('/Users/rahul/City College Dropbox/Rahul Pandare/CUNY/Research/Bidisperse Project/myLibrary')
import myFunctions
import os

'''
Feb 6, 2025
RVP

This script find IDs of particles with non-sliding frictional contacts at each timestep.
NOTE: phi and ar values are user inputs

Command to execute in terminal:
python3 -c "from fricPartsIDs import frictPartsIDs; frictPartsIDs(0.784,1.4)
'''

TopDir = "/media/rahul/Rahul_2TB/high_bidispersity/new_params/"

NP  = 1000
#ar  = [1.0, 1.4, 2.0, 4.0]
numRuns = 2

def frictPartsIDs(phi, ar):
    for m in range(numRuns):
        run = m + 1
        Dir = f"{TopDir}/NP_{NP}/phi_0.{int(phi * 1000)}/ar_{ar}/Vr_0.5/run_{run}/"
        workingFileName = f"{Dir}fric_parts.txt"
        if os.path.exists(workingFileName):
            print(f">> fric_parts already exists - phi_0.{int(phi * 1000)}/ar_{ar}/run_{run}")
        else:
            myFunctions.frict_parts_IDs(Dir)
            print(f"Done - phi_0.{int(phi * 1000)}/ar_{ar}/run_{run}")