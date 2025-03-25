import sys
# path where supporting files exist
sys.path.append('/Users/rahul/City College Dropbox/Rahul Pandare/CUNY/research/bidisperse_project/myLibrary/rigCalc')
import myFunctions    # type: ignore
#import FilesGenerator # type: ignore
import os

'''
Feb 6, 2025
RVP

This script to get Z_net
NOTE: phi and ar are user inputs

Command to execute in terminal:
python -c "from znet import znet; znet(ar)"
'''

TopDir = "/Volumes/rahul_2TB/high_bidispersity/new_data"

NP      = 1000
numRuns = 2
phiv    = [0.72, 0.74, 0.75, 0.76, 0.765, 0.77, 0.78, 0.785, 0.79, 0.795, 0.8]
vr      = ['0.25', '0.5', '0.75']
arr     = [1.0, 1.4, 2.0, 4.0]

def znet(ar):
    for i, phii in enumerate(phiv):
        phir      = f"{phii:.3f}" if phii != round(phii, 2) else f"{phii:.2f}"
        for j, vrj in enumerate(vr):
            for m in range(numRuns):
                run = m+1
                Dir = f"{TopDir}/NP_{NP}/phi_{phir}/ar_{ar}/Vr_{vrj}/run_{run}/"
                if not os.path.exists(Dir):
                    print(f"directory not found - phi_{phir}/ar_{ar}/Vr_{vrj}/run_{run}")
                    continue

                workingFileName = Dir + "Z_Znet.txt"
                if os.path.exists(workingFileName):
                    print(f">> Z_net already exists - phi_{phir}/ar_{ar}/Vr_{vrj}/run_{run}")
                else:
                    myFunctions.Z_C(Dir)
                    print(f"Done - phi_{phir}/ar_{ar}/Vr_{vrj}/run_{run}")
