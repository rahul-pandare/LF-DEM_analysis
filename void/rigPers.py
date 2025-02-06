# function script to find rigidity persistance of particles in rigid cluster

import myFunctions
import os

TopDir = "/media/rahul/Rahul_2TB/high_bidispersity"

NP  = [1000]

phi = [0.74]#, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.792]

ar  = [1.0, 1.4, 1.8, 2.0, 4.0]

run = {500:8, 1000:4, 2000:2, 4000:1}

off = 100

for i in range(len(NP)):
    for j in range(len(phi)):
        phir = '{:.3f}'.format(phi[j]) if len(str(phi[j]).split('.')[1])>2 else '{:.2f}'.format(phi[j])
        for k in range(len(ar)):
            for l in range (run[NP[i]]):
                dataname = f'{TopDir}/NP_{NP[i]}/phi_{phir}/ar_{ar[k]}/Vr_0.5/run_{l+1}/'
                if os.path.exists(dataname):
                    rigPersFile = f'{dataname}/rigPers.txt'
                    if os.path.exists(rigPersFile):
                        print(f'Rigidity persistance file already exists - {dataname}')
                    else:
                        myFunctions.rigPers(dataname, off, 'gamma')
                        print(f'Done - {dataname}')