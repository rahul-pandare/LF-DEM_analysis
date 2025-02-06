import os
import numpy as np 
import glob
from tqdm import tqdm
import sys

'''
Aug 15, 2024
RVP

This script calculates the pair distribution function for a particular case
'''

# Simulation data mount point
#topDir      = "/Volumes/Rahul_2TB/high_bidispersity/"
topDir      = "/media/rahul/Rahul_2TB/high_bidispersity/"

# Simulation parameters.
NP          = [1000]

run         = {500:8,1000:1,2000:2,4000:1}

phi         = [0.74]

ar          = [1.4]

# PDF parameters
dtheta = 5    # in degrees
dr     = 0.2  # in unit length
off    = 100  # no. of timesteps to skip for steady state

def parametersList(ParametersFile):
    '''
    This function reads the parameters file and creates a nested-list,
    each list inside contains the array of all interaction parameters for
    that timestep.

    Input: ParametersFile - the location of the parameters data file
    '''

    parFile = open(ParametersFile, 'r')

    hashCounter = 0
    temp        = []
    parList     = [] # list with parameters parameters for each element at each timestep

    fileLines = parFile.readlines()[22:] # skipping the comment lines
    for line in fileLines:
        if not line.split()[0] == '#':
            lineList = [float(value) for value in line.split()]
            temp.append(lineList)
        else:
            hashCounter += 1 # checking if counter reaches 7 (7 lines of comments after every timestep data)
            if hashCounter == 7: 
                parList.append(np.array(temp))
                temp        = []
                hashCounter = 0
    parFile.close()
    return parList

"====================================================================================================================================="

contType = 'll'

for i in range(len(NP)):
    for j in range(len(phi)):
        phir = '{:.3f}'.format(phi[j]) if len(str(phi[j]).split('.')[1])>2 else '{:.2f}'.format(phi[j])
        for k in range(len(ar)):
            dataname = topDir+'NP_'+str(NP[i])+'/phi_'+phir+'/ar_'+str(ar[k])+'/Vr_0.5'
            if os.path.exists(dataname):
                for l in range (run[NP[i]]):
                    ranSeedFile = glob.glob(f'{dataname}/run_{l+1}/random_*.dat')[0] #"random_seed.dat"
                    datFile     = glob.glob(f'{dataname}/run_{l+1}/data_*')[0]
                    parFile     = glob.glob(f'{dataname}/run_{l+1}/par_*')[0]

                    particleSize = np.genfromtxt(ranSeedFile, skip_header=2, usecols=-1)
                    parList      = parametersList(parFile)

                    # Box dimensions
                    lx = np.genfromtxt(datFile, skip_header=3, max_rows=1, comments = '_')[2]
                    lz = np.genfromtxt(datFile, skip_header=5, max_rows=1, comments = '_')[2]

                    # Reading simulation results
                    _, gamma, _, _,  _, _, _, _, _, _, \
                    _, _, _, minGap, _, _, _, _, _, _,       \
                    _, _, _, _,  _, _, _, _, _, _            \
                    = np.loadtxt(datFile, skiprows=37).transpose()

                    # Bin parameters
                    dtheta   *= np.pi/180  # converting to radians
                    #rmin      = np.min(particleSize) + np.min(minGap)
                    rmin      = (np.max(particleSize) + np.min(minGap) if contType == 'll' else np.min(particleSize) + np.min(minGap))
                    rmax      = np.max([lx,lz])/2.
                    rbin      = np.arange(rmin,   rmax + dr,      dr)
                    thetabin  = np.arange(-np.pi, np.pi + dtheta, dtheta)

                    g_r_theta = np.zeros((len(rbin), len(thetabin)))
                    
                    SSi = parList[off:off+10]
                    
                    #with tqdm(enumerate(SSi), desc="Progress", ncols=100, total=len(SSi), leave=False) as progress_bar:
                        #for ii, mat in progress_bar:
                    #for _ in tqdm(range(len(SSi)), desc="Progress", leave=False):
                        #for ii, mat in enumerate(SSi):  
                    for idx, (ii, mat) in tqdm(enumerate(enumerate(SSi)), desc="Progress", leave=False, total=len(SSi)):
                        xp = mat[:,2]
                        zp = mat[:,3]

                        if contType == 'ss':
                            cond1 = (mat[:,1] == 1) # small particle condition
                            xp,zp = xp[cond1], zp[cond1]

                        if contType == 'll':
                            cond1 = (mat[:,1] > 1) # small particle condition
                            xp,zp = xp[cond1], zp[cond1]

                        xmat, zmat = np.outer(xp, np.ones(len(xp))), np.outer(zp, np.ones(len(xp)))
                        dxij, dzij = xmat.transpose() - xmat,      zmat.transpose() - zmat

                        # Lees Edwards boundary:
                        dxij[dzij > lz/2.]  -= gamma[ii]*lx
                        dzij[dzij > lz/2.]  -= lz

                        dxij[dzij < -lz/2.] += gamma[ii]*lx
                        dzij[dzij < -lz/2.] += lz

                        # X peridodic:
                        dxij[dxij >  lx/2.] -= lx
                        dxij[dxij < -lx/2.] += lx

                        dij = np.sqrt(dxij**2 + dzij**2) # Abosulte distance btw each pair
                        tij = np.arctan2(dzij, dxij)     # Angle btw each pair

                        dij1 = np.zeros([1000,1000])

                        if contType == 'sl':
                            for im in range(dij.shape[0]):
                                for ikk in range(dij.shape[1]):
                                    cond2 = (mat[im,1] != mat[ikk,1])
                                    dij1[im, ikk] = cond2
                            dij *= dij1

                        #del xp, zp, xmat, zmat, dxij, dzij

                        for ij in range(len(rbin[0:-1])):
                            condr = np.logical_and(dij >= rbin[ij], dij < (rbin[ij] + dr))
                            t1ij  = tij[condr]
                            theta_surf = (dtheta/2) * (2*rbin[ij]*dr + dr**2) #different from michel's code
                            for ik in range(len(thetabin[0:-1])):
                                condnt = np.logical_and(t1ij >= thetabin[ik], t1ij < (thetabin[ik] + dtheta))
                                g_r_theta[ij, ik] += np.sum(condnt)/NP[i]/theta_surf

                        #sys.stdout.write("\r" + " " * 100)  # Clear the line
                        #sys.stdout.flush()

                    g_r_theta /= len(SSi)
#                     txtFile = open(f'{dataname}/run_{l+1}/' + 'PDF_g_r_theta.txt', 'w')

#                     txtFile.write('# r bins \n')
#                     txtFile.write(" ".join(map(str, rbin)))
#                     txtFile.write("\n\n")

#                     txtFile.write('# theta bins \n')
#                     txtFile.write(" ".join(map(str, thetabin)))
#                     txtFile.write("\n\n")

#                     txtFile.write("\n".join(" ".join(map(str, row)) for row in g_r_theta) + "\n")
#                     txtFile.close()
                    #progress_bar.clear()
                    print(f'\n    Done - NP_{str(NP[i])}/phi_{phir}/ar_{str(ar[k])}/run_{l+1}')
            else:
                print(f'{dataname} - Not found')