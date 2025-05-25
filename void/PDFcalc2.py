import os
import glob
import numpy as np      # type: ignore
from   tqdm import tqdm # type: ignore
import readFiles        # type: ignore

'''
Apr 15, 2025
RVP
This script calculates the pair distribution function for a particular size pair with square pixels.

command:
python3 -c "from PDFcalc2 import PDF; PDF('all')"
'''

# Simulation data mount point
topDir = "/Volumes/rahul_2TB/high_bidispersity/new_data"
#topDir  = "/media/rahul/rahul_2TB/high_bidispersity/new_data"

# Simulation parameters.
npp     = 1000
phi     = [0.76]
ar      = [1.4]
vr      = '0.75'
numRuns = 1

# PDF parameters
dr  = 0.62   # in unit length
off = 100    # no. of timesteps to skip for steady state

def PDF(sizePair = 'all'):
    '''
    This function calculates the pair distribution function density for a particular size pair

    Input: sizePair - 'all', 'ss', 'sl' or 'll'
    '''
    global dr, off

    for j in range(len(phi)):
        phir = '{:.3f}'.format(phi[j]) if len(str(phi[j]).split('.')[1])>2 else '{:.2f}'.format(phi[j])
        if ar[j] == 1 and sizePair != 'all':
            print(f"     Skipping since ar = 1 and not all pairs are considered (phi = {phir})\n")
            continue  # Skip the rest of this iteration of the 'ar' loop

        for l in range (numRuns):
            dataname = f'{topDir}/NP_{npp}/phi_{phir}/ar_{ar[j]}/Vr_{vr}/run_{l+1}'
            if os.path.exists(dataname):
                # if os.path.exists(f'{dataname}/PDF_{sizePair}_g_r_theta.txt'):
                #     print(f'     PDF file already exists skipping - phi_{phir}/ar_{ar[j]}/Vr_{vr}/run_{l+1}\n')
                #     continue

                print(f'  Working on - phi_{phir}/ar_{ar[j]}/Vr_{vr}/run_{l+1}\n')
                ranSeedFile = glob.glob(f'{dataname}/random_*.dat')[0] #"random_seed.dat"
                datFile     = glob.glob(f'{dataname}/data_*')[0]
                parFile     = glob.glob(f'{dataname}/par_*')[0]
        
                # Readind particle sizes and reading parameters files into a list
                particleSize = np.genfromtxt(ranSeedFile, skip_header=2, usecols=-1)
                parList      = readFiles.parametersList(parFile)
        
                # Box dimensions
                lx = np.genfromtxt(datFile, skip_header=3, max_rows=1, comments = '_')[2]
                lz = np.genfromtxt(datFile, skip_header=5, max_rows=1, comments = '_')[2]
        
                # Reading simulation results
                data = np.loadtxt(datFile, skiprows=37).transpose()
                gamma, minGap = data[1], data[13]
        
                # Bin parameters
                #rmin    = (np.max(particleSize) + np.min(minGap)) if sizePair == 'll' else (np.min(particleSize) + np.min(minGap))
                rmax    = np.max([lx,lz])/2.
                rbin    = np.arange(-rmax, rmax + dr, dr)
                pixSurf = dr**2. # area of a pixel
                gxy     = np.zeros((len(rbin), len(rbin))) # initializing the PDF array
                SSi     = parList[off:] # parameter arrays for all time steps to consider
        
                for _, (ii, mat) in tqdm(enumerate(enumerate(SSi)), desc="Progress", leave=False, total=len(SSi)):
                    xp, zp, rp = mat[:,2], mat[:,3], mat[:,1] # all particle co-ordinates and radii
        
                    if sizePair in ['ss', 'll']:
                        cond   = (rp == 1) if sizePair == 'ss' else None
                        cond   = (rp  > 1) if sizePair == 'll' else None
                        xp, zp = xp[cond], zp[cond]
        
                    # Inter-particle distances array 
                    xmat, zmat = np.outer(xp, np.ones(len(xp))), np.outer(zp, np.ones(len(xp)))
                    dxij, dzij = xmat.transpose() - xmat,        zmat.transpose() - zmat
                    
                    # Lees Edwards boundary:
                    dxij[dzij >  lz/2.] -= gamma[ii]*lx
                    dzij[dzij >  lz/2.] -= lz
                    
                    dxij[dzij < -lz/2.] += gamma[ii]*lx
                    dzij[dzij < -lz/2.] += lz
                    
                    # X peridodic:
                    dxij[dxij >  lx/2.] -= lx
                    dxij[dxij < -lx/2.] += lx
                
                    if sizePair == 'sl':
                        for im in range(dxij.shape[0]):
                            for ikk in range(dzij.shape[1]):
                                cond2 = (mat[im,1] != mat[ikk,1])
                                dxij[im, ikk] *= cond2
                                dzij[im, ikk] *= cond2
        
                    del xp, zp, xmat, zmat

                    for ij in range(len(rbin[0:-1])):
                        for ik in range(len(rbin[0:-1])):
                            condxy = ((dxij >= rbin[ij]) & (dxij < rbin[ij] + dr) &
                                      (dzij >= rbin[ik]) & (dzij < rbin[ik] + dr))
                            gxy[ij, ik] += np.sum(condxy)/pixSurf
                            
                gxy /= len(SSi)
                gxy /= np.mean(gxy)
            
                # Writing the calculated PDF array into a text file
                txtFile = open(f'{dataname}/PDF_{sizePair}_gxy.txt', 'w')
                txtFile.write('# x bins \n')
                txtFile.write(" ".join(map(str, rbin)))
                txtFile.write("\n\n")
        
                txtFile.write('# y bins \n')
                txtFile.write(" ".join(map(str, rbin)))
                txtFile.write("\n\n")
        
                txtFile.write("\n".join(" ".join(map(str, row)) for row in gxy) + "\n")
                txtFile.close()
        
                print(f'\n    Done - NP_{str(npp)}/phi_{phir}/ar_{str(ar[j])}/Vr_{vr}/run_{l+1}\n')
                del gxy
                
            else:
                print(f'{dataname} - Not found')