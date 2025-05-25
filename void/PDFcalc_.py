import os
import glob
import numpy as np            # type: ignore
from   tqdm  import tqdm      # type: ignore
import readFiles              # type: ignore
import multiprocessing as mp

'''
Mar 18, 2025
RVP
Parallel version: This script calculates the pair distribution function (PDF) using annular sector elements.
It parallelizes over radial bins to speed up the g(r, θ) calculation.
'''

topDir     = "/Volumes/rahul_2TB/high_bidispersity/new_data"
npp        = 1000
numRuns    = 1
vr         = '0.75'
sizePairs  = ['all'] #['ss', 'sl', 'll', 'all']
off        = 100

# below lists should be same size
phi        = [0.76]
ar         = [1.4]

# discretization parameters
dangle = 5    # in degrees
dr     = 0.2  # in unit length

def compute_bin(args):
    ij, dij, tij, thetabin, dtheta, dr, rbin_val, npp = args

    condr = np.logical_and(dij >= rbin_val, dij < (rbin_val + dr))
    t1ij = tij[condr]
    theta_surf = (dtheta / 2) * (2 * rbin_val * dr + dr ** 2)

    row_vals = np.zeros(len(thetabin) - 1)
    for ik in range(len(thetabin) - 1):
        condt = np.logical_and(t1ij >= thetabin[ik], t1ij < (thetabin[ik] + dtheta))
        row_vals[ik] = np.sum(condt) / npp / theta_surf

    return ij, row_vals

def main():
    for sizePair in sizePairs:
        for i, phii in enumerate(phi):
            phir = '{:.3f}'.format(phii) if len(str(phii).split('.')[1]) > 2 else '{:.2f}'.format(phii)
            if ar[i] == 1 and sizePair != 'all':
                print(f"     Skipping since ar = 1 and not all pairs are considered (phi = {phir})\n")
                continue

            for l in range(numRuns):
                dataname = f'{topDir}/NP_{npp}/phi_{phir}/ar_{ar[i]}/Vr_{vr}/run_{l+1}'
                if os.path.exists(dataname):
                    outFile = f'{dataname}/PDF_{sizePair}_g_r_theta_.txt'
                    if os.path.exists(outFile):
                        print(f'     PDF file already exists skipping - phi_{phir}/ar_{ar[i]}/Vr_{vr}/run_{l+1}\n')
                        continue

                    print(f'  Working on - phi_{phir}/ar_{ar[i]}/Vr_{vr}/run_{l+1}\n')
                    ranSeedFile = glob.glob(f'{dataname}/random_*.dat')[0]
                    datFile     = glob.glob(f'{dataname}/data_*')[0]
                    parFile     = glob.glob(f'{dataname}/par_*')[0]

                    particleSize = np.genfromtxt(ranSeedFile, skip_header=2, usecols=-1)
                    parList      = readFiles.parametersList(parFile)

                    lx = np.genfromtxt(datFile, skip_header=3, max_rows=1, comments='_')[2]
                    lz = np.genfromtxt(datFile, skip_header=5, max_rows=1, comments='_')[2]

                    data = np.loadtxt(datFile, skiprows=37).transpose()
                    gamma, minGap = data[1], data[13]

                    dtheta    = dangle * np.pi / 180
                    rmin      = (np.max(particleSize) + np.min(minGap)) if sizePair == 'll' else (np.min(particleSize) + np.min(minGap))
                    rmax      = np.max([lx, lz]) / 2.
                    rbin      = np.arange(rmin, rmax + dr, dr)
                    thetabin  = np.arange(-np.pi, np.pi + dtheta, dtheta)
                    g_r_theta = np.zeros((len(rbin), len(thetabin)))

                    SSi = parList[off:]
                    for ii, mat in tqdm(enumerate(SSi), desc="Time steps", leave=False, total=len(SSi)):
                        xp, zp = mat[:, 2], mat[:, 3]

                        if sizePair in ['ss', 'll']:
                            cond = (mat[:, 1] == 1) if sizePair == 'ss' else (mat[:, 1] > 1)
                            xp, zp = xp[cond], zp[cond]

                        xmat, zmat = np.outer(xp, np.ones(len(xp))), np.outer(zp, np.ones(len(xp)))
                        dxij, dzij = xmat.T - xmat, zmat.T - zmat

                        # Lees Edwards boundaries
                        dxij[dzij > lz / 2.] -= gamma[ii] * lx
                        dzij[dzij > lz / 2.] -= lz
                        dxij[dzij < -lz / 2.] += gamma[ii] * lx
                        dzij[dzij < -lz / 2.] += lz

                        dxij[dxij > lx / 2.] -= lx
                        dxij[dxij < -lx / 2.] += lx

                        dij = np.sqrt(dxij ** 2 + dzij ** 2)
                        tij = np.arctan2(dzij, dxij)

                        if sizePair == 'sl':
                            dij1 = np.zeros_like(dij)
                            for im in range(dij.shape[0]):
                                for ikk in range(dij.shape[1]):
                                    dij1[im, ikk] = mat[im, 1] != mat[ikk, 1]
                            dij *= dij1

                        # Parallel computation over radial bins
                        args = [(ij, dij.copy(), tij.copy(), thetabin, dtheta, dr, rbin[ij], npp) for ij in range(len(rbin) - 1)]
                        with mp.Pool() as pool:
                            results = pool.map(compute_bin, args)

                        for ij, row_vals in results:
                            g_r_theta[ij, :len(row_vals)] += row_vals

                    g_r_theta /= len(SSi)
                    g_r_theta /= np.mean(g_r_theta)

                    with open(outFile, 'w') as txtFile:
                        txtFile.write('# r bins \n')
                        txtFile.write(" ".join(map(str, rbin)) + "\n\n")
                        txtFile.write('# theta bins \n')
                        txtFile.write(" ".join(map(str, thetabin)) + "\n\n")
                        txtFile.write("\n".join(" ".join(map(str, row)) for row in g_r_theta) + "\n")

                    print(f'\n    Done - NP_{npp}/phi_{phir}/ar_{ar[i]}/Vr_{vr}/run_{l+1}\n')
                    del g_r_theta
                else:
                    print(f'{dataname} - Not found')

if __name__ == '__main__':
    mp.set_start_method("fork")
    main()