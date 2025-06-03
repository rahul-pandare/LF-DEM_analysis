import os
import glob
import numpy as np     #type: ignore
from tqdm import tqdm  #type: ignore
import platform
import importlib
import readFiles
importlib.reload(readFiles)

'''
May 26, 2025
RVP

Script to calculate the virial stress tensor componenets per particle in LF-DEM.
The output has all 4 componenets (xx, xy, yx, yy) of the contact stress, lubrication stress
and the total stress tensor for all particles
'''
# Simulation data mount point
if platform.system() == "Darwin":
    # macOS
    topDir = "/Volumes/rahul_2TB/high_bidispersity/new_data/"
elif platform.system() == "Linux":
    topDir = "/media/rahul/rahul_2TB/high_bidispersity/new_data/"
else:
    raise OSError("Unsupported operating system")

# Some simulation parameters.
npp  = 1000
runs = 2
phi  = [0.77]
vr   = ['0.5']
ar   = [1.4] #[1.0, 1.4, 2.0, 4.0]
off  = 0

tstrain   = 0.01
overwrite = True # overwrite the existing files

for i, phii in enumerate(phi):
    phir = '{:.3f}'.format(phii) if len(str(phii).split('.')[1])>2 else '{:.2f}'.format(phii)
    for j, arj in enumerate(ar):
        for k, vrk in enumerate(vr):
            for l in range(runs):
                dataname = f"{topDir}NP_{npp}/phi_{phir}/ar_{arj}/Vr_{vrk}/run_{l+1}"
                if os.path.exists(dataname):
                    filename = f'{dataname}/virial_stress_tensor.txt'
                    if not os.path.exists(filename) or overwrite:
                        if os.path.exists(filename):
                            print(f'\nOverwriting existing stress file: phi_{phir}/ar_{arj}/vr_{vrk}/run_{l+1}\n')
                        else:
                            print(f'\nCalculating stress tensors for - phi_{phir}/ar_{arj}/vr_{vrk}/run_{l+1}\n')

                        ranSeedFile = glob.glob(f'{dataname}/random_*.dat')[0] #"random_seed.dat"
                        parFile     = glob.glob(f'{dataname}/par_*')[0]
                        intFile     = glob.glob(f'{dataname}/int_*')[0]
                        
                        # Readind particle sizes and reading parameters files into a list
                        radList = readFiles.particleSizeList(open(ranSeedFile, 'r'), arj, npp)
                        parList = readFiles.readParFile(open(parFile, 'r'))
                        intList = readFiles.interactionsList(open(intFile, 'r'))
                        stress  = float(parFile[-14:-11])

                        with open(filename, 'w') as file:
                            file.write('# npp phi ar vr stress\n')
                            file.write(f'# {npp} {phir} {arj} {vrk} {stress}\n\n')
                            file.write('#Virial stress tensor components per particle\n')
                            file.write('# 0   : Particle index\n')
                            file.write('# 1-4 : Contact stress tensor elements (sigma_xx, sigma_xy, sigma_yx, sigma_yy)\n')
                            file.write('# 5-8 : Lubrication stress tensor elements (sigma_xx, sigma_xy, sigma_yx, sigma_yy)\n')
                            file.write('# 9-12: Total stress tensor elements (sigma_xx, sigma_xy, sigma_yx, sigma_yy)\n')
        
                            ## calulating virial stress tensor
                            strain = -tstrain + off/100
                            for ii, intframe in enumerate(tqdm(intList[off:])):
                                stress_particle = []
                                for pindex in range(npp):
                                    stress_tensor_cont   = np.zeros((2, 2))
                                    stress_tensor_lub    = np.zeros((2, 2))
                                    stress_tensor_tot    = np.zeros((2, 2))
                                    parea                = float(np.pi * radList[pindex]**2)
                                    interaction_in_frame = intframe.shape[0]
                                    for jj in range(interaction_in_frame):
                                        p1 = int(intframe[jj][0]) # index of P1
                                        p2 = int(intframe[jj][1]) # index of P2
                                        if pindex in (p1, p2):
                                            norm_vec = np.array([intframe[jj][2], intframe[jj][4]])
                                            fn_cont  = float(intframe[jj][11]) * norm_vec
                                            ft_cont  = np.array([intframe[jj][12], intframe[jj][14]])
                                            fn_lub   = float(intframe[jj][6]) * norm_vec
                                            ft_lub   = np.array([intframe[jj][7], intframe[jj][9]])
                                            p1_pos   = np.array([parList[ii][p1][2], parList[ii][p1][3]])
                                            p2_pos   = np.array([parList[ii][p2][2], parList[ii][p2][3]])
                                            dist_ij  = p1_pos - p2_pos
                                            
                                            stress_tensor_cont += 1/2 * (np.outer(fn_cont + ft_cont, dist_ij) + np.outer(dist_ij, fn_cont + ft_cont))
                                            stress_tensor_lub  += 1/2 * (np.outer(fn_lub  + ft_lub,  dist_ij) + np.outer(dist_ij, fn_lub  + ft_lub ))
                                            stress_tensor_tot  += 1/2 * (np.outer(fn_cont + ft_cont, dist_ij) + np.outer(dist_ij, fn_cont + ft_cont)
                                                                      +  np.outer(fn_lub  + ft_lub,  dist_ij) + np.outer(dist_ij, fn_lub  + ft_lub ))

                                    all_stress_comp = np.array([int(pindex)] + 
                                                               (stress_tensor_cont.flatten() / parea).tolist() +
                                                               (stress_tensor_lub.flatten()  / parea).tolist() +
                                                               (stress_tensor_tot.flatten()  / parea).tolist())

        
                                    stress_particle.append(all_stress_comp)

                                strain       += tstrain
                                stress_array  = np.array(stress_particle)

                                file.write(f'\n#cumulative strain = {strain:.2f}\n')
                                for row in stress_array:
                                    formatted_row = f"{int(row[0])} " + " ".join(f"{num:.6f}" for num in row[1:])
                                    file.write(formatted_row + "\n")

                        print(f'\nDone - phi_{phir}/ar_{arj}/vr_{vrk}/run_{l+1}\n')
                    else:
                        print(f'\n File already exits (no overwrite) - phi_{phir}/ar_{arj}/vr_{vrk}/run_{l+1}\n')
                else:
                    print(f'DNE - {dataname}\n')