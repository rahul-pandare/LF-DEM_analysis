import os
import glob
import numpy as np
from tqdm import tqdm
import readFiles

'''
May 26, 2025
RVP

Script to calculate the virial stress tensor componenets per particle in LF-DEM.
The output has all 4 componenets (xx, xy, yx, yy) of the contact stress, lubrication stress
and the total stress tensor for all particles
'''

# Simulation data mount point
topDir = "/Volumes/rahul_2TB/high_bidispersity/new_data/"
#topDir  = "/media/rahul/rahul_2TB/high_bidispersity/new_data/"

# Some simulation parameters.
npp  = 1000
runs = 2
phi  = [0.77]
vr   = ['0.5']
ar   = [1.4] #[1.0, 1.4, 2.0, 4.0]
off  = 100
tstrain = 0.01

for i, phii in enumerate(phi):
    phir = '{:.3f}'.format(phii) if len(str(phii).split('.')[1])>2 else '{:.2f}'.format(phii)
    for j, arj in enumerate(ar):
        for k, vrk in enumerate(vr):
            for l in range(runs):
                dataname = f"{topDir}NP_{npp}/phi_{phir}/ar_{arj}/Vr_{vrk}/run_{l}"
                if os.path.exists(dataname):
                    print(f'Calculating stress tensors for - phi_{phir}/ar_{arj}/vr_{vrk}/run_{l+1}\n')

                    ranSeedFile = glob.glob(f'{dataname}/random_*.dat')[0] #"random_seed.dat"
                    parFile     = glob.glob(f'{dataname}/par_*')[0]
                    intFile     = glob.glob(f'{dataname}/int_*')[0]
                    
                    # Readind particle sizes and reading parameters files into a list
                    radList = readFiles.particleSizeList(open(ranSeedFile, 'r'), arj, npp)
                    parList = readFiles.readParFile(open(parFile, 'r'))
                    intList = readFiles.interactionsList(open(intFile, 'r'))

                    stress   = float(parFile[-14:-11])
                    filename = f'{dataname}/virial_stress_tensor.txt'
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
                                stress_tensor_cont = np.zeros((2, 2))
                                stress_tensor_lub  = np.zeros((2, 2))
                                stress_tensor_tot  = np.zeros((2, 2))
                                parea = float(np.pi * radList[pindex]**2)
                                interaction_in_frame = intframe.shape[0]
                                for jj in range(interaction_in_frame):
                                    if intframe[jj][0] == pindex or intframe[jj][1] == pindex:
                                        sign     = 1 if intframe[jj][0] == pindex else -1
                                        norm_vec = sign * np.array([intframe[jj][2], intframe[jj][4]])
                                        
                                        fn_cont = float(intframe[jj][11]) * norm_vec
                                        ft_cont = np.array([intframe[jj][12], intframe[jj][14]])
                                        fn_lub  = float(intframe[jj][6]) * norm_vec
                                        ft_lub  = np.array([intframe[jj][7], intframe[jj][9]])
    
                                        p1 = int(intframe[jj][0]) # index of P1
                                        p2 = int(intframe[jj][1]) # index of P2
                                        p1_pos = np.array([parList[ii][p1][2], parList[ii][p1][3]])
                                        p2_pos = np.array([parList[ii][p2][2], parList[ii][p2][3]])
                                        
                                        overlap_dist = np.linalg.norm(p1_pos - p2_pos) - (radList[p1] + radList[p2])
                                        contact_vec  = norm_vec * float(radList[pindex] + overlap_dist/2)
                                        
                                        stress_tensor_cont += (1/parea)*(np.outer(fn_cont, contact_vec) + np.outer(ft_cont, contact_vec))
                                        stress_tensor_lub  += (1/parea)*(np.outer(fn_lub , contact_vec) + np.outer(ft_lub , contact_vec))
                                        stress_tensor_tot  += (1/parea)*(np.outer(fn_cont, contact_vec) + np.outer(ft_cont, contact_vec)
                                                                       + np.outer(fn_lub , contact_vec) + np.outer(ft_lub , contact_vec))

                                all_stress_comp = [int(pindex), *stress_tensor_cont.flatten().tolist(), 
                                                   *stress_tensor_lub.flatten().tolist() ,
                                                   *stress_tensor_tot.flatten().tolist()]
    
                                stress_particle.append(all_stress_comp)

                            strain += tstrain
                            stress_array = np.array(stress_particle)

                            file.write(f'\n#cumulative strain = {strain:.2f}\n')
                            for row in stress_array:
                                formatted_row = f"{int(row[0])} " + " ".join(f"{num:.6f}" for num in row[1:])
                                file.write(formatted_row + "\n")

                print(f'Done - phi_{phir}/ar_{arj}/vr_{vrk}/run_{l+1}\n')