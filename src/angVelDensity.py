import os
import glob
import numpy             as np  # type: ignore
import matplotlib.pyplot as plt # type: ignore
from   pathlib           import Path
import readFiles

'''
Jul 3, 2025
Density distribution of angular velocity of particles in a suspension (normalised by shear rate).
'''
plt.rcParams["text.usetex"] = True # LaTeX rendering

# Parametrers
npp     = 1000
phi     = 0.55
ar      = '1.4'
vr      = '0.5'
numRuns = 2
off     = 100
wRange  = 15 # range of angular velocity for histogram

particleFile  = 'par_*.dat'
dataFile      = 'data_*.dat'
rigFile       = 'rig_*.dat'
cmap          = plt.get_cmap("coolwarm")

angVelAllx, angVelAlly,  angVelAllz= [], [], []
phir     = f"{phi:.3f}" if phi != round(phi, 2) else f"{phi:.2f}"
topDir   = Path(__file__).resolve().parent.parent  # Change this to the working path of your project
dataname = f"{topDir}/NP_{npp}/phi_{phir}/ar_{ar}/Vr_{vr}"

if os.path.exists(dataname):
    for k in range(numRuns):
        par_file = glob.glob(f'{dataname}/run_{k+1}/{particleFile}')[0]
        dat_file = glob.glob(f'{dataname}/run_{k+1}/{dataFile}')[0]
        rig_file = glob.glob(f'{dataname}/run_{k+1}/{rigFile}')[0]
        
        if par_file and rig_file:
            data    = np.loadtxt(dat_file)
            parList = readFiles.readParFile(open(par_file))
            rigList = readFiles.rigList(open(rig_file))
            srate   = data[:, 2]
            
            for frame in range(off, len(parList)):
                frameList   = parList[frame]
                rigParts    = [i for sublist in rigList[frame] for i in sublist]
                nonRigParts = list(set(np.arange(npp)) - set(rigParts))
    
                # Reading angylar velocity (and normalising by shear rate)
                angVelx = frameList[rigParts, 8]/srate[frame] #nomalised
                angVely = frameList[rigParts, 9]/srate[frame] #nomalised
                angVelz = frameList[rigParts,10]/srate[frame] #nomalised

                angVelAllx.extend(angVelx)
                angVelAlly.extend(angVely)
                angVelAllz.extend(angVelz)
                
    angVelAllRig    = np.array(angVelAllx) 
    hist, bins      = np.histogram(angVelAllRig, bins=100, range=(-wRange, wRange), density=True)
    bin_centers     = 0.5 * (bins[:-1] + bins[1:])
    plt.plot(bin_centers, hist, label = 'x', color = 'red', linewidth=1.5)
    
    angVelAllRig    = np.array(angVelAlly) 
    hist, bins      = np.histogram(angVelAllRig, bins=100, range=(-wRange, wRange), density=True)
    bin_centers     = 0.5 * (bins[:-1] + bins[1:])
    plt.plot(bin_centers, hist, label = 'y', color = 'blue', linewidth=1.5)
    
    angVelAllRig    = np.array(angVelAllz)
    hist, bins      = np.histogram(angVelAllRig, bins=100, range=(-wRange, wRange), density=True)
    bin_centers     = 0.5 * (bins[:-1] + bins[1:])
    plt.plot(bin_centers, hist, label = 'z', color = 'green', linewidth=1.5)

#plt.axvline(x=0.5, color='k', linestyle='--', linewidth=1.2, alpha=0.6)

plt.xlabel(r'$\omega/ \dot \gamma$', fontsize=16, fontweight='bold')
plt.ylabel(r'$P(n)$', fontsize=14, fontweight='bold')
plt.title(fr'$\phi = {phi}, \; \delta = {ar}, \; \zeta = {float(vr):.2f}$', fontsize=16)
plt.legend(fontsize=12, loc='upper left', labelspacing=1.2, borderpad=1.1, framealpha=0.5, ncol=1)
plt.xticks(fontsize=13, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
#plt.grid(True, alpha=0.2)
plt.grid(False)

plt.savefig(f'{fig_save_path}/angVel_rigid_phi{phir}ar{ar}_vr_{vr}.pdf', bbox_inches="tight", dpi=400)
plt.show()