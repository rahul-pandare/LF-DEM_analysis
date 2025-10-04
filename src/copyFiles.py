import platform
import shutil
from   pathlib import Path

'''
Jun 05, 2025
RVP

Script to copy files onto a different path with similar path built
'''

# Set base paths depending on OS
if platform.system() == 'Darwin':  # macOS
    topDir  = Path("/Volumes/rahul_2TB/high_bidispersity/new_data/")
    copyDir = Path("/Users/rahul/Library/CloudStorage/Box-Box/suspensions/2D_DATA/mu_infinity/sigma_100")
elif platform.system() == 'Linux':
    topDir  = Path("/media/rahul/rahul_2TB/high_bidispersity/new_data/")
    #copyDir = Path("/media/rahul/CloudStorage/Box-Box/suspensions/2D_DATA/mu_infinity/sigma_100")
else:
    raise OSError("Unsupported OS")

# Parameters
npp   = 1000
runs  = 1
phi   = [0.72, 0.74, 0.75, 0.76, 0.765, 0.77, 0.78]
vr    = ['0.5']
ar    = [1.4]
files = ['data_*.dat', 'int_*.dat', 'par_*.dat', 'rig_*.dat']

for phii in phi:
    phir = '{:.3f}'.format(phii) if len(str(phii).split('.')[1]) > 2 else '{:.2f}'.format(phii)
    for arj in ar:
        for vrk in vr:
            for run in range(1, runs + 1):
                dataname = topDir / f"NP_{npp}" / f"phi_{phir}" / f"ar_{arj}" / f"Vr_{vrk}" / f"run_{run}"
                if dataname.exists():
                    for pattern in files:
                        for src_file in dataname.glob(pattern):
                            rel_path  = src_file.relative_to(topDir / f"NP_{npp}") # reletive path starts after NP_{npp}
                            dest_path = copyDir / rel_path
                            dest_path.parent.mkdir(parents=True, exist_ok=True) # if path exiths
                            shutil.copy2(src_file, dest_path) # copying file
                    
                    print(f'Files copied to box for - phi_{phir}/ar_{arj}/Vr_{vrk}/run_{run}')