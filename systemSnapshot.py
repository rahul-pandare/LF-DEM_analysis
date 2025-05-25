import os
import matplotlib                   # type: ignore
import numpy             as     np  # type: ignore
import matplotlib.pyplot as     plt # type: ignore
import colorsys
from matplotlib import colors       # type: ignore
import readFiles
matplotlib.use('TkAgg')

'''
July 22, 2024
RVP

This script produces snapshot at a particular strain unit (here default is set at timestep = 300).
Snapshots are produced for all the phi and ar values mentioned.
NOTE: snapshots produced for just one run (run = 1)

Command to execute in terminal:
python3 -c "from systemSnapshot import snapshot; snapshot"
'''

# Input and output paths.
topDir        = '/Volumes/rahul_2TB/high_bidispersity/new_data/'
fig_save_path = '/Users/rahul/City College Dropbox/Rahul Pandare/CUNY/research/bidisperse_project/figures/for_presentation/system_snapshots/'

# Path errors.
print(f"Error: Path '{topDir}' not found. Check mount point") if not os.path.exists(topDir) else None
print(f"Error: Path '{fig_save_path}' not found. Check mount point") if not os.path.exists(fig_save_path) else None

# Simulation parameters.
npp = 1000

phi = [0.77]
ar  = [1.4]
vr  = '0.5'
numRun = 1

# Particles data file.
parFile = 'par_random_seed_params_stress100r_shear.dat'

cmap      = matplotlib.colormaps['gist_rainbow'] # type: ignore
alpha     = 0.75
hls       = np.array([colorsys.rgb_to_hls(*c) for c in cmap(np.arange(cmap.N))[:,:3]])
hls[:,1] *= alpha
rgb       = np.clip(np.array([colorsys.hls_to_rgb(*c) for c in hls]), 0,1)
cmap      = colors.LinearSegmentedColormap.from_list("", rgb)

frame = 300
#def snapshot(frame = 300):
for j in range(len(phi)):
    phii = phi[j]
    phii = '{:.3f}'.format(phii) if len(str(phii).split('.')[1]) > 2 else '{:.2f}'.format(phii)
    for k in range(len(ar)):
        vrr      = '0.25' if ar[k] == 1.0 else '0.5'
        dataname = f"{topDir}NP_{npp}/phi_{phii}/ar_{ar[k]}/Vr_{vrr}"
        if os.path.exists(dataname):
            with open(f'{dataname}/run_{numRun}/{parFile}', 'r') as particleFile:
                lines         = particleFile.readlines()
                particlesList = readFiles.readParFile(particleFile)

            # Box dimensions.
            Lx = float(lines[3].split()[2])
            Lz = float(lines[3].split()[2])

            # Particle sizes and radii.
            px = particlesList[frame][:, 2]
            pz = particlesList[frame][:, 3]
            pr = particlesList[frame][:, 1]

            # Setting up axis and box walls.
            fig, ax = plt.subplots(figsize=(8, 8))  # Create figure and axes
            newLx   = Lx + 2 * np.max(pr)
            newLz   = Lz + 2 * np.max(pr)

            for i in range(len(particlesList[frame])):
                if pr[i] == 1:
                    circle = plt.Circle((px[i], pz[i]), pr[i], facecolor='#fc8d59', fill=True, edgecolor='None')
                else:
                    circle = plt.Circle((px[i], pz[i]), pr[i], facecolor='#91bfdb', fill=True, edgecolor='None')
                ax.add_artist(circle)

            ax.set_xlim([-(newLx / 2 + 0.2), (newLx / 2 + 0.2)])
            ax.set_ylim([-(newLz / 2 + 0.2), (newLz / 2 + 0.2)])
            ax.axis('off')
            ax.set_aspect('equal')

            # Saving figure with transparent background.
            figFormat = ".png"
            plt.savefig(f"{fig_save_path}snapshot_phi_{phii}_ar_{ar[k]}_vr_{vr}{figFormat}", bbox_inches="tight", dpi=800, transparent=True)

            print(f'      > Snapshot saved for phi = {phii}, ar = {ar[k]}, vr = {vr}')
            plt.close(fig)  # Close the figure to free up memory
        else:
            print(f"{dataname} - Not Found")