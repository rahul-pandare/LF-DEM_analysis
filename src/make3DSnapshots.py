import os
import glob
import platform
from   tqdm              import tqdm      # type: ignore
from   pathlib           import Path      # type: ignore
import numpy             as     np        # type: ignore
import matplotlib.pyplot as     plt       # type: ignore
import src.readFiles     as     readFiles # type: ignore
import pyvista as pv                      # type: ignore
import gc

'''
Dec 17, 2025
RVP

This script produces snapshots for particles in 3D dense suspensions.
'''
plt.rcParams.update({
    "figure.max_open_warning": 0,
    "text.usetex": True,
    "figure.autolayout": True,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size":        10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "patch.linewidth":  .2,
    "lines.markersize":  5,
    "hatch.linewidth":  .2
})
plt.rcParams['text.latex.preamble']= r"\usepackage{amsmath}"
#matplotlib.use('Agg')

system_platform = platform.system()

if system_platform == 'Darwin':  # macOS
    topDir = Path("/home/rahul/City College Dropbox/Rahul Pandare/CUNY/research/bidisperse_project/miscelleneous/sharepoint_Rahul_Brolin/new 3D data/")
    figsavepath = Path("/Users/rahul/City College Dropbox/Rahul Pandare/CUNY/research/bidisperse_project/figures/3d/")
elif system_platform == 'Linux':
    topDir = Path("/home/rahul/City College Dropbox/Rahul Pandare/CUNY/research/bidisperse_project/miscelleneous/sharepoint_Rahul_Brolin/new 3D data/")
    figsavepath = Path("/home/rahul/City College Dropbox/Rahul Pandare/CUNY/research/bidisperse_project/figures/3d/")
else:
    raise OSError(f"Unsupported OS: {system_platform}")

# Validate paths
for path in [topDir, figsavepath]:
    if not path.exists():
        print(f"Error: Path '{path}' not found. Check mount point.")

# Simulation parameters.
npp    = 1000
stress = ['1r']#, '10r', '100r']
phi    = [0.56]#, 0.55, 0.56]
ar     = [1.4]
vr     = ['0.5']
numRum = 1

# Particles data file.
parFile = 'par_*.dat'

# Frame details
startFrame = 100
endFrame   = 300
#or
#frames     = [101, 224, 228]

for i, s in enumerate(stress):
    for j, phii in enumerate(phi):
        phii = '{:.3f}'.format(phii) if len(str(phii).split('.')[1]) > 2 else '{:.2f}'.format(phii)
        for k, arr in enumerate(ar):
            for l, vrr in enumerate(vr):
                dataname = topDir / f"Stress {s}/phi_{phii}/ar_{arr}/Vr_{vrr}"
                if os.path.exists(dataname):
                    particleFile  = open(glob.glob(f'{dataname}/run_{numRum}/{parFile}')[0])
                    parLines      = particleFile.readlines()
                    particlesList = readFiles.readParFile(particleFile)

                    # Box dimensions.
                    lx = float(parLines[3].split()[2])
                    ly = float(parLines[4].split()[2]) 
                    lz = float(parLines[5].split()[2])

                    directory = f'{figsavepath}/Stress_{s}_phi_{phii}'
                    os.makedirs(directory, exist_ok=True)
                    
                    for frame in tqdm(range(startFrame, endFrame), desc="Outer loop"):
                    #for frame in tqdm(frames, desc="Inner loop", leave=False):

                        # particle positions and radii
                        r = particlesList[frame][:, 1]
                        x = particlesList[frame][:, 2]
                        y = particlesList[frame][:, 3]
                        z = particlesList[frame][:, 4]
                        
                        points = np.column_stack([x, y, z])
                        
                        ######################################################################################
                        # Create sphere glyphs
                        cloud = pv.PolyData(points)
                        cloud["diameter"] = r * 2
                        
                        spheres = cloud.glyph(
                            geom=pv.Sphere(theta_resolution=64, phi_resolution=64),
                            scale="diameter",
                            orient=False
                        )

                        # Plotter object
                        p = pv.Plotter(off_screen=True)
                        
                        ## If want to see translucent paticles at boundaries uncomment below
                        # opacity = np.ones(len(x))
                        
                        # for i in range(len(x)):
                        #     if np.any(points[i, :] > (lx/2)*0.95) or np.any(points[i, :] < (-lx/2)*0.95):
                        #         opacity[i] = 0.3

                        # for i, pt in enumerate(points):
                        #     sphere = pv.Sphere(center=pt, radius=r[i], theta_resolution=64, phi_resolution=64)
                        #     p.add_mesh(sphere, color="#9c8c49", smooth_shading=True, specular=0.8, specular_power=30, opacity=opacity[i])
                        
                        p.add_mesh( spheres, color="#9c8c49", smooth_shading=True, specular=0.8, specular_power=30 ) # comment this line if using above loop
                        
                        #translucent simulation box
                        box = pv.Box(bounds=(-lx/2, lx/2, -lx/2, lx/2, -lx/2, lx/2))
                        
                        p.add_mesh(box, color="lightblue", opacity=0.2, style="surface",     
                                   show_edges=True, edge_color="black",line_width=3)     
                        
                        ######################################################################################

                        # Camera angle and lighting
                        distance = lx * 3.0    # distance from center
                        elev = np.deg2rad(25)  # degrees to radians
                        azim = np.deg2rad(305)
                        
                        camx = distance * np.cos(elev) * np.cos(azim)
                        camy = distance * np.cos(elev) * np.sin(azim)
                        camz = distance * np.sin(elev)
                        
                        p.camera_position = ((camx, camy, camz), # camera location
                                             (0.5, -1.5, -1.5),    # focal point (center)
                                             (0, 0, 1))          # up vector (z-axis)

                        # Lighting
                        light = pv.Light(
                                position    = (camx*1.05, camy*1.05, camz*2), # light position
                                focal_point = (0, 0, 0),
                                intensity   = 0.8)
                        
                        p.add_light(light)
                        
                        ######################################################################################
                        
                        # top and bottom arrows
                        tstart = (-lx*0.1, -lx*0.3,    lx/2 + 0.1*lx)  
                        bstart = ( lx*0.1, -lx/2*1.1, -lx/2 - 1.5) 

                        tarrow = pv.Arrow(start=tstart, direction=(1, 0, 0), tip_length=0.1,
                             tip_radius=0.05, shaft_radius = 0.03, scale=10, tip_resolution=100,shaft_resolution=500)
                        
                        barrow = pv.Arrow(start=bstart, direction=(-1, 0, 0), tip_length=0.1, tip_radius=0.05,
                            shaft_radius = 0.03, scale=10, tip_resolution=100,shaft_resolution=500)
                        
                        # Add arrows to plotter
                        p.add_mesh(tarrow, color="red", smooth_shading=True) 
                        p.add_mesh(barrow, color="blue", smooth_shading=True)

                        p.add_axes(line_width=2, color='black', labels_off=False)

                        #p.set_background([1, 1, 1, 1]) # white background [1, 1, 1, 1(opacity)], transperent background [1, 1, 1, 0(opacity)]
                        p.set_background("white")
                        #p.add_title(fr'$\sigma/\sigma_0 = {s[:-1]},\; \phi = {phii},\; \gamma = {frame/100:.2f}$', font_size=24, color="black") #not working

                        # Saving figure
                        p.screenshot(f'{directory}/frame_{frame}.png', transparent_background=False, window_size=[600,600])
                        #p.show()
                        #print(f'{frame}')
                        p.close()
                        del p
                        gc.collect()