import numpy             as np          # type:ignore   
import matplotlib                       # type:ignore
import matplotlib.pyplot as plt         # type:ignore   
import os
import glob
from matplotlib.patches import Wedge    # type:ignore   
import matplotlib.gridspec as gridspec  # type:ignore   
matplotlib.use('Qt5Agg')    

'''
Aug 21, 2024
RVP

This script plots the radial pair distribution function for a particular size pair.
And also plots all size pairs together.

command:
python3 -c "from PDF_plots import PDF_plot; PDF_plot('all')"
python3 -c "from PDF_plots import PDF_AllPlots; PDF_AllPlots()"
'''

# Modified plot RC parameters
plt.close('all')
plt.rcParams.update({
  'figure.max_open_warning' : 0,
  'text.usetex'             : True,
  'figure.autolayout'       : True,
  'font.family'             : "STIXGeneral",
  'mathtext.fontset'        : "stix",
  'font.size'               : 10,
  'axes.titlesize'          : 10,
  'figure.labelsize'        : 10,
  'figure.titlesize'        : 10,
  'legend.fontsize'         : 10,
  'legend.handlelength'     : 1,
  'legend.handletextpad'    : 0.5,
  'legend.borderpad'        : 0.5,
  'legend.borderaxespad'    : 0.5,
  'legend.columnspacing'    : 1,
  'legend.framealpha'       : 1,
  'legend.fancybox'         : True,
  'axes.grid'               : True,
  'axes.grid.axis'          : 'both',
  'grid.alpha'              : 0.2,
  'grid.linewidth'          : 0.4,
  'xtick.labelsize'         : 10,
  'ytick.labelsize'         : 10,
  'lines.linewidth'         : 1,
  'lines.markersize'        : 5,
  'savefig.transparent'     : True,
  'savefig.pad_inches'      : 0.01,
  'savefig.format'          : 'pdf',
  'savefig.bbox'            : 'tight'
})
plt.rcParams['text.latex.preamble']= r"\usepackage{amsmath}"

"====================================================================================================================================="

# Simulation data mount point and figure save path.
topDir        = "/Volumes/Rahul_2TB/high_bidispersity"
fig_save_path = "/Users/rahul/City College Dropbox/Rahul Pandare/CUNY/Research/Bidisperse Project/analysis/figures/PDF"

# Some simulation parameters.
NP          = [1000]

run         = {500:8,1000:1,2000:2,4000:1}

phi         = [0.76]

ar          = [1.4, 4.0]

sizePairs   = ['all', 'ss', 'sl', 'll']

def PDF_plot(sizePair = 'all', figsave = True):
    '''
    Plotting PDF radial figures for a size pair

    Input: sizePair - 'all', 'ss', 'sl' or 'll'
    '''
    global sizePairs
    bottomDet = ['All Particles', 'Small-Small Particles', 'Small-large Particles', 'Large-Large Particles']
    
    for i in range(len(NP)):
        for j in range(len(phi)):
            phir = '{:.3f}'.format(phi[j]) if len(str(phi[j]).split('.')[1]) > 2 else '{:.2f}'.format(phi[j])
            for k in range(len(ar)):
                if ar[k] == 1 and sizePair != 'all':
                    print(f"\n     Skipping since ar = 1 and not all pairs are considered (phi = {phir})")
                    continue
                for l in range(run[NP[i]]):
                    dataname = f'{topDir}/NP_{NP[i]}/phi_{phir}/ar_{ar[k]}/Vr_0.5/run_{l+1}'
                    if os.path.exists(dataname): 
                        plt.clf()

                        datFile   = glob.glob(f'{dataname}/PDF_{sizePair}_*.txt')[0]
                        rbin      = np.genfromtxt(datFile, skip_header=1, max_rows=1)
                        thetabin  = np.genfromtxt(datFile, skip_header=2, max_rows=1)
                        g_r_theta = np.genfromtxt(datFile, skip_header=5)
                        
                        fig, ax   = plt.subplots(subplot_kw = {'projection': 'polar'}, figsize=(8,6))
                        
                        ax.clear()
                        ax.grid(False)

                        rlim = 40 # far limit for plotting

                        for ii in range(len(rbin[0:rlim]) - 1):
                            for j in range(len(thetabin) - 1):
                                theta_start = np.degrees(thetabin[j])
                                theta_end   = np.degrees(thetabin[j + 1])
                                r_start     = rbin[ii]
                                r_end       = rbin[ii + 1]

                                wedge = Wedge(
                                    (0, 0), r_end, theta_start, theta_end,
                                    width       = r_end - r_start,
                                    facecolor   = plt.cm.viridis(g_r_theta[ii, j]),
                                    #edgecolor  = (1, 1, 1, 0.3),  # Set the edge color to white
                                    edgecolor   = 'none',
                                    #linewidth  = 0.005, 
                                    antialiased = True,
                                    transform   = ax.transData._b
                                )
                                ax.add_patch(wedge)

                        ax.set_ylim(0, rbin[rlim])
                        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1)) #norm=plt.Normalize(vmin=g_r_theta.min(), vmax=g_r_theta.max()))
                        sm.set_array([])
                        cbar = fig.colorbar(sm, ax=ax, fraction=0.0455, pad=0.05, extend='max')
                        cbar.set_label(r'$g(r,\theta)$', fontsize=12)  
                        ticks = np.arange(0, 2 * np.pi, np.pi/4)  # Define ticks from 0 to 2π every π/4 radians
                        ax.set_xticks(ticks)  # Set the angular ticks
                        
                        del g_r_theta, rbin, thetabin

                        # Label ticks in terms of pi
                        angle_labels = [r'0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$',
                                        r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$']
                        ax.set_xticklabels(angle_labels)
                        ax.set_yticklabels([])
                        ax.spines['polar'].set_visible(False)
                        ax.set_title(f'{bottomDet[sizePairs.index(sizePair)]}', va='bottom', x=0.6)

                        fig.suptitle(fr'$\phi = {phir}, \delta = {ar[k]}$', fontsize=16, x=0.6)

                        #plt.tight_layout()
                        if figsave:
                            figFormat     = ".pdf"
                            plt.savefig(f'{fig_save_path}/PDF_{sizePair}_NP_{NP[i]}_phi_{phir}_ar_{ar[k]}_run_{l+1}{figFormat}', bbox_inches="tight", dpi=300)
                        #plt.show()

"=============================================================================================================================================================="

def PDF_AllPlots(figsave=True):
    '''
    Plotting PDF radial figures for a size pair

    Input: sizePair - 'all', 'ss', 'sl' or 'll'
    '''
    global sizePairs
    titles = ['All', 'Small-Small', 'Small-large', 'Large-Large']

    for i in range(len(NP)):
        for j in range(len(phi)):
            phir = '{:.3f}'.format(phi[j]) if len(str(phi[j]).split('.')[1]) > 2 else '{:.2f}'.format(phi[j])
            for k in range(len(ar)):
                if ar[k] == 1:
                    print(f"\n     Skipping since ar = 1 (phi = {phir})")
                    continue
                for l in range(run[NP[i]]):
                    dataname = f'{topDir}/NP_{NP[i]}/phi_{phir}/ar_{ar[k]}/Vr_0.5/run_{l+1}'
                    if os.path.exists(dataname):
                        plt.clf()

                        fig = plt.figure(figsize = (8, 6))
                        gs  = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.07], wspace=0.3, hspace=0.6)

                        axs.clear()

                        axs = [
                            fig.add_subplot(gs[0, 0], projection='polar'),
                            fig.add_subplot(gs[0, 1], projection='polar'),
                            fig.add_subplot(gs[1, 0], projection='polar'),
                            fig.add_subplot(gs[1, 1], projection='polar')
                        ]

                        for ii, ax in enumerate(axs): 
                            datFile   = glob.glob(f'{dataname}/PDF_{sizePairs[ii]}_*.txt')[0]
                            rbin      = np.genfromtxt(datFile, skip_header=1, max_rows=1)
                            thetabin  = np.genfromtxt(datFile, skip_header=2, max_rows=1)
                            g_r_theta = np.genfromtxt(datFile, skip_header=5)

                            for ij in range(len(rbin) - 1):
                                for j in range(len(thetabin) - 1):
                                    theta_start = np.degrees(thetabin[j])
                                    theta_end   = np.degrees(thetabin[j + 1])
                                    r_start     = rbin[ij]
                                    r_end       = rbin[ij + 1]

                                    wedge = Wedge(
                                        (0, 0), r_end, theta_start, theta_end,
                                        width     = r_end - r_start,
                                        facecolor = plt.cm.viridis(g_r_theta[ij, j]),
                                        edgecolor = 'none',
                                        transform = ax.transData._b
                                    )
                                    ax.add_patch(wedge)

                            ax.set_ylim(0, rbin[40])
                            ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 4))
                            angle_labels = [r'0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$',
                                            r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$']
                            ax.set_xticklabels(angle_labels)
                            ax.set_yticklabels([])
                            ax.spines['polar'].set_visible(False)
                            ax.set_title(titles[ii], va='bottom')

                        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
                        sm.set_array([])

                        del g_r_theta, rbin, thetabin

                        # Colorbar
                        cbar = fig.colorbar(sm, cax=fig.add_subplot(gs[:, 2]), extend='max')
                        cbar.set_label(r'$g(r,\theta)$', fontsize=12)
                        
                        # Figure title and adjusting the subplots position
                        fig.suptitle(fr'$\phi = {phir}, \delta = {ar[k]}$', fontsize=16, y=1.01, x=0.48)
                        plt.subplots_adjust(left=0.1, right=0.8, top=0.85, bottom=0.1, wspace=0.3, hspace=0.5)

                        if figsave:
                            figFormat = ".pdf"
                            plt.savefig(f'{fig_save_path}/PDF_allplots_NP_{NP[i]}_phi_{phir}_ar_{ar[k]}_run_{l+1}{figFormat}', dpi=500)