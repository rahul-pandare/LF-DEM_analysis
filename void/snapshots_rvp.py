import os
import re
import sys
import glob
import shutil
import colorsys
import matplotlib
import myRigidClusterProcessor
import numpy             as     np
import matplotlib.pyplot as     plt
from   matplotlib        import colors
from   collections       import defaultdict


#%% SNAPSHOTS

def make_snapshots(Dir):
    
    plt.close('all')
    plt.rcParams.update({
      "figure.max_open_warning": 0,
      "text.usetex": True,
      "figure.autolayout": True,
      "font.family": "STIXGeneral",
      "mathtext.fontset": "stix",
      "font.size":        10,
      "xtick.labelsize":  10,
      "ytick.labelsize":  10,
      "lines.linewidth":  1,
      "lines.markersize": 5
    })
    plt.rcParams['text.latex.preamble']= r"\usepackage{amsmath}"

    matplotlib.use('Agg')
    
    cmap           = matplotlib.colormaps['gist_rainbow']
    alpha          = 0.75

    hls            = np.array([colorsys.rgb_to_hls(*c) for c in cmap(np.arange(cmap.N))[:,:3]])
    hls[:,1]      *= alpha
    rgb            = np.clip(np.array([colorsys.hls_to_rgb(*c) for c in hls]), 0,1)
    cmap           = colors.LinearSegmentedColormap.from_list("", rgb)

    figFormat      = ".png"
    figDPI         = 200
    maxLineWidth   = 5
    
    baseName       = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile       = Dir + 'data_' + baseName
    rigFile        = Dir + 'rig_'  + baseName
    intFile        = Dir + 'int_'  + baseName
    parFile        = Dir + 'par_'  + baseName
    frictPartsFile = Dir + 'frictPartsIDs.txt'
    
    # this function requires myRigidClusters to be previously run
    # let's check if it has been run, let's do it if not
    if not os.path.exists(rigFile):
        myRigidClusters(Dir)
        
    # this function requires frict_clusters to be previously run
    # let's check if it has been run, let's do it if not
    if not os.path.exists(frictPartsFile):
        frict_parts_IDs(Dir)
    
    print("   >> Reading data")
    
    t,     gamma, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
        = np.loadtxt(dataFile, skiprows=37).transpose()
        
    NP  = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    phi =     np.genfromtxt(dataFile, skip_header=2, max_rows=1, comments='!')[2]
    
    Lx  = np.genfromtxt(parFile, comments=None, skip_header=3, max_rows=1)[2]
    Lz  = np.genfromtxt(parFile, comments=None, skip_header=5, max_rows=1)[2]
    
    dummy, a, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy = np.loadtxt(parFile, skiprows=23, max_rows=NP).transpose()
    
    rx               = []
    rz               = []
    ip               = []
    jp               = []
    nx               = []
    nz               = []
    xi               = []
    normLub          = []
    tanLubX          = []
    tanLubZ          = []
    contState        = []
    normCont         = []
    tanContX         = []
    tanContZ         = []
    normRep          = []
    clustersSizes    = []
    numberOfClusters = []
    rigPartsIDs      = []
    normInts         = []
    numInts          = []
    maxForces        = []
    frictPartsIDs    = []
    
    int_skiprows       = 20
    numInteractions    = 0
    morelines          = 0
    
    for it in range(len(t)):
        
        int_skiprows    += 7 + numInteractions
        numInteractions  = 0
        with open(intFile, 'r') as file:
            for i in range(int_skiprows):
                line = file.readline()
            while True:
                line = file.readline()
                if not line or line.split()[0] == '#':
                    break
                else:
                    numInteractions += 1
    
        print("    - time step " + str(it+1) + " out of " + str(len(t)))
        
        numInts.append(numInteractions)
        
        dummy, dummy, rx_it, rz_it, dummy, dummy, dummy, dummy, dummy, dummy, dummy = np.loadtxt(parFile, skiprows=16+it*NP+(it+1)*7, max_rows=NP).transpose()
        
        ip_it, jp_it, nx_it, dummy, nz_it, xi_it, normLub_it, tanLubX_it, dummy, tanLubZ_it, contState_it, normCont_it, tanContX_it, dummy, tanContZ_it, dummy, normRep_it = np.loadtxt(intFile, skiprows=int_skiprows, max_rows=numInteractions).transpose()
        
        normInts_it = np.abs(normLub_it + normCont_it + normRep_it + np.linalg.norm(np.array([tanLubX_it,tanLubZ_it]),axis=0) + np.linalg.norm(np.array([tanContX_it,tanContZ_it]),axis=0))
        
        clustersSizes_it = [int(x) for x in np.frombuffer(np.loadtxt(rigFile, skiprows=it+1, max_rows=1))]
        
        if clustersSizes_it == [0]:
            morelines          += 1
            numberOfClusters_it = 0
        else:
            numberOfClusters_it = len(clustersSizes_it)
        
        rigPartsIDs_it = []
        for i in range(numberOfClusters_it):
            rigPartsIDs_it.append(np.unique([int(x) for x in np.loadtxt(rigFile, skiprows=(len(t)+2)*2+2+it+i+int(np.sum(numberOfClusters))+morelines, max_rows=1, delimiter=',')]))
            if (len(rigPartsIDs_it[i]) != clustersSizes_it[i]):
                sys.exit("ERROR: there is a problem with the clusters sizes")
        
        frictPartsIDs_it = np.genfromtxt(frictPartsFile, skip_header=it, max_rows=1)
        
        if np.any(np.isnan(frictPartsIDs_it)):
            frictPartsIDs_it = []
        else:
            if frictPartsIDs_it.size == 1:
                frictPartsIDs_it = [int(frictPartsIDs_it)]
            else:
                frictPartsIDs_it = list([int(x) for x in frictPartsIDs_it])
            
        rx.append(rx_it)
        rz.append(rz_it)
        ip.append([int(x) for x in ip_it])
        jp.append([int(x) for x in jp_it])
        nx.append(nx_it)
        nz.append(nz_it)
        xi.append(xi_it)
        normLub.append(normLub_it)
        tanLubX.append(tanLubX_it)
        tanLubZ.append(tanLubZ_it)
        contState.append(contState_it)
        normCont.append(normCont_it)
        tanContX.append(tanContX_it)
        tanContZ.append(tanContZ_it)
        normRep.append(normRep_it)
        clustersSizes.append(clustersSizes_it)
        numberOfClusters.append(numberOfClusters_it)
        rigPartsIDs.append(rigPartsIDs_it)
        normInts.append(normInts_it)
        maxForces.append(np.max(normInts_it))
        frictPartsIDs.append(frictPartsIDs_it)
    
    maxForce = np.max(maxForces)
    
    print("   >> Generating snapshots")
    
    a2    = np.max(a)
    newLx = Lx + 2*a2
    newLz = Lz + 2*a2
    
    rangeSnapshots     = np.linspace(0, len(t)-1, len(t),             dtype=int)
    rangeSomeSnapshots = np.linspace(0, len(t)-1, int((len(t))/50)+1, dtype=int)
    
    if os.path.exists(Dir+"snapshots"):
        shutil.rmtree(Dir+"snapshots")
    if os.path.exists(Dir+"some_snapshots"):
        shutil.rmtree(Dir+"some_snapshots")
        
    os.mkdir(Dir+"snapshots")
    os.mkdir(Dir+"snapshots/clusters")
    os.mkdir(Dir+"snapshots/interactions")
    os.mkdir(Dir+"snapshots/frictParts")
    os.mkdir(Dir+"snapshots/superposition")
    
    os.mkdir(Dir+"some_snapshots")
    os.mkdir(Dir+"some_snapshots/clusters")
    os.mkdir(Dir+"some_snapshots/interactions")
    os.mkdir(Dir+"some_snapshots/frictParts")
    os.mkdir(Dir+"some_snapshots/superposition")
    
    fig1, ax1 = plt.subplots(1,1, figsize=(5,5))
    fig2, ax2 = plt.subplots(1,1, figsize=(5,5))
    fig3, ax3 = plt.subplots(1,1, figsize=(5,5))
    fig4, ax4 = plt.subplots(1,1, figsize=(5,5))
    fig5, ax5 = plt.subplots(1,1, figsize=(5,5))
    
    for ss in rangeSnapshots:
        
        print("    - time step " + str(ss+1) + " out of " + str(len(t)))
        
        title = r"$NP =\ $" + str(NP) + r"$\quad \phi =\ $" + str(phi) + r"$\quad t^{*} =\ $" + '{:.1f}'.format(t[ss]) + r"$\quad \gamma =\ $" + '{:.2f}'.format(gamma[ss])
        
        if len(rigPartsIDs[ss]) > 0:
            RigClustersPartsIDs = np.concatenate(rigPartsIDs[ss])
        else:
            RigClustersPartsIDs = []
            
        allPartsIDs                = np.array(range(0,NP))
        NoRigClustersPartsIDs      = allPartsIDs[np.isin(allPartsIDs,RigClustersPartsIDs)==False]
        NoFrictPartsIDs            = allPartsIDs[np.isin(allPartsIDs,frictPartsIDs[ss])==False]
        NeitherFrictNorRigPartsIDs = allPartsIDs[np.logical_and(np.isin(allPartsIDs,RigClustersPartsIDs)==False, np.isin(allPartsIDs,frictPartsIDs[ss])==False)]
        BothFrictAndRigPartsIDs    = allPartsIDs[np.logical_and(np.isin(allPartsIDs,RigClustersPartsIDs)==True, np.isin(allPartsIDs,frictPartsIDs[ss])==True)]
        lineWidths                 = maxLineWidth * normInts[ss] / maxForce
        colorInts                  = np.array(['r'] * numInts[ss], dtype=object)
        contactLess                = np.where(contState[ss]==0)[0]
        frictionLess               = np.where(contState[ss]==1)[0]
        if contactLess.size        > 0: colorInts[contactLess]  = 'tab:cyan'
        if frictionLess.size       > 0: colorInts[frictionLess] = 'g'
        
        # plot clusters
        ax1.clear()
        ax1.set_title(title)
        for i in NoRigClustersPartsIDs:
            circle = plt.Circle((rx[ss][i],rz[ss][i]), a[i], color='#323232', fill='#323232')
            ax1.add_artist(circle)
        for i in RigClustersPartsIDs:
            circle = plt.Circle((rx[ss][i],rz[ss][i]), a[i], color='#A00000', fill='#A00000')
            ax1.add_artist(circle)
        ax1.set_xlim([-(newLx/2+0.2),(newLx/2+0.2)])
        ax1.set_ylim([-(newLz/2+0.2),(newLz/2+0.2)])
        ax1.axis('off')
        ax1.set_aspect('equal')
        fig1.savefig(Dir+"snapshots/clusters/"+str(ss+1)+figFormat, dpi=figDPI)
        if np.isin(ss, rangeSomeSnapshots):
            ax3.clear()
            ax3.set_title(title)
            for i in NoRigClustersPartsIDs:
                circle = plt.Circle((rx[ss][i],rz[ss][i]), a[i], color='#787878', fill='#787878')
                ax3.add_artist(circle)
            for i in range(numberOfClusters[ss]):
                Color = cmap((numberOfClusters[ss]-i)/numberOfClusters[ss])
                for j in range(clustersSizes[ss][i]):
                    ipCC   = rigPartsIDs[ss][i][j]
                    circle = plt.Circle((rx[ss][ipCC], rz[ss][ipCC]), a[ipCC], color=Color, fill=Color)
                    ax3.add_artist(circle)
            ax3.set_xlim([-(newLx/2+0.2),(newLx/2+0.2)])
            ax3.set_ylim([-(newLz/2+0.2),(newLz/2+0.2)])
            ax3.axis('off')
            ax3.set_aspect('equal')
            fig3.savefig(Dir+"some_snapshots/clusters/"+str(ss+1)+figFormat, dpi=figDPI)
        
        # plot interactions
        ax2.clear()    
        ax2.set_title(title)
        for i in allPartsIDs:
            circle = plt.Circle((rx[ss][i],rz[ss][i]), a[i], color='#323232', fill='#323232')
            ax2.add_artist(circle)
        for i in range(numInts[ss]):
            ipInt = ip[ss][i]
            jpInt = jp[ss][i]
            nij   = np.array([nx[ss][i], nz[ss][i]])
            rij   = nij * (xi[ss][i]+2.) * (a[ipInt]+a[jpInt]) * 0.5
            p1    = np.array([rx[ss][ipInt], rz[ss][ipInt]])
            p2    = p1 + rij
            ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], color=colorInts[i], linewidth=lineWidths[i])
            if (np.sign(nij[0]) != np.sign(rx[ss][jpInt]-rx[ss][ipInt]) or
                np.sign(nij[1]) != np.sign(rz[ss][jpInt]-rz[ss][ipInt])):   # periodicity
                p3 = np.array([rx[ss][jpInt], rz[ss][jpInt]])
                p4 = p3 - rij
                ax2.plot([p3[0], p4[0]], [p3[1], p4[1]], color=colorInts[i], linewidth=lineWidths[i])
        ax2.set_xlim([-(newLx/2+0.2),(newLx/2+0.2)])
        ax2.set_ylim([-(newLz/2+0.2),(newLz/2+0.2)])
        ax2.axis('off')
        ax2.set_aspect('equal')
        fig2.savefig(Dir+"snapshots/interactions/"+str(ss+1)+figFormat, dpi=figDPI)
        if np.isin(ss, rangeSomeSnapshots):
            fig2.savefig(Dir+"some_snapshots/interactions/"+str(ss+1)+figFormat, dpi=figDPI)
        
        # plot frictional clusters
        ax4.clear()
        ax4.set_title(title)
        for i in NoFrictPartsIDs:
            circle = plt.Circle((rx[ss][i],rz[ss][i]), a[i], color='#323232', fill='#323232')
            ax4.add_artist(circle)
        for i in frictPartsIDs[ss]:
            circle = plt.Circle((rx[ss][i],rz[ss][i]), a[i], color='#A00000', fill='#A00000')
            ax4.add_artist(circle)
        ax4.set_xlim([-(newLx/2+0.2),(newLx/2+0.2)])
        ax4.set_ylim([-(newLz/2+0.2),(newLz/2+0.2)])
        ax4.axis('off')
        ax4.set_aspect('equal')
        fig4.savefig(Dir+"snapshots/frictParts/"+str(ss+1)+figFormat, dpi=figDPI)
        if np.isin(ss, rangeSomeSnapshots):
            fig4.savefig(Dir+"some_snapshots/frictParts/"+str(ss+1)+figFormat, dpi=figDPI)
            
        # plot superposition of rigid and frictional clusters
        ax5.clear()
        ax5.set_title(title)
        for i in NeitherFrictNorRigPartsIDs:
            circle = plt.Circle((rx[ss][i],rz[ss][i]), a[i], color='#323232', fill='#323232')
            ax5.add_artist(circle)
        for i in frictPartsIDs[ss]:
            circle = plt.Circle((rx[ss][i],rz[ss][i]), a[i], color='y', fill='y')
            ax5.add_artist(circle)
        for i in RigClustersPartsIDs:
            circle = plt.Circle((rx[ss][i],rz[ss][i]), a[i], color='b', fill='b')
            ax5.add_artist(circle)
        for i in BothFrictAndRigPartsIDs:
            circle = plt.Circle((rx[ss][i],rz[ss][i]), a[i], color='g', fill='g')
            ax5.add_artist(circle)
        ax5.set_xlim([-(newLx/2+0.2),(newLx/2+0.2)])
        ax5.set_ylim([-(newLz/2+0.2),(newLz/2+0.2)])
        ax5.axis('off')
        ax5.set_aspect('equal')
        fig5.savefig(Dir+"snapshots/superposition/"+str(ss+1)+figFormat, dpi=figDPI)
        if np.isin(ss, rangeSomeSnapshots):
            fig5.savefig(Dir+"some_snapshots/superposition/"+str(ss+1)+figFormat, dpi=figDPI)
    
    plt.close('all')
    # matplotlib.use('QtAgg')