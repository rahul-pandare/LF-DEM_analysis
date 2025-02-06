import os
import sys
import glob
import steadyStateTime
import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy             as np



plt.close('all')
plt.rcParams.update({
  "figure.max_open_warning": 0,
  "text.usetex": True,
  "figure.autolayout": True,
  "font.family": "STIXGeneral",
  "mathtext.fontset": "stix",
  "font.size":        8,
  "xtick.labelsize":  8,
  "ytick.labelsize":  8,
  "lines.linewidth":  1,
  "lines.markersize": 5
})
plt.rcParams['text.latex.preamble']= r"\usepackage{amsmath}"

#mpl.use('QtAgg')

figFormat = ".pdf"
myCmap    = mpl.colormaps['viridis_r']
FigSize   = (13,13*9/16)




indSig = 1

sigmas = np.array([20, 100])

sigma  = sigmas[indSig]

TopDir = "../2D_sigma" + str(sigma) + "/"

NP     = [100, 250, 500, 750, 1000, 2000, 4000, 8000]

phi    = [[0.705, 0.708, 0.710, 0.712, 0.715, 0.718, 0.720, 0.722,
           0.725, 0.728, 0.730, 0.732, 0.735, 0.738, 0.740, 0.742,
           0.745, 0.748, 0.750, 0.751, 0.752, 0.753, 0.754, 0.755,
           0.756, 0.757, 0.758, 0.759, 0.760, 0.761, 0.762, 0.763,
           0.764, 0.765, 0.766, 0.767, 0.768, 0.769, 0.770, 0.772, 0.775],
          [0.705, 0.708, 0.710, 0.712, 0.715, 0.718, 0.720, 0.722,
           0.725, 0.728, 0.730, 0.732, 0.735, 0.738, 0.740, 0.742,
           0.745, 0.748, 0.750, 0.751, 0.752, 0.753, 0.754, 0.755,
           0.756, 0.757, 0.758, 0.759, 0.760, 0.762, 0.765]]



index_jam        = -np.ones(len(NP), dtype=int)
etaS_mean        = np.zeros((len(NP),len(phi[indSig])))
etaS_std         = np.zeros((len(NP),len(phi[indSig])))
f_rig_mean       = np.zeros((len(NP),len(phi[indSig])))
f_rig_std        = np.zeros((len(NP),len(phi[indSig])))
F_rig_mean       = np.zeros((len(NP),len(phi[indSig])))
F_rig_var        = np.zeros((len(NP),len(phi[indSig])))
f_prime_rig_mean = np.zeros((len(NP),len(phi[indSig])))
f_prime_rig_std  = np.zeros((len(NP),len(phi[indSig])))
F_prime_rig_mean = np.zeros((len(NP),len(phi[indSig])))
F_prime_rig_var  = np.zeros((len(NP),len(phi[indSig])))
Z_mean           = np.zeros((len(NP),len(phi[indSig])))
Z_std            = np.zeros((len(NP),len(phi[indSig])))
Znet_mean        = np.zeros((len(NP),len(phi[indSig])))
Znet_std         = np.zeros((len(NP),len(phi[indSig])))



etaSDir = TopDir + "00_etaS_vs_time"
frigDir = TopDir + "00_frig_vs_time"
histDir = TopDir + "00_histograms"

if not os.path.exists(etaSDir):
    os.mkdir(etaSDir)
if not os.path.exists(frigDir):
    os.mkdir(frigDir)
if not os.path.exists(histDir):
    os.mkdir(histDir)



fbinwidth = 0.025
fbins     = np.arange(0,1+fbinwidth,fbinwidth).tolist()

if indSig == 0:
    
    fig1,  axs1  = plt.subplots(len(NP)-2, 7, figsize=FigSize)
    fig1.suptitle(r"Histograms for $\eta^{S}$",  fontsize=8)
    
    fig2,  axs2  = plt.subplots(len(NP)-2, 7, sharex=True, sharey='row', figsize=FigSize)
    fig2.suptitle(r"Histograms for $f_{rig}$",   fontsize=8)
    
    fig3,  axs3  = plt.subplots(len(NP)-2, 7, figsize=FigSize)
    fig3.suptitle(r"Histograms for $\eta^{S}$",  fontsize=8)
    
    fig4,  axs4  = plt.subplots(len(NP)-2, 7, sharex=True, sharey='row', figsize=FigSize)
    fig4.suptitle(r"Histograms for $f_{rig}$",   fontsize=8)
    
    fig5,  axs5  = plt.subplots(len(NP)-2, 7, figsize=FigSize)
    fig5.suptitle(r"Histograms for $\eta^{S}$",  fontsize=8)
    
    fig6,  axs6  = plt.subplots(len(NP)-2, 7, sharex=True, sharey='row', figsize=FigSize)
    fig6.suptitle(r"Histograms for $f_{rig}$",   fontsize=8)
    
    fig7,  axs7  = plt.subplots(len(NP)-2, 7, figsize=FigSize)
    fig7.suptitle(r"Histograms for $\eta^{S}$",  fontsize=8)
    
    fig8,  axs8  = plt.subplots(len(NP)-2, 7, sharex=True, sharey='row', figsize=FigSize)
    fig8.suptitle(r"Histograms for $f_{rig}$",   fontsize=8)
    
    fig9,  axs9  = plt.subplots(len(NP)-2, 7, figsize=FigSize)
    fig9.suptitle(r"Histograms for $\eta^{S}$",  fontsize=8)
    
    fig10, axs10 = plt.subplots(len(NP)-2, 7, sharex=True, sharey='row', figsize=FigSize)
    fig10.suptitle(r"Histograms for $f_{rig}$",  fontsize=8)
    
    fig11, axs11 = plt.subplots(len(NP)-2, 6, figsize=FigSize)
    fig11.suptitle(r"Histograms for $\eta^{S}$", fontsize=8)
    
    fig12, axs12 = plt.subplots(len(NP)-2, 6, sharex=True, sharey='row', figsize=FigSize)
    fig12.suptitle(r"Histograms for $f_{rig}$",  fontsize=8)
    
elif indSig == 1:

    fig1, axs1 = plt.subplots(len(NP)-2, 8, figsize=FigSize)
    fig1.suptitle(r"Histograms for $\eta^{S}$", fontsize=8)
    
    fig2, axs2 = plt.subplots(len(NP)-2, 8, sharex=True, sharey='row', figsize=FigSize)
    fig2.suptitle(r"Histograms for $f_{rig}$",  fontsize=8)
    
    fig3, axs3 = plt.subplots(len(NP)-2, 8, figsize=FigSize)
    fig3.suptitle(r"Histograms for $\eta^{S}$", fontsize=8)
    
    fig4, axs4 = plt.subplots(len(NP)-2, 8, sharex=True, sharey='row', figsize=FigSize)
    fig4.suptitle(r"Histograms for $f_{rig}$",  fontsize=8)
    
    fig5, axs5 = plt.subplots(len(NP)-2, 8, figsize=FigSize)
    fig5.suptitle(r"Histograms for $\eta^{S}$", fontsize=8)
    
    fig6, axs6 = plt.subplots(len(NP)-2, 8, sharex=True, sharey='row', figsize=FigSize)
    fig6.suptitle(r"Histograms for $f_{rig}$",  fontsize=8)
    
    fig7, axs7 = plt.subplots(len(NP)-2, 7, figsize=FigSize)
    fig7.suptitle(r"Histograms for $\eta^{S}$", fontsize=8)
    
    fig8, axs8 = plt.subplots(len(NP)-2, 7, sharex=True, sharey='row', figsize=FigSize)
    fig8.suptitle(r"Histograms for $f_{rig}$",  fontsize=8)
    


for i in range(len(NP)):
    print('')
    print("- CHECKING FILES FOR NP  = " + str(NP[i]))
    for j in range(len(phi[indSig])):
        Dir        = TopDir + "NP" + str(NP[i]) + "/phi0." + str(int(phi[indSig][j]*1000)) + "/"
        baseName   = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
        dataFile   = Dir + 'data_' + baseName
        FrigFile   = Dir + 'F_rig.txt'
        FprimeRig  = Dir + 'F_prime_rig.txt'
        ZnetFile   = Dir + 'Z_Znet.txt'
        if not os.path.exists(dataFile)  or not os.path.exists(FrigFile) or \
           not os.path.exists(FprimeRig) or not os.path.exists(ZnetFile):
               sys.exit("- phi = " + str(phi[indSig][j]) + "  >>  MISSING SOME FILES  >>  STOP")
        else:
            print("- phi = " + str(phi[indSig][j]) + "  >>  OK")



data = open(TopDir+"data.txt", "w")
    
for i in range(len(NP)):
        
    print('')
    print("- NP  = " + str(NP[i]))
    
    m = -1
    n = -1
    o = -1
    p = -1
    q = -1
    r = -1
    
    if indSig == 0:
        
        figf, axsf = plt.subplots(5,8, sharex=True, sharey=True, figsize=FigSize)
        figf.suptitle("NP = "+str(NP[i]))
        
        fige, axse = plt.subplots(5,8, sharex=True, sharey=True, figsize=FigSize)
        fige.suptitle("NP = "+str(NP[i]))
        
    elif indSig == 1:
    
        figf, axsf = plt.subplots(5,6, sharex=True, sharey=True, figsize=FigSize)
        figf.suptitle("NP = "+str(NP[i]))
        
        fige, axse = plt.subplots(5,6, sharex=True, sharey=True, figsize=FigSize)
        fige.suptitle("NP = "+str(NP[i]))
        
    figP, axsP = plt.subplots(1,2, width_ratios=[20,1])
    figP.suptitle("NP = "+str(NP[i]))
    axsP[0].set_xlabel(r'$n$')
    axsP[0].set_ylabel(r'$P\left(n\right)$')
    cb = mpl.colorbar.ColorbarBase(axsP[1], orientation='vertical', cmap=myCmap, label=r'$\phi$', norm=mpl.colors.Normalize(np.min(phi[indSig]),np.max(phi[indSig])))
        
    data.write('NP=' + str(NP[i]) + '\n')
    data.write("phi        <etaS>             std(etaS)          <frig>           std(frig)        <Frig>             var(Frig)          <f'rig>           std(f'rig)        <F'rig>             var(F'rig)        <Z>              std(Z)           <Znet>           std(Znet)" + '\n')
        
            
    for j in range(len(phi[indSig])):
        
        myColor = myCmap((j+1)/len(phi[indSig]))
        
        Dir           = TopDir + "NP" + str(NP[i]) + "/phi0." + str(int(phi[indSig][j]*1000)) + "/"
        baseName      = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
        dataFile      = Dir + 'data_' + baseName
        rigFile       = Dir + 'rig_'  + baseName
        FrigFile      = Dir + 'F_rig.txt'
        FprimeRigFile = Dir + 'F_prime_rig.txt'
        ZnetFile      = Dir + 'Z_Znet.txt'
        
        t,     gamma, dummy, etaS,  dummy, dummy, dummy, dummy, dummy, dummy, \
        dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
        dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
            = np.loadtxt(dataFile, skiprows=37).transpose()
            
        SSi = np.where(t==steadyStateTime.t_SS(indSig,i,j))[0][0]
        # SSi = np.where(np.abs(gamma-2)==np.min(np.abs(gamma-2)))[0][0]
            
        Z, dummy, Znet, dummy = np.loadtxt(ZnetFile, skiprows=1).transpose()
        
        F_rig = np.loadtxt(FrigFile)
        f_rig = F_rig / NP[i]
        
        dummy, F_prime_rig = np.loadtxt(FprimeRigFile, skiprows=1).transpose()
        f_prime_rig = F_prime_rig / NP[i]
        
        clusterSizes = [] * len(t)
        for it in range(len(t)):
            clusterSizes.append([int(x) for x in np.frombuffer(np.loadtxt(rigFile, skiprows=1+it, max_rows=1))])
        clusterSizes = [item for sublist in clusterSizes for item in sublist]
        clusterSizes = [i for i in clusterSizes if i != 0]
        
        cedges = np.arange(np.min(clusterSizes)-0.5, np.max(clusterSizes)+1.5)
        cbins  = np.arange(np.min(clusterSizes),     np.max(clusterSizes)+1)
        Pn, dummy = np.histogram(clusterSizes[SSi:], cedges, density=True)
        axsP[0].loglog(cbins, Pn, marker='.', linestyle='None', color=myColor)
        
        if not np.all(np.diff(cbins)==1):
            sys.exit('- Something is wrong with the bins for P(n)  >  STOP !')
        if not np.all(np.diff(cedges)==1):
            sys.exit('- Something is wrong with the bin edges for P(n)  >  STOP !')
        if not np.abs(np.sum(Pn)-1)<1e-9:
            sys.exit('- Something is wrong with P(n) (sum(P) = ' + str(np.sum(Pn)) + ')  >  STOP !')
        
        
        
        etaS_mean[i][j]        = np.mean(etaS[SSi:])
        etaS_std[i][j]         = np.std(etaS[SSi:])
        f_rig_mean[i][j]       = np.mean(f_rig[SSi:])
        f_rig_std[i][j]        = np.std(f_rig[SSi:])
        F_rig_mean[i][j]       = np.mean(F_rig[SSi:])
        F_rig_var[i][j]        = np.var(F_rig[SSi:])
        f_prime_rig_mean[i][j] = np.mean(f_prime_rig[SSi:])
        f_prime_rig_std[i][j]  = np.std(f_prime_rig[SSi:])
        F_prime_rig_mean[i][j] = np.mean(F_prime_rig[SSi:])
        F_prime_rig_var[i][j]  = np.var(F_prime_rig[SSi:])
        Z_mean[i][j]           = np.mean(Z[SSi:])
        Z_std[i][j]            = np.std(Z[SSi:])
        Znet_mean[i][j]        = np.mean(Znet[SSi:])
        Znet_std[i][j]         = np.std(Znet[SSi:])
        
        data.write('{:.3f}'.format(phi[indSig][j])         + '      ' + 
                   '{:.9f}'.format(etaS_mean[i][j])        + '      ' + 
                   '{:.9f}'.format(etaS_std[i][j])         + '      ' + 
                   '{:.9f}'.format(f_rig_mean[i][j])       + '      ' + 
                   '{:.9f}'.format(f_rig_std[i][j])        + '      ' + 
                   '{:.9f}'.format(F_rig_mean[i][j])       + '      ' + 
                   '{:.9f}'.format(F_rig_var[i][j])        + '      ' +
                   '{:.9f}'.format(f_prime_rig_mean[i][j]) + '      ' + 
                   '{:.9f}'.format(f_prime_rig_std[i][j])  + '      ' + 
                   '{:.9f}'.format(F_prime_rig_mean[i][j]) + '      ' + 
                   '{:.9f}'.format(F_prime_rig_var[i][j])  + '      ' +
                   '{:.9f}'.format(Z_mean[i][j])           + '      ' +
                   '{:.9f}'.format(Z_std[i][j])            + '      ' +
                   '{:.9f}'.format(Znet_mean[i][j])        + '      ' +
                   '{:.9f}'.format(Znet_std[i][j])         + '      ' + '\n')
        
        
        
        print("- phi = 0." + str(int(phi[indSig][j]*1000)) + "  >  etaS = " + '{:.1f}'.format(etaS_mean[i][j]) + ",  total strain = " + '{:.1f}'.format(gamma[-1]))
        
        if np.max(np.abs(etaS)) > 10000 and index_jam[i] == -1:
            index_jam[i] = j
          
        if j > 0:
            row = int((j-1)/len(axsf[0]))
            col = (j-1) % len(axsf[0])
            axsf[row,col].set_title(r"$\phi = \ $"+str(phi[indSig][j]), fontsize=8)
            axse[row,col].set_title(r"$\phi = \ $"+str(phi[indSig][j]), fontsize=8)
            if row == len(axsf)-1:
                axsf[row,col].set_xlabel(r'$t$')
                axse[row,col].set_xlabel(r'$t$')
            if col == 0:
                axsf[row,col].set_ylabel(r'$f_{rig}$')
                axse[row,col].set_ylabel(r'$\eta^{S}$')
            axsf[row,col].plot(t, f_rig, 'k')
            axsf[row,col].axvline(x=t[SSi], linestyle='--', color='r')
            axsf[row,col].grid(which='Both', alpha=0.2)
            if index_jam[i] == -1:
                axse[row,col].plot(t, etaS, 'k')
                axse[row,col].axvline(x=t[SSi], linestyle='--', color='r')
                axse[row,col].grid(which='Both', alpha=0.2)
                
        if (i >= 2):
            if indSig == 0:
                if phi[indSig][j] >= 0.705 and phi[indSig][j] < 0.722:
                    m += 1
                    if index_jam[i] == -1:
                        axs1[i-2][m].hist(etaS[SSi:],    bins='auto', stacked=True)
                        axs2[i-2][m].hist(f_rig[SSi:],   bins=fbins,  stacked=True, density=True)
                        axs1[i-2][m].set_title(r"$NP = \ $" +str(NP[i])+r"$,\ \phi = \ $"+str(phi[indSig][j]), fontsize=8)
                        axs2[i-2][m].set_title(r"$NP = \ $" +str(NP[i])+r"$,\ \phi = \ $"+str(phi[indSig][j]), fontsize=8)
                elif phi[indSig][j] >= 0.722 and phi[indSig][j] < 0.740:
                    n += 1
                    if index_jam[i] == -1:
                        axs3[i-2][n].hist(etaS[SSi:],    bins='auto', stacked=True)
                        axs4[i-2][n].hist(f_rig[SSi:],   bins=fbins,  stacked=True, density=True)
                        axs3[i-2][n].set_title(r"$NP = \ $" +str(NP[i])+r"$,\ \phi = \ $"+str(phi[indSig][j]), fontsize=8)
                        axs4[i-2][n].set_title(r"$NP = \ $" +str(NP[i])+r"$,\ \phi = \ $"+str(phi[indSig][j]), fontsize=8)
                elif phi[indSig][j] >= 0.740 and phi[indSig][j] < 0.753:
                    o += 1
                    if index_jam[i] == -1:
                        axs5[i-2][o].hist(etaS[SSi:],    bins='auto', stacked=True)
                        axs6[i-2][o].hist(f_rig[SSi:],   bins=fbins,  stacked=True, density=True)
                        axs5[i-2][o].set_title(r"$NP = \ $" +str(NP[i])+r"$,\ \phi = \ $"+str(phi[indSig][j]), fontsize=8)
                        axs6[i-2][o].set_title(r"$NP = \ $" +str(NP[i])+r"$,\ \phi = \ $"+str(phi[indSig][j]), fontsize=8)
                elif phi[indSig][j] >= 0.753 and phi[indSig][j] < 0.760:
                    p += 1
                    if index_jam[i] == -1:
                        axs7[i-2][p].hist(etaS[SSi:],   bins='auto', stacked=True)
                        axs8[i-2][p].hist(f_rig[SSi:],  bins=fbins,  stacked=True, density=True)
                        axs7[i-2][p].set_title(r"$NP = \ $" +str(NP[i])+r"$,\ \phi = \ $"+str(phi[indSig][j]), fontsize=8)
                        axs8[i-2][p].set_title(r"$NP = \ $" +str(NP[i])+r"$,\ \phi = \ $"+str(phi[indSig][j]), fontsize=8)
                elif phi[indSig][j] >= 0.760 and phi[indSig][j] < 0.767:
                    q += 1
                    if index_jam[i] == -1:
                        axs9[i-2][q].hist(etaS[SSi:],   bins='auto', stacked=True)
                        axs10[i-2][q].hist(f_rig[SSi:], bins=fbins,  stacked=True, density=True)
                        axs9[i-2][q].set_title(r"$NP = \ $" +str(NP[i])+r"$,\ \phi = \ $"+str(phi[indSig][j]), fontsize=8)
                        axs10[i-2][q].set_title(r"$NP = \ $" +str(NP[i])+r"$,\ \phi = \ $"+str(phi[indSig][j]), fontsize=8)
                elif phi[indSig][j] >= 0.767:
                    r += 1
                    if index_jam[i] == -1:
                        axs11[i-2][r].hist(etaS[SSi:],  bins='auto', stacked=True)
                        axs12[i-2][r].hist(f_rig[SSi:], bins=fbins,  stacked=True, density=True)
                        axs11[i-2][r].set_title(r"$NP = \ $" +str(NP[i])+r"$,\ \phi = \ $"+str(phi[indSig][j]), fontsize=8)
                        axs12[i-2][r].set_title(r"$NP = \ $" +str(NP[i])+r"$,\ \phi = \ $"+str(phi[indSig][j]), fontsize=8)
            elif indSig == 1:
                if phi[indSig][j] >= 0.705 and phi[indSig][j] < 0.725:
                    m += 1
                    if index_jam[i] == -1:
                        axs1[i-2][m].hist(etaS[SSi:],   bins='auto', stacked=True)
                        axs2[i-2][m].hist(f_rig[SSi:],  bins=fbins,  stacked=True, density=True)
                        axs1[i-2][m].set_title(r"$NP = \ $" +str(NP[i])+r"$,\ \phi = \ $"+str(phi[indSig][j]), fontsize=8)
                        axs2[i-2][m].set_title(r"$NP = \ $" +str(NP[i])+r"$,\ \phi = \ $"+str(phi[indSig][j]), fontsize=8)
                elif phi[indSig][j] >= 0.725 and phi[indSig][j] < 0.745:
                    n += 1
                    if index_jam[i] == -1:
                        axs3[i-2][n].hist(etaS[SSi:],   bins='auto', stacked=True)
                        axs4[i-2][n].hist(f_rig[SSi:],  bins=fbins,  stacked=True, density=True)
                        axs3[i-2][n].set_title(r"$NP = \ $" +str(NP[i])+r"$,\ \phi = \ $"+str(phi[indSig][j]), fontsize=8)
                        axs4[i-2][n].set_title(r"$NP = \ $" +str(NP[i])+r"$,\ \phi = \ $"+str(phi[indSig][j]), fontsize=8)
                elif phi[indSig][j] >= 0.745 and phi[indSig][j] < 0.756:
                    o += 1
                    if index_jam[i] == -1:
                        axs5[i-2][o].hist(etaS[SSi:],   bins='auto', stacked=True)
                        axs6[i-2][o].hist(f_rig[SSi:],  bins=fbins,  stacked=True, density=True)
                        axs5[i-2][o].set_title(r"$NP = \ $" +str(NP[i])+r"$,\ \phi = \ $"+str(phi[indSig][j]), fontsize=8)
                        axs6[i-2][o].set_title(r"$NP = \ $" +str(NP[i])+r"$,\ \phi = \ $"+str(phi[indSig][j]), fontsize=8)
                elif phi[indSig][j] >= 0.756:
                    p += 1
                    if index_jam[i] == -1:
                        axs7[i-2][p].hist(etaS[SSi:],   bins='auto', stacked=True)
                        axs8[i-2][p].hist(f_rig[SSi:],  bins=fbins,  stacked=True, density=True)
                        axs7[i-2][p].set_title(r"$NP = \ $" +str(NP[i])+r"$,\ \phi = \ $"+str(phi[indSig][j]), fontsize=8)
                        axs8[i-2][p].set_title(r"$NP = \ $" +str(NP[i])+r"$,\ \phi = \ $"+str(phi[indSig][j]), fontsize=8)
        
    data.write('\n')
    
    figf.savefig(frigDir+"/NP_"+str(NP[i])+figFormat, bbox_inches="tight")
    fige.savefig(etaSDir+"/NP_"+str(NP[i])+figFormat, bbox_inches="tight")
    
    axsP[0].grid(which='Both', alpha=0.2)
    figP.savefig(histDir+"/Pn__NP_"+str(NP[i])+figFormat, bbox_inches="tight")
    
data.close()

# save plots

fig1.savefig(histDir      +"/etaS_1"+figFormat, bbox_inches="tight")
fig3.savefig(histDir      +"/etaS_2"+figFormat, bbox_inches="tight")
fig5.savefig(histDir      +"/etaS_3"+figFormat, bbox_inches="tight")
fig7.savefig(histDir      +"/etaS_4"+figFormat, bbox_inches="tight")

fig2.savefig(histDir      +"/frig_1"+figFormat, bbox_inches="tight")
fig4.savefig(histDir      +"/frig_2"+figFormat, bbox_inches="tight")
fig6.savefig(histDir      +"/frig_3"+figFormat, bbox_inches="tight")
fig8.savefig(histDir      +"/frig_4"+figFormat, bbox_inches="tight")

if indSig == 0:
    
    fig9.savefig(histDir  +"/etaS_5"+figFormat, bbox_inches="tight")
    fig11.savefig(histDir +"/etaS_6"+figFormat, bbox_inches="tight")
    
    fig10.savefig(histDir +"/frig_5"+figFormat, bbox_inches="tight")
    fig12.savefig(histDir +"/frig_6"+figFormat, bbox_inches="tight")




