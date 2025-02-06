import os
import sys
import glob
import steadyStateTime
import numpy             as np
import pandas            as pd
import matplotlib        as mpl
import matplotlib.pyplot as plt
import scipy.optimize    as opt



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

#matplotlib.use('QtAgg')

figFormat = ".pdf"
myCmap    = mpl.colormaps['viridis_r']
FigSize   = (13,13*9/16)





#%% INITIALIZING



plt.close('all')

print('')
print('Reading data')



TopDir_20r  = "../2D_sigma20/"
TopDir_100r = "../2D_sigma100/"

sigmas      = np.array([20, 100])

etaS_maxFit = 5000

symbols     = np.array(['v', '^', '>', '<', 's', 'o', '*', 'p'])

NP          = [100, 250, 500, 750, 1000, 2000, 4000, 8000]

phi         = [[0.705, 0.708, 0.710, 0.712, 0.715, 0.718, 0.720, 0.722,
                0.725, 0.728, 0.730, 0.732, 0.735, 0.738, 0.740, 0.742,
                0.745, 0.748, 0.750, 0.751, 0.752, 0.753, 0.754, 0.755,
                0.756, 0.757, 0.758, 0.759, 0.760, 0.761, 0.762, 0.763,
                0.764, 0.765, 0.766, 0.767, 0.768, 0.769, 0.770, 0.772, 0.775],
               [0.705, 0.708, 0.710, 0.712, 0.715, 0.718, 0.720, 0.722,
                0.725, 0.728, 0.730, 0.732, 0.735, 0.738, 0.740, 0.742,
                0.745, 0.748, 0.750, 0.751, 0.752, 0.753, 0.754, 0.755,
                0.756, 0.757, 0.758, 0.759, 0.760, 0.762, 0.765]]

maxPhi = 0
for k in range(len(sigmas)):
    maxPhiK = np.max(phi[k])
    maxPhi  = np.max([maxPhi, maxPhiK])
    
minPhi = 1e6
for k in range(len(sigmas)):
    minPhiK = np.min(phi[k])
    minPhi  = np.min([minPhi, minPhiK])



data_20r = {}
for i in range(len(NP)):
    skip = 1
    if i > 0:
        skip = 2 + 2*i + (i-1)
        for j in range(i):
            skip += len(phi[0])
    data_20r[i] = pd.read_csv(TopDir_20r+"data.txt", delim_whitespace=True, skiprows=skip, nrows=len(phi[0]))

etaS_mean_20r        = []
etaS_std_20r         = []
f_rig_mean_20r       = []
f_rig_std_20r        = []
F_rig_mean_20r       = []
F_rig_var_20r        = []
f_prime_rig_mean_20r = []
f_prime_rig_std_20r  = []
F_prime_rig_mean_20r = []
F_prime_rig_var_20r  = []
Z_mean_20r           = []
Z_std_20r            = []
Znet_mean_20r        = []
Znet_std_20r         = []
for i in range(len(NP)):
    etaS_mean_20r.append(np.array(data_20r[i]['<etaS>']))
    etaS_std_20r.append(np.array(data_20r[i]['std(etaS)']))
    f_rig_mean_20r.append(np.array(data_20r[i]['<frig>']))
    f_rig_std_20r.append(np.array(data_20r[i]['std(frig)']))
    F_rig_mean_20r.append(np.array(data_20r[i]['<Frig>']))
    F_rig_var_20r.append(np.array(data_20r[i]['var(Frig)']))
    f_prime_rig_mean_20r.append(np.array(data_20r[i]["<f'rig>"]))
    f_prime_rig_std_20r.append(np.array(data_20r[i]["std(f'rig)"]))
    F_prime_rig_mean_20r.append(np.array(data_20r[i]["<F'rig>"]))
    F_prime_rig_var_20r.append(np.array(data_20r[i]["var(F'rig)"]))
    Z_mean_20r.append(np.array(data_20r[i]['<Z>']))
    Z_std_20r.append(np.array(data_20r[i]['std(Z)']))
    Znet_mean_20r.append(np.array(data_20r[i]['<Znet>']))
    Znet_std_20r.append(np.array(data_20r[i]['std(Znet)']))
    
del data_20r



data_100r = {}
for i in range(len(NP)):
    skip = 1
    if i > 0:
        skip = 2 + 2*i + (i-1)
        for j in range(i):
            skip += len(phi[1])
    data_100r[i] = pd.read_csv(TopDir_100r+"data.txt", delim_whitespace=True, skiprows=skip, nrows=len(phi[1]))

etaS_mean_100r        = []
etaS_std_100r         = []
f_rig_mean_100r       = []
f_rig_std_100r        = []
F_rig_mean_100r       = []
F_rig_var_100r        = []
f_prime_rig_mean_100r = []
f_prime_rig_std_100r  = []
F_prime_rig_mean_100r = []
F_prime_rig_var_100r  = []
Z_mean_100r           = []
Z_std_100r            = []
Znet_mean_100r        = []
Znet_std_100r         = []
for i in range(len(NP)):
    etaS_mean_100r.append(np.array(data_100r[i]['<etaS>']))
    etaS_std_100r.append(np.array(data_100r[i]['std(etaS)']))
    f_rig_mean_100r.append(np.array(data_100r[i]['<frig>']))
    f_rig_std_100r.append(np.array(data_100r[i]['std(frig)']))
    F_rig_mean_100r.append(np.array(data_100r[i]['<Frig>']))
    F_rig_var_100r.append(np.array(data_100r[i]['var(Frig)']))
    f_prime_rig_mean_100r.append(np.array(data_100r[i]["<f'rig>"]))
    f_prime_rig_std_100r.append(np.array(data_100r[i]["std(f'rig)"]))
    F_prime_rig_mean_100r.append(np.array(data_100r[i]["<F'rig>"]))
    F_prime_rig_var_100r.append(np.array(data_100r[i]["var(F'rig)"]))
    Z_mean_100r.append(np.array(data_100r[i]['<Z>']))
    Z_std_100r.append(np.array(data_100r[i]['std(Z)']))
    Znet_mean_100r.append(np.array(data_100r[i]['<Znet>']))
    Znet_std_100r.append(np.array(data_100r[i]['std(Znet)']))
    
del data_100r

TopDir = []
TopDir.append(TopDir_20r)
TopDir.append(TopDir_100r)

etaS_mean = []
etaS_mean.append(etaS_mean_20r)
etaS_mean.append(etaS_mean_100r)

f_rig_mean = []
f_rig_mean.append(f_rig_mean_20r)
f_rig_mean.append(f_rig_mean_100r)

F_rig_var = []
F_rig_var.append(F_rig_var_20r)
F_rig_var.append(F_rig_var_100r)

f_prime_rig_mean = []
f_prime_rig_mean.append(f_prime_rig_mean_20r)
f_prime_rig_mean.append(f_prime_rig_mean_100r)

F_prime_rig_var = []
F_prime_rig_var.append(F_prime_rig_var_20r)
F_prime_rig_var.append(F_prime_rig_var_100r)

Z_mean = []
Z_mean.append(Z_mean_20r)
Z_mean.append(Z_mean_100r)

Znet_mean = []
Znet_mean.append(Znet_mean_20r)
Znet_mean.append(Znet_mean_100r)



def funcMP(phi, phiM, alpha):
    return np.log10(alpha / ((1.-(phi/phiM))**2.))





#%% PLOTS ALL NPs

comparingDir = "../00_comparingSigmas"
if not os.path.exists(comparingDir):
    os.mkdir(comparingDir)

plt.close('all')



fig, axs = plt.subplots(1,len(sigmas), sharex=True, sharey=True, figsize=(10,5))
axs[0].set_ylabel(r"$\eta^{S}$")
for k in range(len(sigmas)):
    index_jam = -np.ones(len(NP), dtype=int)
    axs[k].set_xlabel(r"$\phi$")
    axs[k].set_title(r'$\sigma_0 =\ $'+str(sigmas[k])+r'$\sigma_{r}$')
    for i in range(len(NP)):
        myColor = myCmap(i/(len(NP)-1))
        for j in range(len(phi[k])):
            if np.abs(etaS_mean[k][i][j]) > etaS_maxFit and index_jam[i] == -1:
                index_jam[i] = j
        if index_jam[i] == -1:
            MP_fit    = opt.curve_fit(funcMP, phi[k], np.log10(etaS_mean[k][i]), [phi[k][-1]+0.01,1.0])
            phi_dense = np.linspace(phi[k][0], MP_fit[0][0]-1e-6, 100)
            axs[k].axvline(x=MP_fit[0][0], color=myColor, linestyle='--')
            axs[k].plot(phi_dense, 10.**(funcMP(phi_dense, MP_fit[0][0], MP_fit[0][1])), color=myColor)
            axs[k].plot(phi[k], etaS_mean[k][i], linestyle="--", marker=symbols[i], color=myColor, label=r"$NP = \ $"+str(NP[i])+r"$\ \left(\phi_{J}=\right.$"+'{:.4f}'.format(MP_fit[0][0])+r"$\left.\right)$")
        else:
            MP_fit    = opt.curve_fit(funcMP, phi[k][:index_jam[i]], np.log10(etaS_mean[k][i][:index_jam[i]]), [phi[k][index_jam[i]]+0.02,1.0])
            phi_dense = np.linspace(phi[k][0], MP_fit[0][0]-1e-6, 100)
            axs[k].axvline(x=MP_fit[0][0], color=myColor, linestyle='--')
            axs[k].plot(phi_dense, 10.**(funcMP(phi_dense, MP_fit[0][0], MP_fit[0][1])), color=myColor)
            axs[k].plot(phi[k][:index_jam[i]], etaS_mean[k][i][:index_jam[i]], linestyle="--", marker=symbols[i], color=myColor, label=r"$NP = \ $"+str(NP[i])+r"$\ \left(\phi_{J}=\right.$"+'{:.4f}'.format(MP_fit[0][0])+r"$\left.\right)$")
    axs[k].set_ylim([100,4000])
    axs[k].set_yscale('log')
    axs[k].grid(which='Both', alpha=0.2)
    axs[k].legend(fontsize=8)
fig.savefig(comparingDir+"/etaS_vs_phi"+figFormat, bbox_inches="tight")



fig, axs = plt.subplots(2,len(sigmas), sharex=True, sharey='row', figsize=(10,8))
for k in range(len(sigmas)):
    axs[0][k].set_title(r'$\sigma_0 =\ $'+str(sigmas[k])+r'$\sigma_{r}$')
    for i in range(len(NP)):
        if i > 1:
            myColor = myCmap((i-2)/(len(NP)-3))
            axs[0][k].plot(phi[k],     f_rig_mean[k][i], linestyle="--", marker=symbols[i], color=myColor, label="NP = "+str(NP[i]))
            axs[1][k].semilogy(phi[k], F_rig_var[k][i]/NP[i],  linestyle="--", marker=symbols[i], color=myColor, label="NP = "+str(NP[i]))
    axs[1][k].set_xlabel(r'$\phi$')
    axs[0][k].grid(which='Both', alpha=0.2)
    axs[1][k].grid(which='Both', alpha=0.2)
axs[0][0].legend(fontsize=8)
axs[0][0].set_ylabel(r'$\left\langle f_{rig} \right\rangle$')
axs[1][0].set_ylabel(r'$\left\langle \mathrm{var}\left(F_{rig}\right) \right\rangle$')
fig.savefig(comparingDir+"/frig_varFrig"+figFormat, bbox_inches="tight")



fig, axs = plt.subplots(2,len(sigmas), sharex=True, sharey='row', figsize=(10,8))
for k in range(len(sigmas)):
    axs[0][k].set_title(r'$\sigma_0 =\ $'+str(sigmas[k])+r'$\sigma_{r}$')
    for i in range(len(NP)):
        if i > 1:
            myColor = myCmap((i-2)/(len(NP)-3))
            axs[0][k].plot(phi[k],     f_prime_rig_mean[k][i], linestyle="--", marker=symbols[i], color=myColor, label="NP = "+str(NP[i]))
            axs[1][k].semilogy(phi[k], F_prime_rig_var[k][i],  linestyle="--", marker=symbols[i], color=myColor, label="NP = "+str(NP[i]))
    axs[1][k].set_xlabel(r'$\phi$')
    axs[0][k].grid(which='Both', alpha=0.2)
    axs[1][k].grid(which='Both', alpha=0.2)
axs[0][0].legend(fontsize=8)
axs[0][0].set_ylabel(r"$\left\langle f^{'}_{rig} \right\rangle$")
axs[1][0].set_ylabel(r"$\left\langle \mathrm{var}\left(F^{'}_{rig}\right) \right\rangle$")
fig.savefig(comparingDir+"/frig_varFrig_prime"+figFormat, bbox_inches="tight")



fig, axs = plt.subplots(2,len(sigmas), sharex=True, sharey='row', figsize=(10,8))
for k in range(len(sigmas)):
    axs[0][k].set_title(r'$\sigma_0 =\ $'+str(sigmas[k])+r'$\sigma_{r}$')
    axs[1][k].set_xlabel(r'$\phi$')
    for i in range(len(NP)):
        if i > 1:
            myColor = myCmap((i-2)/(len(NP)-3))
            axs[0][k].plot(phi[k], Z_mean[k][i],    linestyle="--", marker=symbols[i], color=myColor, label="NP = "+str(NP[i]))
            axs[1][k].plot(phi[k], Znet_mean[k][i], linestyle="--", marker=symbols[i], color=myColor, label="NP = "+str(NP[i]))
    axs[0][k].grid(which='Both', alpha=0.2)
    axs[1][k].grid(which='Both', alpha=0.2)
axs[0][0].legend(fontsize=8)
axs[0][0].set_ylabel(r'$\left\langle Z \right\rangle$')
axs[1][0].set_ylabel(r'$\left\langle Z_{net} \right\rangle$')
fig.savefig(comparingDir+"/Z_Znet"+figFormat, bbox_inches="tight")



fig, axs = plt.subplots(2,len(sigmas), sharex='row', sharey='row', figsize=(10,8))
fig.suptitle(r'$NP = 2000$')
axs[0][0].set_ylabel(r'$C\left(\Delta t\right) / C\left(0\right)$')
axs[1][0].set_ylabel(r'$C\left(\Delta\gamma\right) / C\left(0\right)$')
for k in range(len(sigmas)):
    axs[0][k].set_title(r'$\sigma_0 =\ $'+str(sigmas[k])+r'$\sigma_{r}$')
    axs[0][k].set_xlabel(r'$\Delta t$')
    axs[1][k].set_xlabel(r'$\Delta \gamma$')
    for i in range(len(NP)):
        if NP[i] == 2000:
            for j in range(len(phi[k])):
                myColor = myCmap((phi[k][j]-minPhi)/(maxPhi-minPhi))
                Dir = TopDir[k] + "NP" + str(NP[i]) + "/phi0." + str(int(phi[k][j]*1000)) + "/"
                tau, rigPers = np.loadtxt(Dir+"rigPers.txt", skiprows=1).transpose()
                axs[0][k].semilogx(tau,                    rigPers/rigPers[0], color=myColor)
                axs[1][k].semilogx(tau/etaS_mean[k][i][j], rigPers/rigPers[0], color=myColor)
    axs[0][k].grid(which='Both', alpha=0.2)
    axs[1][k].grid(which='Both', alpha=0.2)
del Dir, tau, rigPers
fig.savefig(comparingDir+"/rigPers"+figFormat, bbox_inches="tight")



fig, axs = plt.subplots(2,len(sigmas), sharex='row', sharey='row', figsize=(10,8))
fig.suptitle(r'$NP = 2000$')
axs[0][0].set_ylabel(r'$n_{max}\left(\Delta t\right) / n_{max}\left(0\right)$')
axs[1][0].set_ylabel(r'$n_{max}\left(\Delta\gamma\right) / n_{max}\left(0\right)$')
for k in range(len(sigmas)):
    axs[0][k].set_title(r'$\sigma_0 =\ $'+str(sigmas[k])+r'$\sigma_{r}$')
    axs[0][k].set_xlabel(r'$\Delta t$')
    axs[1][k].set_xlabel(r'$\Delta \gamma$')
    for i in range(len(NP)):
        if NP[i] == 2000:
            for j in range(len(phi[k])):
                myColor = myCmap((phi[k][j]-minPhi)/(maxPhi-minPhi))
                Dir = TopDir[k] + "NP" + str(NP[i]) + "/phi0." + str(int(phi[k][j]*1000)) + "/"
                tau, nMax_corr = np.loadtxt(Dir+"maxClusterSize_corr.txt", skiprows=1).transpose()
                axs[0][k].semilogx(tau,                    nMax_corr/nMax_corr[0], color=myColor)
                axs[1][k].semilogx(tau/etaS_mean[k][i][j], nMax_corr/nMax_corr[0], color=myColor)
    axs[0][k].grid(which='Both', alpha=0.2)
    axs[1][k].grid(which='Both', alpha=0.2)
del Dir, tau, nMax_corr
fig.savefig(comparingDir+"/maxClustersSize_corr"+figFormat, bbox_inches="tight")



fig, axs = plt.subplots(1,len(sigmas), sharex=True, sharey=True, figsize=(8,4))
axs[0].set_ylabel(r'$C\left(\tau=0\right) / NP$')
for k in range(len(sigmas)):
    axs[k].set_title(r'$\sigma_0 =\ $'+str(sigmas[k])+r'$\sigma_{r}$')
    axs[k].set_xlabel(r'$\phi$')
    for i in range(2,len(NP)):
        myColor = myCmap((i-2)/(len(NP)-3))
        rigPersZero = np.zeros(len(phi[k]))
        for j in range(len(phi[k])):
            Dir = TopDir[k] + "NP" + str(NP[i]) + "/phi0." + str(int(phi[k][j]*1000)) + "/"
            tau, rigPers = np.loadtxt(Dir+"rigPers.txt", skiprows=1).transpose()
            rigPersZero[j] = rigPers[0]
        axs[k].plot(phi[k], rigPersZero/NP[i], linestyle="--", marker=symbols[i], color=myColor, label="NP = "+str(NP[i]))
    axs[k].grid(which='Both', alpha=0.2)
axs[0].legend()
del rigPersZero
fig.savefig(comparingDir+"/rigPers_tau0"+figFormat, bbox_inches="tight")



fig, axs = plt.subplots(1,len(sigmas), sharex=True, sharey=True, figsize=(8,4))
axs[0].set_ylabel(r'$C\left(\tau=0\right) / NP$')
for k in range(len(sigmas)):
    axs[k].set_title(r'$\sigma_0 =\ $'+str(sigmas[k])+r'$\sigma_{r}$')
    axs[k].set_xlabel(r'$\phi$')
    for i in range(2,len(NP)):
        myColor = myCmap((i-2)/(len(NP)-3))
        frictPersZero = np.zeros(len(phi[k]))
        for j in range(len(phi[k])):
            Dir = TopDir[k] + "NP" + str(NP[i]) + "/phi0." + str(int(phi[k][j]*1000)) + "/"
            tau, frictPers = np.loadtxt(Dir+"frictPers.txt", skiprows=1).transpose()
            frictPersZero[j] = frictPers[0]
        axs[k].plot(phi[k], frictPersZero/NP[i], linestyle="--", marker=symbols[i], color=myColor, label="NP = "+str(NP[i]))
    axs[k].grid(which='Both', alpha=0.2)
axs[0].legend()
del frictPersZero
fig.savefig(comparingDir+"/frictPers_tau0"+figFormat, bbox_inches="tight")



# -----------------------------------------------------------------------------
# PLOTS FOR NP = 2000

myColors  = ['r', 'b']
mySymbols = ['o', '^']



slope = -2



fig, axs = plt.subplots(1,len(sigmas)+1, width_ratios=[20,20,1], figsize=(8,4))
fig.suptitle(r'$NP = 2000$')
cb = mpl.colorbar.ColorbarBase(axs[len(sigmas)], orientation='vertical', cmap=myCmap, label=r'$\phi$', norm=mpl.colors.Normalize(minPhi,maxPhi))
axs[0].set_ylabel(r'$P\left(n\right)$')
for k in range(len(sigmas)):
    axs[k].set_title(r'$\sigma_0 =\ $'+str(sigmas[k])+r'$\sigma_{r}$')
    axs[k].set_xlabel(r'$n$')
    for i in range(len(NP)):
        if NP[i] == 2000:
            for j in range(len(phi[k])):
                if phi[k][j] < 0.763:
                    myColor  = myCmap((phi[k][j]-minPhi)/(maxPhi-minPhi))
                    Dir      = TopDir[k] + "NP" + str(NP[i]) + "/phi0." + str(int(phi[k][j]*1000)) + "/"
                    baseName = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
                    dataFile = Dir + 'data_' + baseName
                    rigFile  = Dir + 'rig_'  + baseName
                    t,     dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
                    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
                    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
                        = np.loadtxt(dataFile, skiprows=37).transpose()
                    SSi = np.where(t==steadyStateTime.t_SS(k,i,j))[0][0]
                    clusterSizes = []
                    for it in range(SSi,len(t)):
                        clusterSizes.append([int(x) for x in np.frombuffer(np.loadtxt(rigFile, skiprows=1+it, max_rows=1))])
                    clusterSizes = [item for sublist in clusterSizes for item in sublist]
                    clusterSizes = [i for i in clusterSizes if i != 0]
                    cedges = np.arange(np.min(clusterSizes)-0.5, np.max(clusterSizes)+1.5)
                    cbins  = np.arange(np.min(clusterSizes),     np.max(clusterSizes)+1)
                    Pn, dummy = np.histogram(clusterSizes[SSi:], cedges, density=True)
                    axs[k].loglog(cbins, Pn, marker='.', linestyle='None', color=myColor)
                    if not np.all(np.diff(cbins)==1):
                        sys.exit('- Something is wrong with the bins for P(n)  >  STOP !')
                    if not np.all(np.diff(cedges)==1):
                        sys.exit('- Something is wrong with the bin edges for P(n)  >  STOP !')
                    if not np.abs(np.sum(Pn)-1)<1e-9:
                        sys.exit('- Something is wrong with P(n) (sum(P) = ' + str(np.sum(Pn)) + ')  >  STOP !')
    axs[k].loglog(np.linspace(2.7,2000,100), np.power(np.linspace(2.7,2000,100),slope), 'r', label='slope '+str(slope))
    axs[k].grid(which='Both', alpha=0.2)
    axs[k].set_xlim([2.7,2000])
    axs[k].set_ylim([1e-5,1])
axs[0].legend()
fig.savefig(comparingDir+"/Pn"+figFormat, bbox_inches="tight")



fig, axs = plt.subplots(1,len(sigmas)+1, width_ratios=[20,20,1], figsize=(8,4))
fig.suptitle(r'$NP = 2000$')
cb = mpl.colorbar.ColorbarBase(axs[len(sigmas)], orientation='vertical', cmap=myCmap, label=r'$\phi$', norm=mpl.colors.Normalize(minPhi,maxPhi))
axs[0].set_ylabel(r"$P^{'}\left(n\right)$")
for k in range(len(sigmas)):
    axs[k].set_title(r'$\sigma_0 =\ $'+str(sigmas[k])+r'$\sigma_{r}$')
    axs[k].set_xlabel(r'$n$')
    for i in range(len(NP)):
        if NP[i] == 2000:
            for j in range(len(phi[k])):
                if phi[k][j] < 0.763:
                    myColor  = myCmap((phi[k][j]-minPhi)/(maxPhi-minPhi))
                    Dir      = TopDir[k] + "NP" + str(NP[i]) + "/phi0." + str(int(phi[k][j]*1000)) + "/"
                    baseName = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
                    dataFile = Dir + 'data_' + baseName
                    rigFile  = Dir + 'rig_'  + baseName
                    t,     dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
                    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
                    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
                        = np.loadtxt(dataFile, skiprows=37).transpose()
                    SSi = np.where(t==steadyStateTime.t_SS(k,i,j))[0][0]
                    primeClusterSizes = []
                    for it in range(SSi,len(t)):
                        primeClusterSizes.append([int(x) for x in np.frombuffer(np.loadtxt(rigFile, skiprows=1+it, max_rows=1))])
                    primeClusterSizes = [item for sublist in primeClusterSizes for item in sublist]
                    primeClusterSizes = [i for i in primeClusterSizes if i != 0]
                    cedges = np.arange(np.min(primeClusterSizes)-0.5, np.max(primeClusterSizes)+1.5)
                    cbins  = np.arange(np.min(primeClusterSizes),     np.max(primeClusterSizes)+1)
                    Pn_prime, dummy = np.histogram(primeClusterSizes[SSi:], cedges, density=True)
                    axs[k].loglog(cbins, Pn_prime, marker='.', linestyle='None', color=myColor)
                    if not np.all(np.diff(cbins)==1):
                        sys.exit("- Something is wrong with the bins for P'(n)  >  STOP !")
                    if not np.all(np.diff(cedges)==1):
                        sys.exit("- Something is wrong with the bin edges for P'(n)  >  STOP !")
                    if not np.abs(np.sum(Pn_prime)-1)<1e-9:
                        sys.exit("- Something is wrong with P'(n) (sum(P') = " + str(np.sum(Pn_prime)) + ")  >  STOP !")
    axs[k].loglog(np.linspace(2.7,2000,100), np.power(np.linspace(2.7,2000,100),slope), 'r', label='slope '+str(slope))
    axs[k].grid(which='Both', alpha=0.2)
    axs[k].set_xlim([2.7,2000])
    axs[k].set_ylim([1e-5,1])
axs[0].legend()
fig.savefig(comparingDir+"/Pn_prime"+figFormat, bbox_inches="tight")



fig, axs = plt.subplots(1,len(sigmas)+1, width_ratios=[20,20,1], figsize=(8,4))
fig.suptitle(r'$NP = 2000$')
cb = mpl.colorbar.ColorbarBase(axs[len(sigmas)], orientation='vertical', cmap=myCmap, label=r'$\phi$', norm=mpl.colors.Normalize(minPhi,maxPhi))
axs[0].set_ylabel(r"$P\left(n^{'}\right)$")
for k in range(len(sigmas)):
    axs[k].set_title(r'$\sigma_0 =\ $'+str(sigmas[k])+r'$\sigma_{r}$')
    axs[k].set_xlabel(r"$n^{'}$")
    for i in range(len(NP)):
        if NP[i] == 2000:
            for j in range(len(phi[k])):
                if phi[k][j] < 0.763:
                    myColor  = myCmap((phi[k][j]-minPhi)/(maxPhi-minPhi))
                    Dir      = TopDir[k] + "NP" + str(NP[i]) + "/phi0." + str(int(phi[k][j]*1000)) + "/"
                    baseName = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
                    dataFile = Dir + 'data_' + baseName
                    rigFile  = Dir + 'rigPrime.txt'
                    t,     dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
                    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
                    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
                        = np.loadtxt(dataFile, skiprows=37).transpose()
                    SSi = np.where(t==steadyStateTime.t_SS(k,i,j))[0][0]
                    clusterSizes = []
                    for it in range(SSi,len(t)):
                        clusterSizes.append([int(x) for x in np.frombuffer(np.loadtxt(rigFile, skiprows=1+it, max_rows=1))])
                    clusterSizes = [item for sublist in clusterSizes for item in sublist]
                    clusterSizes = [i for i in clusterSizes if i != 0]
                    cedges = np.arange(np.min(clusterSizes)-0.5, np.max(clusterSizes)+1.5)
                    cbins  = np.arange(np.min(clusterSizes),     np.max(clusterSizes)+1)
                    Pn, dummy = np.histogram(clusterSizes[SSi:], cedges, density=True)
                    axs[k].loglog(cbins, Pn, marker='.', linestyle='None', color=myColor)
                    if not np.all(np.diff(cbins)==1):
                        sys.exit("- Something is wrong with the bins for P(n')  >  STOP !")
                    if not np.all(np.diff(cedges)==1):
                        sys.exit("- Something is wrong with the bin edges for P(n')  >  STOP !")
                    if not np.abs(np.sum(Pn)-1)<1e-9:
                        sys.exit("- Something is wrong with P(n') (sum(P) = " + str(np.sum(Pn)) + ')  >  STOP !')
    axs[k].loglog(np.linspace(2.7,2000,100), np.power(np.linspace(2.7,2000,100),slope), 'r', label='slope '+str(slope))
    axs[k].grid(which='Both', alpha=0.2)
    axs[k].set_xlim([2.7,2000])
    axs[k].set_ylim([1e-5,1])
axs[0].legend()
fig.savefig(comparingDir+"/Pn_prime"+figFormat, bbox_inches="tight")



fig, axs = plt.subplots(2,2, sharex=True, sharey='col', figsize=(10,8))
fig.suptitle('NP = 2000')
for k in range(len(sigmas)):
    for i in range(len(NP)):
        if NP[i] == 2000:
            axs[0][0].plot(phi[k],     f_rig_mean[k][i],            linestyle="--", marker=mySymbols[k], color=myColors[k], label=r'$\sigma_0 =\ $'+str(sigmas[k])+r'$\sigma_{r}$')
            axs[0][1].semilogy(phi[k], F_rig_var[k][i]/NP[i],       linestyle="--", marker=mySymbols[k], color=myColors[k], label=r'$\sigma_0 =\ $'+str(sigmas[k])+r'$\sigma_{r}$')
            axs[1][0].plot(phi[k],     f_prime_rig_mean[k][i],      linestyle="--", marker=mySymbols[k], color=myColors[k], label=r'$\sigma_0 =\ $'+str(sigmas[k])+r'$\sigma_{r}$')
            axs[1][1].semilogy(phi[k], F_prime_rig_var[k][i]/NP[i], linestyle="--", marker=mySymbols[k], color=myColors[k], label=r'$\sigma_0 =\ $'+str(sigmas[k])+r'$\sigma_{r}$')
axs[1][0].set_xlabel(r'$\phi$')
axs[1][1].set_xlabel(r'$\phi$')
axs[0][0].grid(which='Both', alpha=0.2)
axs[0][1].grid(which='Both', alpha=0.2)
axs[1][0].grid(which='Both', alpha=0.2)
axs[1][1].grid(which='Both', alpha=0.2)
axs[0][0].legend(fontsize=8)
axs[0][0].set_ylabel(r'$\left\langle f_{rig} \right\rangle$')
axs[0][1].set_ylabel(r'$\left\langle \mathrm{var}\left(F_{rig}\right) \right\rangle$')
axs[1][0].set_ylabel(r"$\left\langle f^{'}_{rig} \right\rangle$")
axs[1][1].set_ylabel(r"$\left\langle \mathrm{var}\left(F^{'}_{rig}\right) \right\rangle$")
fig.savefig(comparingDir+"/frig_varFrig_NP2000"+figFormat, bbox_inches="tight")



fig, axs = plt.subplots(1,2, sharex=True, sharey=True, figsize=(10,5))
fig.suptitle('NP = 2000')
for k in range(len(sigmas)):
    for i in range(len(NP)):
        if NP[i] == 2000:
            axs[0].plot(phi[k], Z_mean[k][i],    linestyle="--", marker=mySymbols[k], color=myColors[k], label=r'$\sigma_0 =\ $'+str(sigmas[k])+r'$\sigma_{r}$')
            axs[1].plot(phi[k], Znet_mean[k][i], linestyle="--", marker=mySymbols[k], color=myColors[k], label=r'$\sigma_0 =\ $'+str(sigmas[k])+r'$\sigma_{r}$')
axs[0].grid(which='Both', alpha=0.2)
axs[1].grid(which='Both', alpha=0.2)
axs[0].legend(fontsize=8)
axs[0].set_xlabel(r'$\phi$')
axs[1].set_xlabel(r'$\phi$')
axs[0].set_ylabel(r'$\left\langle Z \right\rangle$')
axs[1].set_ylabel(r'$\left\langle Z_{net} \right\rangle$')
fig.savefig(comparingDir+"/Z_Znet_NP2000"+figFormat, bbox_inches="tight")



fig, ax = plt.subplots(1,1, figsize=(6,5))
fig.suptitle('NP = 2000')
for k in range(len(sigmas)):
    for i in range(len(NP)):
        if NP[i] == 2000:
            rigPersZero = np.zeros(len(phi[k]))
            for j in range(len(phi[k])):
                Dir = TopDir[k] + "NP" + str(NP[i]) + "/phi0." + str(int(phi[k][j]*1000)) + "/"
                tau, rigPers = np.loadtxt(Dir+"rigPers.txt", skiprows=1).transpose()
                rigPersZero[j] = rigPers[0]
            ax.plot(phi[k], rigPersZero/NP[i], linestyle="--", marker=mySymbols[k], color=myColors[k], label=r'$\sigma_0 =\ $'+str(sigmas[k])+r'$\sigma_{r}$')
ax.grid(which='Both', alpha=0.2)
ax.set_xlabel(r'$\phi$')
ax.set_ylabel(r'$C\left(\tau=0\right) / NP$')
ax.legend()
del rigPersZero
fig.savefig(comparingDir+"/rigPers_tau0_NP2000"+figFormat, bbox_inches="tight")



fig, ax = plt.subplots(1,1, figsize=(6,5))
fig.suptitle('NP = 2000')
for k in range(len(sigmas)):
    for i in range(len(NP)):
        if NP[i] == 2000:
            frictPersZero = np.zeros(len(phi[k]))
            for j in range(len(phi[k])):
                Dir = TopDir[k] + "NP" + str(NP[i]) + "/phi0." + str(int(phi[k][j]*1000)) + "/"
                tau, frictPers = np.loadtxt(Dir+"frictPers.txt", skiprows=1).transpose()
                frictPersZero[j] = frictPers[0]
            ax.plot(phi[k], frictPersZero/NP[i], linestyle="--", marker=mySymbols[k], color=myColors[k], label=r'$\sigma_0 =\ $'+str(sigmas[k])+r'$\sigma_{r}$')
ax.grid(which='Both', alpha=0.2)
ax.set_xlabel(r'$\phi$')
ax.set_ylabel(r'$C\left(\tau=0\right) / NP$')
ax.legend()
del frictPersZero
fig.savefig(comparingDir+"/frictPers_tau0_NP2000"+figFormat, bbox_inches="tight")



