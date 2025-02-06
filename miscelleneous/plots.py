import os
import matplotlib
import numpy             as     np
import pandas            as     pd
import scipy.optimize    as     opt
import matplotlib.pyplot as     plt

from   fractions         import Fraction



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
cmap      = matplotlib.colormaps['viridis_r']
FigSize   = (13,13*9/16)





#%% INITIALIZING



plt.close('all')

print('')
print('Reading data')



indSig      = 1

sigmas      = np.array([20, 100])

sigma       = sigmas[indSig]

TopDir      = "../2D_sigma" + str(sigma) + "/"

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



def funcMP(phi, phiM, alpha):
    return np.log10(alpha / ((1.-(phi/phiM))**2.))

    
    
data = {}
for i in range(len(NP)):
    skip = 1
    if i > 0:
        skip = 2 + 2*i + (i-1)
        for j in range(i):
            skip += len(phi[indSig])
    data[i] = pd.read_csv(TopDir+"data.txt", delim_whitespace=True, skiprows=skip, nrows=len(phi[indSig]))

etaS_mean        = []
etaS_std         = []
f_rig_mean       = []
f_rig_std        = []
F_rig_mean       = []
F_rig_var        = []
f_prime_rig_mean = []
f_prime_rig_std  = []
F_prime_rig_mean = []
F_prime_rig_var  = []
Z_mean           = []
Z_std            = []
Znet_mean        = []
Znet_std         = []
for i in range(len(NP)):
    etaS_mean.append(np.array(data[i]['<etaS>']))
    etaS_std.append(np.array(data[i]['std(etaS)']))
    f_rig_mean.append(np.array(data[i]['<frig>']))
    f_rig_std.append(np.array(data[i]['std(frig)']))
    F_rig_mean.append(np.array(data[i]['<Frig>']))
    F_rig_var.append(np.array(data[i]['var(Frig)']))
    f_prime_rig_mean.append(np.array(data[i]["<f'rig>"]))
    f_prime_rig_std.append(np.array(data[i]["std(f'rig)"]))
    F_prime_rig_mean.append(np.array(data[i]["<F'rig>"]))
    F_prime_rig_var.append(np.array(data[i]["var(F'rig)"]))
    Z_mean.append(np.array(data[i]['<Z>']))
    Z_std.append(np.array(data[i]['std(Z)']))
    Znet_mean.append(np.array(data[i]['<Znet>']))
    Znet_std.append(np.array(data[i]['std(Znet)']))
    
del data





#%% RHEOLOGY PLOTS

rheoDir = TopDir + "00_rheo_plots"
if not os.path.exists(rheoDir):
    os.mkdir(rheoDir)
    
plt.close('all')



fig, ax = plt.subplots(1,1)
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$\eta^{S}$")
index_jam = -np.ones(len(NP), dtype=int)
for i in range(len(NP)):
    myColor = cmap((i+1)/len(NP))
    for j in range(len(phi[indSig])):
        if np.abs(etaS_mean[i][j]) > etaS_maxFit and index_jam[i] == -1:
            index_jam[i] = j
    if index_jam[i] == -1:
        MP_fit    = opt.curve_fit(funcMP, phi[indSig], np.log10(etaS_mean[i]), [phi[indSig][-1]+0.01,1.0])
        phi_dense = np.linspace(phi[indSig][0], MP_fit[0][0]-1e-6, 100)
        ax.axvline(x=MP_fit[0][0], color=myColor, linestyle='--')
        ax.plot(phi_dense, 10.**(funcMP(phi_dense, MP_fit[0][0], MP_fit[0][1])), color=myColor)
        ax.plot(phi[indSig], etaS_mean[i], linestyle="--",  marker=symbols[i],  color=myColor, label=r"$NP = \ $"+str(NP[i])+r"$\ \left(\phi_{J}=\right.$"+'{:.4f}'.format(MP_fit[0][0])+r"$\left.\right)$")
    else:
        MP_fit    = opt.curve_fit(funcMP, phi[indSig][:index_jam[i]], np.log10(etaS_mean[i][:index_jam[i]]), [phi[indSig][index_jam[i]]+0.02,1.0])
        phi_dense = np.linspace(phi[indSig][0], MP_fit[0][0]-1e-6, 100)
        ax.axvline(x=MP_fit[0][0], color=myColor, linestyle='--')
        ax.plot(phi_dense, 10.**(funcMP(phi_dense, MP_fit[0][0], MP_fit[0][1])), color=myColor)
        ax.plot(phi[indSig][:index_jam[i]], etaS_mean[i][:index_jam[i]], linestyle="--",  marker=symbols[i],  color=myColor, label=r"$NP = \ $"+str(NP[i])+r"$\ \left(\phi_{J}=\right.$"+'{:.4f}'.format(MP_fit[0][0])+r"$\left.\right)$")
ax.set_ylim([100,4000])
ax.set_yscale('log')
ax.grid(which='Both', alpha=0.2)
ax.legend(fontsize=8)
fig.savefig(rheoDir+"/etaS_vs_phi"+figFormat, bbox_inches="tight")



fig, (ax1,ax2) = plt.subplots(1,2, figsize=FigSize)
ax1.set_xlabel(r'$\phi$')
ax2.set_xlabel(r'$\phi$')
ax1.set_ylabel(r'$\left\langle f_{rig} \right\rangle$')
ax2.set_ylabel(r'$\text{var}\left(F_{rig}\right)$')
for i in range(len(NP)):
    myColor = cmap((i+1)/len(NP))
    ax1.semilogy(phi[indSig], f_rig_mean[i], linestyle="--", marker=symbols[i], color=myColor, label="NP = "+str(NP[i]))
    ax2.semilogy(phi[indSig], F_rig_var[i],  linestyle="--", marker=symbols[i], color=myColor, label="NP = "+str(NP[i]))
ax1.grid(which='Both', alpha=0.2)
ax2.grid(which='Both', alpha=0.2)
ax1.legend(fontsize=8)
ax2.legend(fontsize=8)
fig.savefig(rheoDir+"/frig_varFrig_vs_phi"+figFormat, bbox_inches="tight")



fig, (ax1,ax2) = plt.subplots(1,2, figsize=FigSize)
ax1.set_xlabel(r'$\phi$')
ax2.set_xlabel(r'$\phi$')
ax1.set_ylabel(r'$\left\langle Z \right\rangle$')
ax2.set_ylabel(r'$\left\langle Z_{net} \right\rangle$')
for i in range(len(NP)):
    myColor = cmap((i+1)/len(NP))
    ax1.plot(phi[indSig], Z_mean[i],    linestyle="--", marker=symbols[i], color=myColor, label="NP = "+str(NP[i]))
    ax2.plot(phi[indSig], Znet_mean[i], linestyle="--", marker=symbols[i], color=myColor, label="NP = "+str(NP[i]))
ax1.grid(which='Both', alpha=0.2)
ax2.grid(which='Both', alpha=0.2)
ax1.legend(fontsize=8)
ax2.legend(fontsize=8)
fig.savefig(rheoDir+"/Z_Znet_vs_phi"+figFormat, bbox_inches="tight")



for i in range(len(NP)):
    
    if indSig == 0:
        fig1, axs1 = plt.subplots(5,8, sharex=True, sharey=True)
        fig1.suptitle("NP = "+str(NP[i]))
        fig2, axs2 = plt.subplots(5,8, sharex=True, sharey=True)
        fig2.suptitle("NP = "+str(NP[i]))
    elif indSig == 1:
        fig1, axs1 = plt.subplots(5,6, sharex=True, sharey=True)
        fig1.suptitle("NP = "+str(NP[i]))
        fig2, axs2 = plt.subplots(5,6, sharex=True, sharey=True)
        fig2.suptitle("NP = "+str(NP[i]))
    
    for j in range(len(phi[indSig])):
        if j > 0:
        
            Dir = TopDir + "NP" + str(NP[i]) + "/phi0." + str(int(phi[indSig][j]*1000)) + "/"
            tau, rigPers = np.loadtxt(Dir+"rigPers.txt", skiprows=1).transpose()
            
            row = int((j-1)/len(axs1[0]))
            col = (j-1) % len(axs1[0])
            if row == len(axs1)-1:
                axs1[row,col].set_xlabel(r'$\left(\Delta t\right)$')
                axs2[row,col].set_xlabel(r'$\left(\Delta\gamma\right)$')
            if col == 0:
                axs1[row,col].set_ylabel(r'$C\left(\Delta t\right) / C\left(0\right)$')
                axs2[row,col].set_ylabel(r'$C\left(\Delta\gamma\right) / C\left(0\right)$')
                
            axs1[row,col].set_title(r"$\phi = \ $"+str(phi[indSig][j]), fontsize=8)
            axs2[row,col].set_title(r"$\phi = \ $"+str(phi[indSig][j]), fontsize=8)
            
            axs1[row,col].semilogx(tau,                 rigPers/rigPers[0], 'k')
            axs2[row,col].semilogx(tau/etaS_mean[i][j], rigPers/rigPers[0], 'k')
            
            axs1[row,col].grid(which='Both', alpha=0.2)
            axs2[row,col].grid(which='Both', alpha=0.2)
    
    fig1.savefig(rheoDir+"/rigPers_time"  +"NP"+str(NP[i])+figFormat, bbox_inches="tight")
    fig2.savefig(rheoDir+"/rigPers_strain"+"NP"+str(NP[i])+figFormat, bbox_inches="tight")



for i in range(len(NP)):
    
    if indSig == 0:
        fig, axs = plt.subplots(5,8, sharex=True, sharey=True)
        fig.suptitle("NP = "+str(NP[i]))
    elif indSig == 1:
        fig, axs = plt.subplots(5,6, sharex=True, sharey=True)
        fig.suptitle("NP = "+str(NP[i]))
    
    for j in range(len(phi[indSig])):
        if j > 0:
        
            Dir = TopDir + "NP" + str(NP[i]) + "/phi0." + str(int(phi[indSig][j]*1000)) + "/"
            tau, frictPers = np.loadtxt(Dir+"frictPers.txt", skiprows=1).transpose()
            
            row = int((j-1)/len(axs[0]))
            col = (j-1) % len(axs[0])
            if row == len(axs)-1:
                axs[row,col].set_xlabel(r'$\left(\Delta t\right)$')
            if col == 0:
                axs[row,col].set_ylabel(r'$C\left(\Delta t\right) / C\left(0\right)$')
                
            axs[row,col].set_title(r"$\phi = \ $"+str(phi[indSig][j]), fontsize=8)
            axs[row,col].semilogx(tau, frictPers/frictPers[0], 'k')
            axs[row,col].grid(which='Both', alpha=0.2)
    
    fig.savefig(rheoDir+"/frictPers_"+"NP"+str(NP[i])+figFormat, bbox_inches="tight")        



fig, ax = plt.subplots(1,1)
ax.set_xlabel(r'$\phi$')
ax.set_ylabel(r'$C\left(\tau=0\right) / NP$')
for i in range(len(NP)):
    if i > 1:
        rigPers_max = np.zeros(len(phi[indSig]))
        for j in range(len(phi[indSig])):
            Dir = TopDir + "NP" + str(NP[i]) + "/phi0." + str(int(phi[indSig][j]*1000)) + "/"
            tau, rigPers = np.loadtxt(Dir+"rigPers.txt", skiprows=1).transpose()
            rigPers_max[j] = rigPers[0]
        myColor = cmap((i+1-2)/(len(NP)-2))
        ax.plot(phi[indSig], rigPers_max/NP[i], linestyle="--", marker=symbols[i], color=myColor, label="NP = "+str(NP[i]))
ax.grid(which='Both', alpha=0.2)
ax.legend(fontsize=8)
fig.savefig(rheoDir+"/rigPers_tau0"+figFormat, bbox_inches="tight")  



fig, axs = plt.subplots(2,2, sharex=True, sharey="row", figsize=FigSize)
for i in range(len(NP)):
    myColor = cmap((i+1)/len(NP))
    axs[0][0].set_xlabel(r'$\phi$')
    axs[0][1].set_xlabel(r'$\phi$')
    axs[1][0].set_xlabel(r'$\phi$')
    axs[1][1].set_xlabel(r'$\phi$')
    axs[0][0].set_ylabel(r'$\left\langle f_{rig} \right\rangle$')
    axs[0][1].set_ylabel(r"$\left\langle f^{'}_{rig} \right\rangle$")
    axs[1][0].set_ylabel(r'$\text{var}\left(F_{rig}\right)$')
    axs[1][1].set_ylabel(r"$\text{var}\left(F^{'}_{rig}\right)$")
    axs[0][0].semilogy(phi[indSig], f_rig_mean[i],       linestyle="--", marker=symbols[i], color=myColor, label="NP = "+str(NP[i]))
    axs[0][1].semilogy(phi[indSig], f_prime_rig_mean[i], linestyle="--", marker=symbols[i], color=myColor, label="NP = "+str(NP[i]))
    axs[1][0].semilogy(phi[indSig], F_rig_var[i],        linestyle="--", marker=symbols[i], color=myColor, label="NP = "+str(NP[i]))
    axs[1][1].semilogy(phi[indSig], F_prime_rig_var[i],  linestyle="--", marker=symbols[i], color=myColor, label="NP = "+str(NP[i]))
    axs[0][0].grid(which='Both', alpha=0.2)
    axs[0][1].grid(which='Both', alpha=0.2)
    axs[1][0].grid(which='Both', alpha=0.2)
    axs[1][1].grid(which='Both', alpha=0.2)
    axs[0][0].legend()
fig.savefig(rheoDir+"/f_prime_rig"+figFormat, bbox_inches="tight")





#%% SCALING PLOTS

scalingDir = TopDir + "00_scaling_plots"
if not os.path.exists(scalingDir):
    os.mkdir(scalingDir)
    
plt.close('all')



nu        = 1
gamma     = 1.75
beta      = 1/8
phi_c_inf = 0.7561



L       = np.sqrt(NP)
corrLen = []
for i in range(len(NP)):
    corrLen.append((np.array(phi[indSig])-phi_c_inf)/phi_c_inf)
    
phi_c_L = np.zeros(len(NP))
for i in range(len(NP)):
    phi_c_L[i] = phi[indSig][np.where(F_rig_var[i]==np.max(F_rig_var[i]))[0][0]]

below = []
above = []
for i in range(len(NP)):
    above.append(np.where(phi[indSig]>phi_c_L[i])[0])
    below.append(np.where(phi[indSig]<phi_c_L[i])[0])





fig, ax = plt.subplots(1,1)
fig.suptitle(r'$\phi_{c}^{inf} = \ $'+str(phi_c_inf)+r'$\qquad\nu = \ $'+'{:.2f}'.format(nu)+r'$\qquad\gamma = \ $'+'{:.2f}'.format(gamma))
ax.set_xlabel(r'$L^{1/\nu} \cdot \left(\dfrac{\phi-\phi_{c}^{inf}}{\phi_{c}^{inf}}\right)$')
ax.set_ylabel(r'$\text{var}\left(F_{rig}\right) \cdot L^{-\gamma/\nu}$')
for i in range(1,len(NP)):
    myColor = cmap((i+1)/len(NP))
    x = np.power(L[i],1./nu) * corrLen[i]
    y = F_rig_var[i] * np.power(L[i], -gamma/nu)
    ax.plot(x, y, marker=symbols[i], color=myColor, linestyle='', label="NP = "+str(NP[i]))
ax.grid(which='Both', alpha=0.2)
ax.legend(fontsize=8)
fig.savefig(scalingDir+"/scaling1"+figFormat, bbox_inches="tight")



fig, ax = plt.subplots(1,1)
fig.suptitle(r'$\phi_{c}^{inf} = \ $'+str(phi_c_inf)+r'$\qquad\nu = \ $'+'{:.2f}'.format(nu)+r'$\qquad\gamma = \ $'+'{:.2f}'.format(gamma))
ax.set_xlabel(r'$L^{1/\nu} \cdot \left(\dfrac{\phi-\phi_{c}^{inf}}{\phi_{c}^{inf}}\right)$')
ax.set_ylabel(r'$\text{var}\left(F_{rig}\right) \cdot L^{-\gamma/\nu}$')
for i in range(1,len(NP)):
    myColor = cmap((i+1)/len(NP))
    x = np.power(L[i],1./nu) * corrLen[i]
    y = F_rig_var[i] * np.power(np.abs(corrLen[i]), -gamma/nu)
    ax.plot(x, y, marker=symbols[i], color=myColor, linestyle='', label="NP = "+str(NP[i]))
ax.grid(which='Both', alpha=0.2)
ax.legend(fontsize=8)
fig.savefig(scalingDir+"/scaling2"+figFormat, bbox_inches="tight")



fig, ax = plt.subplots(1,1)
fig.suptitle(r'$\phi_{c}^{inf} = \ $'+str(phi_c_inf)+r'$\qquad\nu = \ $'+'{:.2f}'.format(nu)+r'$\qquad\beta = \ $'+str(Fraction(beta)))
ax.set_xlabel(r'$L^{1/\nu} \cdot \left(\dfrac{\phi-\phi_{c}^{inf}}{\phi_{c}^{inf}}\right)$')
ax.set_ylabel(r'$f_{rig} \cdot L^{\beta/\nu}$')
for i in range(1,len(NP)):
    myColor = cmap((i+1)/len(NP))
    x = np.power(L[i],1./nu) * corrLen[i]
    y = f_rig_mean[i] * np.power(L[i],beta/nu)
    ax.plot(x, y, marker=symbols[i], color=myColor, linestyle='', label="NP = "+str(NP[i]))
ax.grid(which='Both', alpha=0.2)
ax.legend(fontsize=8)
fig.savefig(scalingDir+"/scaling3"+figFormat, bbox_inches="tight")



fig, ax = plt.subplots(1,1)
fig.suptitle(r'$\nu = \ $'+'{:.2f}'.format(nu)+r'$\qquad\gamma = \ $'+'{:.2f}'.format(gamma))
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'$max\left(\text{var}\left(F_{rig}\right)\right) \ / \ 10^{2}$')
mod = np.linspace(L[0], L[-1], 100)
for i in range(len(NP)):
    ax.semilogy(L[i], np.max(F_rig_var[i]), '-ok')
ax.plot(mod, np.power(mod, gamma/nu), label=r"$L^{\gamma/\nu}$")
ax.grid(which='Both', alpha=0.2)
ax.legend(fontsize=8)
fig.savefig(scalingDir+"/scaling4"+figFormat, bbox_inches="tight")



fig, ax = plt.subplots(1,1)
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'$\phi_{c}^{L}$')
y = np.zeros(len(NP))
for i in range(len(NP)):
    y[i] = phi[indSig][np.where(F_rig_var[i]==np.max(F_rig_var[i]))[0][0]]
ax.plot(L, y, '-ok')
ax.grid(which='Both', alpha=0.2)
fig.savefig(scalingDir+"/scaling5"+figFormat, bbox_inches="tight")



fig, ax = plt.subplots(1,1)
fig.suptitle(r'$\phi_{c}^{inf} = \ $'+str(phi_c_inf)+r'$\qquad\nu = \ $'+'{:.2f}'.format(nu))
ax.set_xlabel(r'$L^{-\nu}-L_{max}^{-\nu}$')
ax.set_ylabel(r'$\phi_{c}^{L}-\phi_{c}^{inf}$')
y = np.zeros(len(NP))
for i in range(len(NP)):
    y[i] = phi[indSig][np.where(F_rig_var[i]==np.max(F_rig_var[i]))[0][0]]
y = y[-1] - y
ax.plot(L**(-nu) - L[-1]**(-nu), y, '-ok')
ax.grid(which='Both', alpha=0.2)
fig.savefig(scalingDir+"/scaling6"+figFormat, bbox_inches="tight")





