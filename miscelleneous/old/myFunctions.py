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





#%% MERGE COMMON

# function to merge sublists having common elements
# link: https://www.geeksforgeeks.org/python-merge-list-with-common-elements-in-a-list-of-lists/
def merge_common(lists):
    neigh = defaultdict(set)
    visited = set()
    for each in lists:
        for item in each:
            neigh[item].update(each)
    def comp(node, neigh = neigh, visited = visited, vis = visited.add):
        nodes = set([node])
        next_node = nodes.pop
        while nodes:
            node = next_node()
            vis(node)
            nodes |= neigh[node] - visited
            yield node
    for node in neigh:
        if node not in visited:
            yield sorted(comp(node))





#%% RIGID CLUSTERS (PEBBLE GAME)

def myRigidClusters(Dir):
    
    myRigidClusterProcessor.rigFileGenerator(Dir, Dir, False, True, True)
    
    # Read in rig_ files and write n_rigid.csv files
    rigFile = Dir + 'rig_' + os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    (rigidClusterSizes,numBonds,clusterIDs) = myRigidClusterProcessor.rigFileReader(rigFile, False, True)
    
    n_rigid = np.array([])
    for clusterLists in rigidClusterSizes:
        n_rigid = np.append(n_rigid, np.sum(np.array(clusterLists)))
    n_rigid = np.array([int(x) for x in n_rigid])
    
    np.savetxt(Dir+'F_rig.txt', np.transpose([n_rigid]), delimiter=' ', fmt='%f')





#%% PRIME RIGID CLUSTERS

def myPrimeRigidClusters(Dir):
    
    baseName = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile = Dir + 'data_' + baseName
    rigFile  = Dir + 'rig_'  + baseName
    intFile  = Dir + 'int_'  + baseName
    
    t,     dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
        = np.loadtxt(dataFile, skiprows=37).transpose()
    
    clusters      = []
    clustersSizes = []
    F_prime_rig   = np.zeros(len(t), dtype=int)
    
    int_skiprows     = 20
    int_Nlines       = 0
    morelines        = 0
    totNumOfClusters = 0
    
    for it in range(len(t)):
        
        int_skiprows += 7 + int_Nlines
        int_Nlines    = 0
        with open(intFile, 'r') as file:
            for i in range(int_skiprows):
                line = file.readline()
            while True:
                line = file.readline()
                if not line or line.split()[0] == '#':
                    break
                else:
                    int_Nlines += 1
                    
        ip, jp, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, contState, dummy, dummy, dummy, dummy, dummy, dummy = np.loadtxt(intFile, skiprows=int_skiprows, max_rows=int_Nlines).transpose()
        ip        = np.array(ip,        dtype=int)
        jp        = np.array(jp,        dtype=int)
        contState = np.array(contState, dtype=int)
        ip        = ip[contState==2]
        jp        = jp[contState==2]
        
        clustersSizes_it = [int(x) for x in np.frombuffer(np.loadtxt(rigFile, skiprows=it+1, max_rows=1))]
        if clustersSizes_it == [0]:
            numberOfClusters = 0
            morelines       += 1
        else:
            numberOfClusters = len(clustersSizes_it)
        
        clusters_it = []
        for i in range(numberOfClusters):
            clusters_it.append(np.unique([int(x) for x in np.loadtxt(rigFile, skiprows=(len(t)+2)*2+2+it+i+totNumOfClusters+morelines, max_rows=1, delimiter=',')]))
            if (len(clusters_it[i]) != clustersSizes_it[i]):
                sys.exit("ERROR: there is a problem with the clusters sizes")
        
        rigPartsIDs = [item for sublist in clusters_it for item in sublist]
       
        totNumOfClusters += numberOfClusters
        
        toBeRemoved = []
        for i in range(len(ip)):
            if ip[i] in rigPartsIDs and not jp[i] in rigPartsIDs:
                toBeRemoved.append(ip[i])
            elif jp[i] in rigPartsIDs and not ip[i] in rigPartsIDs:
                toBeRemoved.append(jp[i])
        toBeRemoved = np.unique(toBeRemoved)
        
        for i in range(len(toBeRemoved)):
            for j in range(numberOfClusters):
                if toBeRemoved[i] in clusters_it[j]:
                    indID = np.where(clusters_it[j]==toBeRemoved[i])[0][0]
                    clusters_it[j] = np.delete(clusters_it[j], indID)
        clusters_it = [ elem for elem in clusters_it if len(elem) > 0]
        
        clustersSizes_it = []
        if len(clusters_it) > 0:
            for i in range(len(clusters_it)):
                clustersSizes_it.append(len(clusters_it[i]))
            
        clustersSizes.append(clustersSizes_it)
        clusters.append(clusters_it)
        
        rigPrimePartsIDs = [item for sublist in clusters_it for item in sublist]
        F_prime_rig[it]  = len(rigPrimePartsIDs)
        
    rigPrimeFile = open(Dir+"rigPrime.txt", "w")
    rigPrimeFile.write("#Prime Rigid Clusters Sizes" + '\n')
    for it in range(len(t)):
        if len(clustersSizes[it]) == 0:
            rigPrimeFile.write("0   \n")
        else:
            for i in range(len(clustersSizes[it])):
                rigPrimeFile.write(str(clustersSizes[it][i]) + '   ')
            rigPrimeFile.write("\n")
    rigPrimeFile.write("\n")
    rigPrimeFile.write("#Prime Rigid Clusters IDs" + '\n')
    for it in range(len(t)):
        rigPrimeFile.write('#snapshot = ' + str(it) + '\n')
        if len(clustersSizes[it]) == 0:
            rigPrimeFile.write("0\n")
        else:
            for i in range(len(clusters[it])):
                for j in range(len(clusters[it][i])):
                    if j < len(clusters[it][i])-1:
                        rigPrimeFile.write(str(clusters[it][i][j]) + ',')
                    else:
                        rigPrimeFile.write(str(clusters[it][i][j]))
                rigPrimeFile.write("\n")
    rigPrimeFile.close()
    
    FPrimeFile = open(Dir+"F_prime_rig.txt", "w")
    FPrimeFile.write("t                F'_rig" + '\n')
    for it in range(len(t)):
        FPrimeFile.write('{:.4f}'.format(t[it]) + '      ' + str(F_prime_rig[it])  + '\n')
    FPrimeFile.close()





#%% Z & ZNET

def Z_Znet(Dir):
    
    #baseName = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    baseName = os.path.basename(glob.glob(Dir+'data_*.dat')[0])[5:]
    dataFile = Dir + 'data_' + baseName
    intFile  = Dir + 'int_'  + baseName
    
    t,     dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
        = np.loadtxt(dataFile, skiprows=37).transpose()
        
    NP = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    
    Z_Znet = np.zeros((len(t),4))
    
    int_skiprows = 20
    int_Nlines   = 0
    
    for it in range(len(t)):
        
        int_skiprows += 7 + int_Nlines
        int_Nlines    = 0
        with open(intFile, 'r') as file:
            for i in range(int_skiprows):
                line = file.readline()
            while True:
                line = file.readline()
                if not line or line.split()[0] == '#':
                    break
                else:
                    int_Nlines += 1
                    
        ip, jp, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, contState, dummy, dummy, dummy, dummy, dummy, dummy = np.loadtxt(intFile, skiprows=int_skiprows, max_rows=int_Nlines).transpose()
        
        ip           = np.array(ip,        dtype=int)
        jp           = np.array(jp,        dtype=int)
        contState    = np.array(contState, dtype=int)
        frictionCont = np.where(contState==2)[0]
        
        if frictionCont.size > 0:
        
            numContsPerPart = np.zeros(NP, dtype=int)
            for i in range(frictionCont.size):
                numContsPerPart[ip[frictionCont[i]]] += 1
                numContsPerPart[jp[frictionCont[i]]] += 1
                
            Z_Znet[it][0] = np.mean(numContsPerPart)
            Z_Znet[it][1] = np.std(numContsPerPart)
            Z_Znet[it][2] = np.mean(numContsPerPart[numContsPerPart!=0])
            Z_Znet[it][3] = np.std(numContsPerPart[numContsPerPart!=0])
    
    np.savetxt(Dir+"Z_Znet.txt", Z_Znet, delimiter='      ', fmt='%.9f', header='mean(Z)      std(Z)      mean(Znet)      std(Znet)')





#%% RIGIDITY PERSISTENCE

def rigPers(Dir, t_SS, outputVar):
    
    baseName = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile = Dir + 'data_' + baseName
    rigFile  = Dir + 'rig_'  + baseName
    
    # this function requires myRigidClusters to be previously run
    # let's check if it has been run, let's do it if not
    if not os.path.exists(rigFile):
        myRigidClusters(Dir)
        
    t,     gamma, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
        = np.loadtxt(dataFile, skiprows=37).transpose()
        
    NP = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    
    SSi = np.where(t==t_SS)[0][0]
    
    morelines        = 0
    totNumOfClusters = 0
    isInCluster      = np.zeros((len(t)-SSi,NP), dtype=bool)
    
    for it in range(len(t)):
        
        clustersSizes_it = [int(x) for x in np.frombuffer(np.loadtxt(rigFile, skiprows=it+1, max_rows=1))]
        
        if clustersSizes_it == [0]:
            morelines          += 1
            numberOfClusters_it = 0
        else:
            numberOfClusters_it = len(clustersSizes_it)
        
        rigPartsIDs_it = []
        for i in range(numberOfClusters_it):
            rigPartsIDs_it.append(np.unique([int(x) for x in np.loadtxt(rigFile, skiprows=(len(t)+2)*2+2+it+i+totNumOfClusters+morelines, max_rows=1, delimiter=',')]))
            if (len(rigPartsIDs_it[i]) != clustersSizes_it[i]):
                sys.exit("ERROR: there is a problem with the clusters sizes")
                
        totNumOfClusters += numberOfClusters_it
        
        if it >= SSi:
            flat_rigPartsIDs_it = [item for sublist in rigPartsIDs_it for item in sublist]
            for ip in flat_rigPartsIDs_it:
                isInCluster[it-SSi][ip] = True
                
    ntaus       = len(t)-SSi
    rigPers     = np.zeros(ntaus)
    corrProd    = np.zeros((ntaus,NP))
    uncorrProd1 = np.zeros((ntaus,NP))
    uncorrProd2 = np.zeros((ntaus,NP))
    for it1 in range(ntaus):
        for it2 in range(it1,ntaus):
            k               = it2 - it1
            corrProd[k]    += isInCluster[it1] * isInCluster[it2]
            uncorrProd1[k] += isInCluster[it1]
            uncorrProd2[k] += isInCluster[it2]
    for k in range(ntaus):
        corrProd[k]    /= (ntaus-k)
        uncorrProd1[k] /= (ntaus-k)
        uncorrProd2[k] /= (ntaus-k)
        rigPers[k]      = np.sum(corrProd[k] - uncorrProd1[k]*uncorrProd2[k])
        
    if outputVar == 't':
        delta   = t[1]-t[0]
        header  = 'Delta t       C'
    elif outputVar == 'gamma':
        delta   = gamma[1]-gamma[0]
        header  = 'Delta gamma       C'
    else:
        sys.exit("ERROR: there is a problem with outputVar")
        
    rigPersFile = open(Dir+"rigPers.txt", "w")
    rigPersFile.write(header + '\n')
    for k in range(ntaus):
        rigPersFile.write(str(round(delta*k,9)) + '      ' +
                          str(rigPers[k])       + '\n')
    rigPersFile.close()





#%% MAX CLUSTER SIZE TIME CORRELATION FUNCTION

def maxClusterSize_corr(Dir, t_SS, outputVar):
    
    baseName = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile = Dir + 'data_' + baseName
    rigFile  = Dir + 'rig_'  + baseName
    
    # this function requires myRigidClusters to be previously run
    # let's check if it has been run, let's do it if not
    if not os.path.exists(rigFile):
        myRigidClusters(Dir)
        
    t,     gamma, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
        = np.loadtxt(dataFile, skiprows=37).transpose()
        
    NP = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    
    SSi = np.where(t==t_SS)[0][0]
    
    maxClustersSize = np.zeros(len(t)-SSi, dtype=int)
    for it in range(SSi,len(t)):
        maxClustersSize[it-SSi] = np.max([int(x) for x in np.frombuffer(np.loadtxt(rigFile, skiprows=it+1, max_rows=1))])
        
    ntaus       = len(t)-SSi
    nMaxPers    = np.zeros(ntaus)
    corrProd    = np.zeros((ntaus,NP))
    uncorrProd1 = np.zeros((ntaus,NP))
    uncorrProd2 = np.zeros((ntaus,NP))
    for it1 in range(ntaus):
        for it2 in range(it1,ntaus):
            k               = it2 - it1
            corrProd[k]    += maxClustersSize[it1] * maxClustersSize[it2]
            uncorrProd1[k] += maxClustersSize[it1]
            uncorrProd2[k] += maxClustersSize[it2]
    for k in range(ntaus):
        corrProd[k]    /= (ntaus-k)
        uncorrProd1[k] /= (ntaus-k)
        uncorrProd2[k] /= (ntaus-k)
        nMaxPers[k]     = np.sum(corrProd[k] - uncorrProd1[k]*uncorrProd2[k])
        
    if outputVar == 't':
        delta   = t[1]-t[0]
        header  = 'Delta t       C'
    elif outputVar == 'gamma':
        delta   = gamma[1]-gamma[0]
        header  = 'Delta gamma       C'
    else:
        sys.exit("ERROR: there is a problem with outputVar")
        
    nMaxPersFile = open(Dir+"maxClusterSize_corr.txt", "w")
    nMaxPersFile.write(header + '\n')
    for k in range(ntaus):
        nMaxPersFile.write(str(round(delta*k,9)) + '      ' +
                           str(nMaxPers[k])      + '\n')
    nMaxPersFile.close()





#%% FRICTIONAL PERSISTENCE

def frictPers(Dir, t_SS, outputVar):
    
    baseName       = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile       = Dir + 'data_' + baseName
    frictPartsFile = Dir + 'frictPartsIDs.txt'
    
    # this function requires myRigidClusters to be previously run
    # let's check if it has been run, let's do it if not
    if not os.path.exists(frictPartsFile):
        frict_parts_IDs(Dir)
        
    t,     gamma, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
        = np.loadtxt(dataFile, skiprows=37).transpose()
        
    NP = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    
    SSi = np.where(t==t_SS)[0][0]
    
    isFrictional = np.zeros((len(t)-SSi,NP), dtype=bool)
    
    for it in range(SSi,len(t)):
        
        frictPartsIDs_it = np.genfromtxt(frictPartsFile, skip_header=it, max_rows=1)
        
        if np.any(np.isnan(frictPartsIDs_it)):
            frictPartsIDs_it = []
        else:
            if frictPartsIDs_it.size == 1:
                frictPartsIDs_it = [int(frictPartsIDs_it)]
            else:
                frictPartsIDs_it = list([int(x) for x in frictPartsIDs_it])
        
        for ip in frictPartsIDs_it:
            isFrictional[it-SSi][ip] = True
                
    ntaus       = len(t)-SSi
    frictPers   = np.zeros(ntaus)
    corrProd    = np.zeros((ntaus,NP))
    uncorrProd1 = np.zeros((ntaus,NP))
    uncorrProd2 = np.zeros((ntaus,NP))
    for it1 in range(ntaus):
        for it2 in range(it1,ntaus):
            k               = it2 - it1
            corrProd[k]    += isFrictional[it1] * isFrictional[it2]
            uncorrProd1[k] += isFrictional[it1]
            uncorrProd2[k] += isFrictional[it2]
    for k in range(ntaus):
        corrProd[k]    /= (ntaus-k)
        uncorrProd1[k] /= (ntaus-k)
        uncorrProd2[k] /= (ntaus-k)
        frictPers[k]    = np.sum(corrProd[k] - uncorrProd1[k]*uncorrProd2[k])
    
    if outputVar == 't':
        delta   = t[1]-t[0]
        header  = 'Delta t       C'
    elif outputVar == 'gamma':
        delta   = gamma[1]-gamma[0]
        header  = 'Delta gamma       C'
    else:
        sys.exit("ERROR: there is a problem with outputVar")
        
    frictPersFile = open(Dir+"maxClusterSize_corr.txt", "w")
    frictPersFile.write(header + '\n')
    for k in range(ntaus):
        frictPersFile.write(str(round(delta*k,9)) + '      ' +
                            str(frictPers[k])     + '\n')
    frictPersFile.close()





#%% SPATIAL CORRELATION

def spatial_correlation(Dir, t_SS):
    
    baseName   = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile   = Dir + 'data_' + baseName
    rigFile    = Dir + 'rig_'  + baseName
    parFile    = Dir + 'par_'  + baseName
    
    sigma      = int((re.sub(r'^.*?stress', '', baseName)).removesuffix('r_shear.dat'))
    configName = Dir + baseName.removesuffix('_nobrownian_2D_stress' + str(sigma) + 'r_shear.dat') + '.dat'
        
    # this function requires myRigidClusters to be previously run
    # let's check if it has been run, let's do it if not
    if not os.path.exists(rigFile):
        myRigidClusters(Dir)
        
    t,     dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
        = np.loadtxt(dataFile, skiprows=37).transpose()
        
    ndt = len(t)
        
    NP  = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    
    SSi = np.where(t==t_SS)[0][0]
    
    dummy, dummy, dummy, dummy, Lx, dummy, Lz, dummy, dummy, dummy, dummy = np.genfromtxt(configName, comments=None, skip_header=1, max_rows=1)
    
    dummy, dummy, dummy, a = np.loadtxt(configName, skiprows=2, max_rows=NP).transpose()
    
    morelines        = 0
    totNumOfClusters = 0
    isInCluster      = np.zeros((len(t)-SSi,NP), dtype=bool)
    
    for it in range(len(t)):
        
        clustersSizes_it = [int(x) for x in np.frombuffer(np.loadtxt(rigFile, skiprows=it+1, max_rows=1))]
        
        if clustersSizes_it == [0]:
            morelines          += 1
            numberOfClusters_it = 0
        else:
            numberOfClusters_it = len(clustersSizes_it)
        
        rigPartsIDs_it = []
        for i in range(numberOfClusters_it):
            rigPartsIDs_it.append(np.unique([int(x) for x in np.loadtxt(rigFile, skiprows=(len(t)+2)*2+2+it+i+totNumOfClusters+morelines, max_rows=1, delimiter=',')]))
            if (len(rigPartsIDs_it[i]) != clustersSizes_it[i]):
                sys.exit("ERROR: there is a problem with the clusters sizes")
                
        totNumOfClusters += numberOfClusters_it
        
        if it >= SSi:
            flat_rigPartsIDs_it = [item for sublist in rigPartsIDs_it for item in sublist]
            for ip in flat_rigPartsIDs_it:
                isInCluster[it-SSi][ip] = True
            
    dr          = 1    # max one decimal
    dtheta      = 10   # degrees
    
    minr        = np.min(a) * 2
    maxr        = round(0.5*np.sqrt(Lx**2+Lz**2)+dr/2,1)
    
    nr          = round((maxr-minr) / dr)
    ntheta      = round(360 / dtheta)
    
    r_bins      = np.linspace(minr+0.5*dr, maxr-0.5*dr, nr)
    theta_bins  = np.linspace(0.5*dtheta, 360-0.5*dtheta, ntheta)   # [deg]
    
    r_vec       = np.linspace(minr, maxr, nr+1)
    theta_vec   = np.linspace(0, 360, ntheta+1) * np.pi / 180   # [rad]
    
    r_vec[0]    = 0.0   # to allow overlapping
    
    corrProd    = np.zeros((NP,nr,ntheta))
    uncorrProd1 = np.zeros((NP,nr,ntheta))
    uncorrProd2 = np.zeros((NP,nr,ntheta))
    
    for it in range(SSi,ndt):

        print("   - time step " + str(it+1) + " out of " + str(ndt))
        
        dummy, dummy, xp, zp, dummy, dummy, dummy, dummy, dummy, dummy, dummy = np.loadtxt(parFile, skiprows=16+it*NP+(it+1)*7, max_rows=NP).transpose()
        
        for ip in range(NP):
            
            rx = xp - xp[ip]
            rz = zp - zp[ip]
            
            # periodicity
            rx[rx>+Lx/2] -= Lx/2
            rx[rx<-Lx/2] += Lx/2
            rz[rz>+Lz/2] -= Lz/2
            rz[rz<-Lz/2] += Lz/2
            
            r     = np.sqrt(rx**2 + rz**2)
            theta = np.arctan2(rz,rx) + np.pi   # arctan2 goes from -pi to pi
            
            for jp in range(NP):
                if jp != ip:
                    for kr in range(nr):
                        if r[jp] > r_vec[kr] and r[jp] <= r_vec[kr+1]:
                            r_index = kr
                            break
                    for kt in range(ntheta):
                        if theta[jp] > theta_vec[kt] and theta[jp] <= theta_vec[kt+1]:
                            t_index = kt
                            break
                    corrProd[ip][r_index][t_index]    += isInCluster[it][ip] * isInCluster[it][jp]
                    uncorrProd1[ip][r_index][t_index] += isInCluster[it][ip]
                    uncorrProd2[ip][r_index][t_index] += isInCluster[it][jp]
                    del r_index, t_index
    
    corrProd    /= ndt
    uncorrProd1 /= ndt
    uncorrProd2 /= ndt
        
    spatialCorr  = np.sum(corrProd - uncorrProd1*uncorrProd2, axis=0)
    
    spatialCorrFile = open(Dir+"spatialCorr.txt", "w")
    spatialCorrFile.write('r       theta [deg]      C(r,theta)' + '\n')
    for kr in range(len(nr)):
        for kt in range(len(ntheta)):
            spatialCorrFile.write(r_bins[kr]          + '      ' +
                                  theta_bins[kt]      + '      ' +
                                  spatialCorr[kr][kt] + '\n')
    spatialCorrFile.close()
    




#%% FRICTIONAL PARTICLES

def frict_parts_IDs(Dir):

    baseName = os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    dataFile = Dir + 'data_' + baseName
    intFile  = Dir + 'int_'  + baseName
    
    t,     dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
        = np.loadtxt(dataFile, skiprows=37).transpose()
        
    NP = int(np.genfromtxt(dataFile, skip_header=1, max_rows=1, comments='!')[2])
    
    int_skiprows  = 20
    int_Nlines    = 0
    frictPartsIDs = []
    
    for it in range(len(t)):
        
        int_skiprows += 7 + int_Nlines
        int_Nlines    = 0
        with open(intFile, 'r') as file:
            for i in range(int_skiprows):
                line = file.readline()
            while True:
                line = file.readline()
                if not line or line.split()[0] == '#':
                    break
                else:
                    int_Nlines += 1
                    
        ip, jp, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, contState, dummy, dummy, dummy, dummy, dummy, dummy = np.loadtxt(intFile, skiprows=int_skiprows, max_rows=int_Nlines).transpose()
        
        ip           = np.array(ip,        dtype=int)
        jp           = np.array(jp,        dtype=int)
        contState    = np.array(contState, dtype=int)
        frictContInd = np.where(contState==2)[0]
        
        numContsPerPart = np.zeros(NP, dtype=int)
        for i in range(frictContInd.size):
            numContsPerPart[ip[frictContInd[i]]] += 1
            numContsPerPart[jp[frictContInd[i]]] += 1
        
        frictPartsIDs.append(np.where(numContsPerPart>=3)[0])
        
    frictPartsIDsFile = open(Dir+"frictPartsIDs.txt", "w")
    for it in range(len(t)):
        if len(frictPartsIDs[it]) == 0:
            frictPartsIDsFile.write("none")
        else:
            for ip in frictPartsIDs[it]:
                frictPartsIDsFile.write(str(ip) + "   ")
        frictPartsIDsFile.write('\n')
    frictPartsIDsFile.close()





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




    
