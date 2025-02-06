import os
import glob
import myFunctions
import numpy as np


def filesGeneratorOneRun(NP, phi, Dir, t_SS, outputVar, makeMovies=False):

    # file_paths = glob.glob(os.path.join(dir, 'data_*.dat'))
    # basenames = [os.path.basename(file_path)[5:] for file_path in file_paths]

    dataFile = Dir + 'data_' + os.path.basename(glob.glob(Dir+'data_*.dat')[0]).removeprefix('data_')
    # dataFile = basenames[0]
    
    t,     dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, \
    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy  \
        = np.loadtxt(dataFile, skiprows=37).transpose()
    
    SSi = np.where(t==t_SS)[0][0]
        
    computeRigidClusters      = False
    computePrimeRigidClusters = False
    computeZZnet              = False
    computeRigPers            = False
    identifyFrictPartsIDs     = False
    computeFrictPers          = False
    computeMaxClusterSizeCorr = False
    computeSpatialCorr        = False
    makeSnapshots             = False
    
    # check for rigid clusters
    if not os.path.exists(Dir+'F_rig.txt'):
        computeRigidClusters = True
    else:
        with open(Dir+'F_rig.txt', 'r') as fp:
            for count, line in enumerate(fp):
                pass
        if count+1 != len(t):
            computeRigidClusters = True
    
    # check for rigid prime clusters
    # if not os.path.exists(Dir+'F_prime_rig.txt') or not os.path.exists(Dir+'rigPrime.txt'):
    #     computePrimeRigidClusters = True
    # else:
    #     with open(Dir+'F_prime_rig.txt', 'r') as fp:
    #         for count, line in enumerate(fp):
    #             pass
    #     if count != len(t):
    #         computePrimeRigidClusters = True
    
    # # check for Z and Znet
    # if not os.path.exists(Dir+'Z_Znet.txt'):
    #     computeZZnet = True
    # else:
    #     with open(Dir+'Z_Znet.txt', 'r') as fp:
    #         for count, line in enumerate(fp):
    #             pass
    #     if count != len(t):
    #         computeZZnet = True

    # # check for rigidity persistence
    # if not os.path.exists(Dir+'rigPers.txt'):
    #     computeRigPers = True
    # else:
    #     with open(Dir+'rigPers.txt', 'r') as fp:
    #         for count, line in enumerate(fp):
    #             pass
    #     if count != len(t)-SSi:
    #         computeRigPers = True

    # # check for frictional particles
    # if not os.path.exists(Dir+'frictPartsIDs.txt'):
    #     identifyFrictPartsIDs = True
    # else:
    #     with open(Dir+'frictPartsIDs.txt', 'r') as fp:
    #         for count, line in enumerate(fp):
    #             pass
    #     if count+1 != len(t):
    #         identifyFrictPartsIDs = True
    
    # # check for frictional persistence
    # if not os.path.exists(Dir+'frictPers.txt'):
    #     computeFrictPers = True
    # else:
    #     with open(Dir+'frictPers.txt', 'r') as fp:
    #         for count, line in enumerate(fp):
    #             pass
    #     if count != len(t)-SSi:
    #         computeFrictPers = True
            
    # # check for maximum cluster size autocorrelation
    # if not os.path.exists(Dir+'maxClusterSize_corr.txt'):
    #     computeMaxClusterSizeCorr = True
    # else:
    #     with open(Dir+'maxClusterSize_corr.txt', 'r') as fp:
    #         for count, line in enumerate(fp):
    #             pass
    #     if count != len(t)-SSi:
    #         computeMaxClusterSizeCorr = True
    
    # # check for spatial correlation (only one system size)
    # # if NP == 2000:
    # #     if not os.path.exists(Dir+'spatialCorr.txt'):
    # #         computeSpatialCorr = True
    # #     else:
    # #         with open(Dir+'spatialCorr.txt', 'r') as fp:
    # #             for count, line in enumerate(fp):
    # #                 pass
    # #         if count+1 != len(t)-SSi:
    # #             computeSpatialCorr = True
                
    # # check for snapshots (only specific cases)
    # if (makeMovies):
    #     if  not os.path.exists(Dir+     "snapshots")               or not os.path.exists(Dir+"some_snapshots") or \
    #         not os.path.exists(Dir+     "snapshots/clusters")      or not os.path.exists(Dir+     "snapshots/interactions") or not os.path.exists(Dir+     "snapshots/frictParts") or \
    #         not os.path.exists(Dir+"some_snapshots/clusters")      or not os.path.exists(Dir+"some_snapshots/interactions") or not os.path.exists(Dir+"some_snapshots/frictParts") or \
    #         not os.path.exists(Dir+     "snapshots/superposition") or not os.path.exists(Dir+"some_snapshots/superposition"):
    #             makeSnapshots = True
    #     else:
    #         if  len([name for name in os.listdir(Dir+"snapshots/clusters")      if os.path.isfile(os.path.join(Dir+"snapshots/clusters",      name))]) != len(t) or \
    #             len([name for name in os.listdir(Dir+"snapshots/interactions")  if os.path.isfile(os.path.join(Dir+"snapshots/interactions",  name))]) != len(t) or \
    #             len([name for name in os.listdir(Dir+"snapshots/frictParts")    if os.path.isfile(os.path.join(Dir+"snapshots/frictParts",    name))]) != len(t) or \
    #             len([name for name in os.listdir(Dir+"snapshots/superposition") if os.path.isfile(os.path.join(Dir+"snapshots/superposition", name))]) != len(t):
    #                 makeSnapshots = True
    
    # del dummy, t, dataFile
        
    # compute only what is missing
    
    if computeRigidClusters:
        print("  >> computing rigid clusters")
        myFunctions.myRigidClusters(Dir)
        
    if computePrimeRigidClusters:
        print("  >> computing prime rigid clusters")
        myFunctions.myPrimeRigidClusters(Dir)
           
    if computeZZnet:
        print("  >> computing Z and Znet")
        myFunctions.Z_Znet(Dir)

    if computeRigPers:
        print("  >> computing rigidity persistence")
        myFunctions.rigPers(Dir, t_SS, outputVar)
        
    if computeFrictPers:
        print("  >> computing frictional persistence")
        myFunctions.frictPers(Dir, t_SS, outputVar)
        
    if computeMaxClusterSizeCorr:
        print("  >> computing maximum cluster size autocorrelation")
        myFunctions.maxClusterSize_corr(Dir, t_SS, outputVar)

    if computeSpatialCorr:
        print("  >> computing spatial correlation")
        myFunctions.spatial_correlation(Dir, t_SS)
        
    if identifyFrictPartsIDs:
        print("  >> identifying frictional particles")
        myFunctions.frict_parts_IDs(Dir)
    
    if makeSnapshots:
        print("  >> making snapshots")
        myFunctions.make_snapshots(Dir)
