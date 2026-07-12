import numpy as np
from scipy.spatial import KDTree, cKDTree
from glob import glob
import consts
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

"""
Find ideal eps and min_freq, plotting or ML technique may be required
Use for DBScan and get outliers
Cluster over all files
Pick representative points
Figure out Sampling method
"""

def estimate_params(freqs : list, num_random_files : int, num_random_rows : int):
    theta_paths = sorted(glob(consts.theta_path))
    random_files = random.sample(theta_paths, num_random_files) 


    R_means = []
    R_stds = []    
    theta_list = []
    query_list = []

    for path in tqdm(random_files) :
        theta_array = np.load(path)[::consts.repeatSize]
        random_rows = np.array(random.sample(range(0,len(theta_array)),num_random_rows))
        query_points = theta_array[random_rows]

        theta_list.append(theta_array)
        query_list.append(query_points)

    
    theta_array = np.vstack(theta_list)
    query_array = np.vstack(query_list)

    print(f"theta_array : {theta_array.shape}")
    print(f"query_array : {query_array.shape}")

    tree = cKDTree(theta_array)

    for min_freq in tqdm(freqs):
        dd , _ = tree.query(query_array, k=min_freq+1)
        R_array = dd[:,-1]
        R_stds.append(np.std(R_array, ddof=0))
        R_means.append(np.mean(R_array))

    
    R_stds = np.array(R_stds)
    R_means = np.array(R_means)
    R_score = R_stds / R_means

    smallest_idx = np.argmin(R_score)

    plt.plot(freqs,R_score)
    plt.savefig("plots/R_score.png")
    plt.clf()

    plt.plot(freqs,R_means)
    plt.savefig("plots/R_means.png")
    plt.clf()

    plt.plot(freqs,R_stds)
    plt.savefig("plots/R_stds.png")
    plt.clf()

    return R_means[smallest_idx] , freqs[smallest_idx]





def find_holes_in_data(thetaMean : np.array, thetaStd : np.array):
    
    eps = 0.4488
    min_freq = 72

    theta_paths = sorted(glob(consts.theta_path))
    outliers = []

    theta_list = []

    for idx in tqdm(range(len(theta_paths))):
        path = theta_paths[idx]
        theta_list.append(np.load(path)[::consts.repeatSize])

        if idx % 5 == 0 and idx != 0 :
            theta_array = np.concatenate(theta_list , axis=0)            
            db = DBSCAN(eps=eps, min_samples=min_freq).fit(theta_array)
            mask = db.labels_ == -1
            outliers.append(theta_array[mask])

            theta_list = []

    if theta_list:
        theta_array = np.concatenate(theta_list , axis=0)            
        db = DBSCAN(eps=eps, min_samples=min_freq).fit(theta_array)
        mask = db.labels_ == -1
        outliers.append(theta_array[mask])
        


    outliers = np.concatenate(outliers, axis=0)
    print(f"outliers : {outliers.shape}")

    db_outliers = DBSCAN(eps=eps, min_samples=min_freq).fit(outliers)
    mask = db_outliers.labels_ == -1

    outliers_final = outliers[mask]
    

    outliers_final *= (thetaStd + consts.EPSILON)
    outliers_final += thetaMean

    return outliers_final



if __name__ == "__main__":
    #freqs = list(range(10,150))
    #num_random_files = 50
    #num_random_rows = 200
    
    #print(estimate_params(freqs,num_random_files,num_random_rows))

    thetaMean = np.load(consts.theta_mean_path)
    thetaStd = np.load(consts.theta_std_path) 

    outliers = find_holes_in_data(thetaMean,thetaStd)

    print(outliers)
    print(outliers.shape)
