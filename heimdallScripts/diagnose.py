import numpy as np
from scipy.spatial import KDTree, cKDTree
from glob import glob
import consts
import random
from tqdm import tqdm

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

    print(theta_array.shape)
    print(query_array.shape)

    tree = cKDTree(theta_array)

    for min_freq in freqs:
        dd , _ = tree.query(query_array, k=min_freq+1)
        R_array = dd[:,-1]
        R_stds.append(np.std(R_array, ddof=0))
        R_means.append(np.mean(R_array))

    
    R_stds = np.array(R_stds)
    R_means = np.array(R_means)

    smallest_idx = np.argmin(R_stds / R_means)

    return R_means[smallest_idx] , freqs[smallest_idx]





def find_holes_in_data():
        
    undersamples = []

    for path in tqdm(theta_paths[:5]):
        theta_array = np.load(path)
        arr_list = np.array_split(theta_array,30)
        tree = KDTree(theta_array) 
       

        for arr in arr_list:
            query_tree = KDTree(arr)
            n_idx = query_tree.query_ball_tree(tree,R)
            n_lens = np.fromiter((len(indices) for indices in n_idx), dtype=np.int64, count=len(arr))
            query_idx = np.flatnonzero(n_lens < 1001)
            undersamples.append(arr[query_idx])
            #n_lens = tree.query_ball_point(arr, R, return_sorted=False, return_length=True)
            #query_idx = np.where(n_lens < 1001)[0]
            print(len(query_idx))
            #undersamples.append(arr[query_idx])               
    
    
    return



if __name__ == "__main__":
    freqs = [10,15,20,25,30,35,40,45,50]
    num_random_files = 50
    num_random_rows = 200
    
    print(estimate_params(freqs,num_random_files,num_random_rows))

    #find_holes_in_data()
