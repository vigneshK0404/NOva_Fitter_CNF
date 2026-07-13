import numpy as np
from scipy.spatial import KDTree, cKDTree
import math
from glob import glob
import consts
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
#import ROOT

"""
Figure out sampling method
If multivariate uniform sphere, then figure out the standard deviations for all the points
if multivariate gaussian, then figure out the covariance matrix
"""


def estimate_params(freqs : list, num_epochs : int, num_random_rows : int , num_combine=5):
    theta_paths = sorted(glob(consts.theta_path))
    random_files = random.sample(theta_paths, num_epochs * num_combine) 

    chunks = [random_files[i : i + num_combine] for i in range(0, len(random_files), num_combine)]

    R_means = []
    R_stds = []
    
    R_dict = {}
    for freq in freqs:
        R_dict[freq] = []

    for chunk in tqdm(chunks):

        theta_list = []
        query_list = []

        for path in chunk :
            theta_array = np.load(path)[::consts.repeatSize]
            random_rows = np.array(random.sample(range(0,len(theta_array)),num_random_rows))
            query_points = theta_array[random_rows]

            theta_list.append(theta_array)
            query_list.append(query_points)

        
        theta_array = np.vstack(theta_list)
        query_array = np.vstack(query_list)

        #print(f"theta_array : {theta_array.shape}")
        #print(f"query_array : {query_array.shape}")

        tree = cKDTree(theta_array)

        for min_freq in freqs:
            dd , _ = tree.query(query_array, k=min_freq)
            R_dict[min_freq].append(dd[:,-1])



    for min_freq in freqs:
        R_dict[min_freq] = np.array(R_dict[min_freq]).flatten()
        R_stds.append(np.std(R_dict[min_freq], ddof=0))
        R_means.append(np.mean(R_dict[min_freq]))

        
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


def check_holes(outliers : np.array, root_name : str, file_name : str, thetaMean : np.array, thetaStd : np.array):

    """
    with ROOT.TFile.Open(root_name) as f:
        h = f.Get(file_name)
        calcVals = np.array([h.GetBinContent(5),math.log10(h.GetBinContent(3)),
                    math.log10(h.GetBinContent(1)),pow(math.sin(h.GetBinContent(2)),2),
                    h.GetBinContent(4),h.GetBinContent(6)])
    """

    true_best_val = np.array([0.370105,math.log10(0.0203667),math.log10(0.0384742),pow(math.sin(0.884828),2),1.21022,0.00253644],dtype=np.float32)
    true_best_val -= thetaMean
    true_best_val /= (thetaStd + consts.EPSILON)

    outlier_tree = KDTree(outliers)
    dd , _ = outlier_tree.query(true_best_val)
    print(dd)
    
    return dd
    



def find_holes_in_data(thetaMean : np.array, thetaStd : np.array):

    true_best_val = np.array([0.370105,math.log10(0.0203667),math.log10(0.0384742),pow(math.sin(0.884828),2),1.21022,0.00253644],dtype=np.float32)
    true_best_val -= thetaMean
    true_best_val /= (thetaStd + consts.EPSILON)
    
    eps, min_freq = estimate_params(freqs=list(range(10,150)), num_epochs = 10, num_random_rows = 200)
    print(f"eps : {eps}")
    print(f"min_freq : {min_freq}")

    theta_paths = sorted(glob(consts.theta_path))
    outliers = []

    theta_list = []
    
    good_chunks = 0
    bad_chunks = 0

    for idx in tqdm(range(len(theta_paths))):
        path = theta_paths[idx]
        theta_list.append(np.load(path)[::consts.repeatSize])

        if len(theta_list) == 5 :
            theta_array = np.concatenate(theta_list , axis=0)            
            db = DBSCAN(eps=eps, min_samples=min_freq).fit(theta_array)
            mask = db.labels_ == -1
            outliers.append(theta_array[mask])

            theta_list = []

            tmp_tree = KDTree(theta_array)
            dists , _ = tmp_tree.query(true_best_val, k=min_freq)

            dist = dists[-1]
            if dist < eps:
                good_chunks += 1
            else:
                bad_chunks += 1



    if theta_list:
        theta_array = np.concatenate(theta_list , axis=0)            
        db = DBSCAN(eps=eps, min_samples=min_freq).fit(theta_array)
        mask = db.labels_ == -1
        outliers.append(theta_array[mask])
        


    outliers = np.concatenate(outliers, axis=0)
    print(f"First Pass : {outliers.shape}")

    db_outliers = DBSCAN(eps=eps, min_samples=min_freq).fit(outliers)
    mask = db_outliers.labels_ == -1

    outliers_reduced = outliers[mask]
    print(f"Second Pass : {outliers_reduced.shape}")


    outliers_bool = np.zeros(len(outliers_reduced), dtype=bool)
    outliers_tree = KDTree(outliers_reduced)
    outliers_final = []

    while np.any(outliers_bool == False):
        query_point = outliers_reduced[np.argmin(outliers_bool)]
        neighbor_idx = outliers_tree.query_ball_point(query_point, eps, return_sorted=False)
        outliers_bool[neighbor_idx] = True

        outliers_final.append(query_point)

    
    outliers_final = np.vstack(outliers_final)

    #outliers_final *= (thetaStd + consts.EPSILON)
    #outliers_final += thetaMean


    print(f"good_chunks : {good_chunks}")
    print(f"bad_chunks : {bad_chunks}")

    return outliers_final , eps



if __name__ == "__main__":
    thetaMean = np.load(consts.theta_mean_path)
    thetaStd = np.load(consts.theta_std_path) 
    
    optSpace = "th24vsdm41";
    opt = "";
    optSysts = "all";
    optSamples = "numusel_ncres30sel_nuonesel_fhc_rhc_neardet_fardet";
    exp_num = 221;
    

    stem = optSpace + "_" + optSamples + "_" + str(exp_num) + "_" + optSysts;
    root_name = "data/minimum_" + stem + ".root";
    file_name = "minimum/" + stem + "/vals";




    outliers , eps = find_holes_in_data(thetaMean,thetaStd)
    dd = check_holes(outliers, root_name, file_name, thetaMean, thetaStd )

    print(outliers)
    print(f"dd,eps : {dd},{eps}")
    print(f"Final shape : {outliers.shape}")
