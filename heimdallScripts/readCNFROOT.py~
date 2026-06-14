import numpy as np
import uproot
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from glob import glob
from tqdm import tqdm
import os 

repeatSample = 150
uniqueSample = 100000
EPSILON = 1e-3

def plotBinnedData(data : np.array, path : str):
    print("OLD function do not use")
    binList = [22,22,22,22,14,14,13,13,6]

    index = 0
    idx = 0

    for i in binList:
        slicedData = data[index:i+index]
        plt.bar(range(index,index+i),slicedData,color="C"+str(idx),edgecolor="black")         
        idx += 1
        index += i
        #plt.clf()

    plt.savefig(path+".png")


def calculate_std(base_path : str):

    files = sorted(glob(f"{base_path}training/*.npz"))

    theta_sum = 0
    num_samples = 0
    theta_sq_sum = 0

    data_sum = 0
    num_data = 0
    data_sq_sum = 0
   
    
    for file_name in tqdm(files):
        x = np.load(file_name)
        theta = x["params"]
        theta_unique = theta[::repeatSample,:]

        theta_sum += theta_unique.sum(axis=0)
        num_samples += len(theta_unique)
        theta_sq_sum += np.square(theta_unique).sum(axis=0)
        
        #print(theta[np.abs(theta) > 3.5])
        data = x["data"]
        data_AT = 2 * np.sqrt(data + 3/8)    


        num_data += len(data_AT)
        data_sum += data_AT.sum(axis=0)
        data_sq_sum +=  np.square(data_AT).sum(axis=0)


            
    theta_mean = theta_sum/num_samples
    theta_std = np.sqrt((theta_sq_sum/num_samples) - np.square(theta_mean))

    data_mean = data_sum/num_data
    data_std = np.sqrt((data_sq_sum/num_data) - np.square(data_mean))
    
    np.save(f"{base_path}processed/stats/theta_mean",theta_mean)
    np.save(f"{base_path}processed/stats/theta_std",theta_std)

    np.save(f"{base_path}processed/stats/data_mean",data_mean)
    np.save(f"{base_path}processed/stats/data_std",data_std)

    
    return


def applyStd(base_path : str, apply_site : str): #apply_site is either training or testing

    files = sorted(glob(f"{base_path}{apply_site}/*.npz"))

    theta_mean = np.load(f"{base_path}processed/stats/theta_mean.npy")
    theta_std = np.load(f"{base_path}processed/stats/theta_std.npy")
    
    data_mean = np.load(f"{base_path}processed/stats/data_mean.npy")
    data_std = np.load(f"{base_path}processed/stats/data_std.npy") 


    data_output_path = f"{base_path}processed/{apply_site}"
    Path(data_output_path).mkdir(parents=True, exist_ok=True)

    for file_name in tqdm(files):
        x = np.load(file_name)
        theta = x["params"]
        theta -= theta_mean
        theta /= (theta_std + EPSILON)
    
        data_AT = 2 * np.sqrt(x["data"] + 3/8)

        data_AT -= data_mean
        data_AT /= (data_std + EPSILON)

        data_AT = data_AT.astype(np.float32, copy=False)
        theta = theta.astype(np.float32, copy=False)

        if apply_site == "training":
            rng_state = np.random.get_state()
            np.random.shuffle(data_AT)
            np.random.set_state(rng_state)
            np.random.shuffle(theta)       
  
        
        split_theta = np.array_split(theta,5)
        split_data = np.array_split(data_AT,5)

        for i in range(len(split_theta)):
            out_file_t = Path(data_output_path) / (Path(file_name).stem + f"_theta_{i}")
            out_file_d = Path(data_output_path) / (Path(file_name).stem + f"_data_{i}")    
            np.save(out_file_t, split_theta[i])
            np.save(out_file_d, split_data[i]) 

        os.remove(file_name)

    return
 

def getSterileData(base_path : str):
 
    calculate_std(base_path)

    print("Calculated Standardizations")

    applyStd(base_path, "training")

    print("Standardized and Randomized Training Data")    

    applyStd(base_path, "testing")

    print("Standardized Testing Data. Complete")


def unpacknpz(base_path : str, handle : str):
    data_path = f"{base_path}processed/{handle}/"

    files = sorted(glob(f"{data_path}*.npz"))

    for file_name in tqdm(files):
        x = np.load(file_name)
        out_file_t = Path(data_path) / ("theta_"+ Path(file_name).stem)
        out_file_d = Path(data_path) / ("data_"+ Path(file_name).stem)
        
        np.save(out_file_t, x["params"])
        np.save(out_file_d, x["data"])

        os.remove(file_name)


       

if __name__ == "__main__":
    base_path = "data/"
    getSterileData(base_path)
    #unpacknpz(base_path,"training")
    #unpacknpz(base_path,"testing")

