import numpy as np
import uproot
from pathlib import Path
from glob import glob
from tqdm import tqdm
import os 
import consts

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
        theta_unique = theta[::consts.repeatSize,:]

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
    
    np.save(consts.theta_mean_path,theta_mean)
    np.save(consts.theta_std_path,theta_std)

    np.save(consts.data_mean_path,data_mean)
    np.save(consts.data_std_path,data_std)

    
    return


def applyStdMain(base_path : str, apply_site : str): #apply_site is either training or testing

    files = sorted(glob(f"{base_path}{apply_site}/*.npz"))

    theta_mean = np.load(consts.theta_mean_path)
    theta_std = np.load(consts.theta_std_path)
    
    data_mean = np.load(consts.data_mean_path)
    data_std = np.load(consts.data_std_path) 


    data_output_path = f"{base_path}processed/{apply_site}"
    Path(data_output_path).mkdir(parents=True, exist_ok=True)

    for file_name in tqdm(files):
        x = np.load(file_name)
        theta = x["params"]
        theta -= theta_mean
        theta /= (theta_std + consts.EPSILON)

        #print(np.percentile(np.abs(theta),99.9,axis=0))
    
        data_AT = 2 * np.sqrt(x["data"] + 3/8)

        data_AT -= data_mean
        data_AT /= (data_std + consts.EPSILON)

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

        #os.remove(file_name) #TODO: Remove if space limited, moving data to lazy
        
    return


def applyStd(data): #overload for validation
  
    data_mean = np.load(consts.data_mean_path)
    data_std = np.load(consts.data_std_path) 

    data_AT = 2 * np.sqrt(data + 3/8)
    data_AT -= data_mean
    data_AT /= (data_std + consts.EPSILON)

    data_AT = data_AT.astype(np.float32, copy=False)
    
    return data_AT



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


def getSterileData(base_path : str):
    print(base_path)
 
    #calculate_std(base_path)

    print("Calculated Standardizations")

    applyStdMain(base_path, "training")

    print("Standardized and Randomized Training Data")    

    applyStdMain(base_path, "testing")

    print("Standardized Testing Data. Complete")
       

if __name__ == "__main__":
    base_path = consts.base_path
    getSterileData(base_path)
    #unpacknpz(base_path,"training")
    #unpacknpz(base_path,"testing")

