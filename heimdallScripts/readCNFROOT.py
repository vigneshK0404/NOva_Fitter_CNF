import numpy as np
import uproot
import matplotlib.pyplot as plt
import pickle
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

    files = sorted(glob(f"{base_path}training/"))
    theta_sum = 0
    num_samples = 0
    theta_vars = 0
    populus = len(files)
    num_data = 0
    meanSlices = []
    stdSlices = []
    binList = [22,22,22,22,14,14,13,13,6]


    for file_name in files:
        x = np.load(file_name)
        theta = x["params"]
        theta_unique = theta[::repeatSample,:]

        theta_sum += theta_unique.sum(axis=0)
        num_samples += len(theta_unique)
        theta_vars += np.var(theta_unique,axis = 0,ddof=1) * (len(theta_unique)-1)
        
        #print(theta[np.abs(theta) > 3.5])
        data = x["data"]
        data_AT = 2 * np.sqrt(data + 3/8)

        index = 0                
        sumSlices = [0,0,0,0,0,0,0,0,0]
        varSlices = [0,0,0,0,0,0,0,0,0]
        numSlices = [0,0,0,0,0,0,0,0,0]
        num_data += len(data_AT)

        idx = 0

        for i in binList:
            slicedData = data_AT[:,index:i+index]
            index += i
            sumSlices[idx] += np.sum(slicedData,axis=0)
            varSlices[idx] += np.var(slicedData,axis-0,ddof=1) * (len(data_AT) - 1)
            numSlices[idx] += len(data_AT)
            idx += 1
    
    theta_mean = theta_sum/num_samples
    theta_std = np.sqrt(theta_vars/(num_samples-populus))

    for i in range(len(sumSlices)):
        meanSlices.append(sumSlices[i]/num_data)
        stdSlices.append(np.sqrt(varSlices[i]/(numSlices[i] - populus)))
    
    with open(f"{base_path}processed/slice_stats.pkl", "wb") as f:
        pickle.dump({
            "meanSlices": meanSlices,
            "stdSlices": stdSlices
        }, f)

    np.save(f"{base_path}processed/theta_mean",theta_mean)
    np.save(f"{base_path}processed/theta_std",theta_std)
    
    return


def applyStd(base_path : str, apply_site : str): #apply_site is either training or testing

    files = sorted(glob(f"{base_path}{apply_site}/"))

    theta_mean = np.load(f"{base_path}processed/theta_mean.npy")
    theta_std = np.load(f"{base_path}processed/theta_std.npy")
    binList = [22,22,22,22,14,14,13,13,6]

    with open("slice_stats.pkl", "rb") as f:
        data = pickle.load(f)

    meanSlices = data["meanSlices"]
    stdSlices = data["stdSlices"]


    for file_name in files:
        x = np.load(file_name)
        theta = x["params"]
        theta = (theta - theta_mean)/(theta_std + EPSILON)
    
        data = x["data"]
        data_AT = 2 * np.sqrt(data + 3/8)

        slices = []
        index = 0
        idx = 0 

        for i in binList:
            slicedData = data_AT[:,index:i+index]
            index += i      
            slices.append((slicedData - meanSlices[idx]) / (stdSlices[idx] + EPSILON))
            idx += 1

        data_AT = np.concatenate(slices,axis=1)

        if apply_site == "training":
            rng_state = np.random.get_state()
            np.random.shuffle(data_AT)
            np.random.set_state(rng_state)
            np.random.shuffle(theta)            

        data_output_path = data_base_path+"processed/"
        os.makedirs(data_output_path,exist_ok=True)
        np.savez(f"{data_output_path}{file_name}", data = data_AT, params = theta)

    return
 

def getSterileData(base_path : str):
 
    calculate_std(base_path)

    print("Calculated Standardizations")

    applyStd(base_path, "training")

    print("Standardized and Randomized Training Data")    

    applyStd(base_path, "testing")

    print("Standardized Testing Data. Complete")
       

if __name__ == "__main__":
    base_path = "/home/vigneshwar/pythonEnvs/CNF/data/"
    getSterileData(base_path)

