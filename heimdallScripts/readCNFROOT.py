import numpy as np
import uproot

repeatSample = 100
EPSILON = 1e-4

def getSterileData(base_path : str):

    dataSaveTrain = base_path + "dataTrain"
    dataSaveTest = base_path + "dataTest"

    paramsSaveTrain = base_path + "paramsTrain"
    paramsSaveTest = base_path + "paramsTest"

    dataSaveMean = base_path + "dataMean"
    dataSaveStd = base_path + "dataStd"

    paramsSaveMean = base_path + "paramsMean"
    paramsSaveStd = base_path + "paramsStd"


    file = uproot.open(base_path + "CNFData.root")
    tree = file["Experimental_Data_Tree"]
    branches = tree.arrays()

    print(tree.keys())
    data = np.array(branches["data"])
    params = np.array(branches["params"])

    print(f"Read Data \n data Shape: {data.shape} \n params Shape : {params.shape}")
    
    params_unique = params[::repeatSample,:]
    params_mean = np.mean(params_unique, axis = 0)
    paramsStd = np.std(params_unique,axis = 0,ddof=0)
    params = (params - params_mean)/(paramsStd+EPSILON)

    print("Standardized Thetas")

    data_AT = 2 * np.sqrt(data + 3/8)
    data_AT_mean = np.mean(data_AT , axis = 0)
    data_AT_std = np.std(data_AT, axis=0, ddof=0)
    data_AT = (data_AT - data_AT_mean) / (data_AT_std + EPSILON)

    print("Standardized Data")


    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(params)

    trainRatio = int(0.8*len(params))
    param_train = params[:trainRatio,:]
    param_test = params[trainRatio:,:]
    data_train = data[:trainRatio,:]
    data_test = data[trainRatio:,:]

    print("Shuffled Data and Split 80/20")

    np.save(dataSaveTrain,data_train)
    np.save(dataSaveTest,data_test)

    np.save(paramsSaveTrain,param_train)
    np.save(paramsSaveTest,param_test)

    np.save(dataSaveMean,data_AT_mean)
    np.save(dataSaveStd,data_AT_std)

    np.save(paramsSaveMean,params_mean)
    np.save(paramsSaveStd,paramsStd)

    print("Complete")

       

if __name__ == "__main__":
    getSterileData("data/")

