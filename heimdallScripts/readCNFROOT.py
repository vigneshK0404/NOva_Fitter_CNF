import numpy as np
import uproot

repeatSample = 100
uniqueSample = 10000
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

    binList = [22,22,22,22,14,14,13,13,6]
    data_AT = 2 * np.sqrt(data + 3/8)

    slices = []
    index = 0
    
    for i in binList:
        slicedData = data_AT[:,index:i+index]
        index += i
        slicedMean = np.mean(slicedData , axis = 0)
        slicedStd = np.std(slicedData, axis=0, ddof=0)
        slices.append((slicedData - slicedMean) / (slicedStd + EPSILON))

    data_AT = np.concatenate(slices,axis=1)

    print("Standardized Data")

    testTrainRatio = int(repeatSample * (0.8*uniqueSample))
    param_train = params[:testTrainRatio,:]
    param_test = params[testTrainRatio:,:]
    data_train = data_AT[:testTrainRatio,:]
    data_test = data_AT[testTrainRatio:,:]

    print("Split training and testing data 80/20")


    rng_state = np.random.get_state()
    np.random.shuffle(data_train)
    np.random.set_state(rng_state)
    np.random.shuffle(param_train) 

    print("Randomized Training Data")
 
    np.save(dataSaveTrain,data_train)
    np.save(dataSaveTest,data_test)

    np.save(paramsSaveTrain,param_train)
    np.save(paramsSaveTest,param_test)    

    np.save(paramsSaveMean,params_mean)
    np.save(paramsSaveStd,paramsStd)

    print("Complete")

    print(param_test)
    print(data_test)

       

if __name__ == "__main__":
    getSterileData("data/")

