import numpy as np
import uproot
import matplotlib.pyplot as plt

repeatSample = 10
uniqueSample = 100000
EPSILON = 1e-3

def plotBinnedData(data : np.array, path : str):
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


def standardize(theta : np.array, data : np.array):
    theta_unique = theta[::repeatSample,:]
    theta_mean = np.mean(theta_unique, axis = 0)
    theta_std = np.std(theta_unique,axis = 0,ddof=0)
    theta = (theta - theta_mean)/(theta_std + EPSILON)

    #print(theta[np.abs(theta) > 3.5])

    binList = [22,22,22,22,14,14,13,13,6]
    data_AT = 2 * np.sqrt(data + 3/8)

    slices = []
    index = 0

    meanSlices = []
    stdSlices = []
    
    for i in binList:
        slicedData = data_AT[:,index:i+index]
        index += i
        slicedMean = np.mean(slicedData , axis = 0)
        slicedStd = np.std(slicedData, axis=0, ddof=0)
        meanSlices.append(slicedMean)
        stdSlices.append(slicedStd)
        slices.append((slicedData - slicedMean) / (slicedStd + EPSILON))

    data_AT = np.concatenate(slices,axis=1)

    return theta , theta_mean, theta_std, data_AT , meanSlices, stdSlices


def applyStd(theta : np.array, theta_mean : np.array, 
             theta_std : np.array, data : np.array, 
             meanSlices : list, stdSlices : list):

    theta = (theta - theta_mean)/(theta_std + EPSILON)

    binList = [22,22,22,22,14,14,13,13,6]
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

    return theta, data_AT

    



def getSterileData(base_path : str):

    dataSaveTrain = base_path + "dataTrain"
    dataSaveTest = base_path + "dataTest"

    paramsSaveTrain = base_path + "paramsTrain"
    paramsSaveTest = base_path + "paramsTest" 

    paramsSaveMean = base_path + "paramsMean"
    paramsSaveStd = base_path + "paramsStd"


    file = uproot.open(base_path + "CNFData_0_100000.root")
    tree = file["Experimental_Data_Tree"]
    branches = tree.arrays()

    print(tree.keys())
    data = np.array(branches["data"])
    params = np.array(branches["params"])

    #plotBinnedData(data[-1],"plots/exampleDataSterile")
    print(f"Read Data \n data Shape: {data.shape} \n params Shape : {params.shape}")

    testTrainRatio = int(repeatSample * (0.8*uniqueSample))
    param_train = params[:testTrainRatio,:]
    param_test = params[testTrainRatio:,:]
    data_train = data[:testTrainRatio,:]
    data_test = data[testTrainRatio:,:]

    print("Split training and testing data 80/20")

    
    theta_train,theta_mean,theta_std,data_train, meanSlices, stdSlices = standardize(param_train, data_train)

    print("Standardized Train")

    theta_test,data_test = applyStd(param_test, theta_mean, theta_std, data_test, meanSlices, stdSlices)  

    print("Standardized Test")    

    rng_state = np.random.get_state()
    np.random.shuffle(data_train)
    np.random.set_state(rng_state)
    np.random.shuffle(theta_train) 

    print("Randomized Training Data")
 
    np.save(dataSaveTrain,data_train)
    np.save(dataSaveTest,data_test)

    np.save(paramsSaveTrain,theta_train)
    np.save(paramsSaveTest,theta_test)    

    np.save(paramsSaveMean,theta_mean)
    np.save(paramsSaveStd,theta_std)

    print("Complete")

    #print(param_test)
    #print(data_test)

       

if __name__ == "__main__":
    getSterileData("data/")

