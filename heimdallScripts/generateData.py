from generateDataFuncs import *

EPSILON = 1e-4

uniqueSample = 10000
repeatSample = 2048
base = "/raid/vigneshk/data/"

thetaSave = base+"theta_data"
theta_standardSave = base+"thetaData_standard"
poissonSave = base+"poissonData"
thetaMeanSave = base + "thetaMean"
thetaStdSave = base + "thetaStd"

poissonScaled = base + "dP_scaled"
poissonScaled_mean = base + "dP_scaled_mean"
poissonScaled_std = base + "dP_scaled_std"

dataPoisson, thetaData, _ = generateTrainingData(uniqueSample,repeatSample)

print(f"Generated Data \n dataPoisson Shape: {dataPoisson.shape} \n Theta Shape : {thetaData.shape}")

thetaData_unique = thetaData[::repeatSample,:]

thetaMean = np.mean(thetaData_unique,axis=0)
thetaStd = np.std(thetaData_unique,axis = 0,ddof=0)
thetaData_standard = (thetaData - thetaMean)/(thetaStd+EPSILON)

print("Calculated Standardized Thetas")

dataPoisson_AT = 2 * np.sqrt(dataPoisson + 3/8)
dP_AT_mean = np.mean(dataPoisson_AT , axis = 0)
dP_AT_std = np.std(dataPoisson_AT, axis=0, ddof=0)
dataPoisson_AT = (dataPoisson_AT - dP_AT_mean) / (dP_AT_std + EPSILON)

print("Calculated Standardized Poisson")

np.save(thetaSave,thetaData)
np.save(theta_standardSave,thetaData_standard)
np.save(thetaMeanSave,thetaMean)
np.save(thetaStdSave,thetaStd)

np.save(poissonSave,dataPoisson)
np.save(poissonScaled, dataPoisson_AT)
np.save(poissonScaled_mean,dP_AT_mean)
np.save(poissonScaled_std,dP_AT_std)

print("Done Saving")
print(thetaData_unique)
print(dataPoisson)


