from generateDataFuncs import *

uniqueSample = 10000
repeatSample = 4096
thetaSave = "theta_data"
theta_standardSave = "thetaData_standard"
poissonSave = "poissonData"

dataPoisson, thetaData, _ = generateTrainingData(uniqueSample,repeatSample)

thetaData_unique = thetaData[::repeatSample,:]

thetaMean = np.mean(thetaData_unique,axis=0)
thetaStd = np.std(thetaData_unique,axis = 0)



thetaData_standard = (thetaData - thetaMean)/thetaStd

print(thetaData_standard)
print(dataPoisson)

np.save(thetaSave,thetaData)
np.save(theta_standardSave,thetaData_standard)
np.save(poissonSave,dataPoisson)
