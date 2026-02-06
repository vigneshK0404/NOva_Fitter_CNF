from generateDataFuncs import *

uniqueSample = 10000
repeatSample = 4096
base = "/raid/vigneshk/data/"

thetaSave = base+"theta_data"
theta_standardSave = base+"thetaData_standard"
poissonSave = base+"poissonData"
thetaMeanSave = base + "thetaMean"
thetaStdSave = base + "thetaStd"

dataPoisson, thetaData, _ = generateTrainingData(uniqueSample,repeatSample)

thetaData_unique = thetaData[::repeatSample,:]

thetaMean = np.mean(thetaData_unique,axis=0)
thetaStd = np.std(thetaData_unique,axis = 0,ddof=0)

thetaData_standard = (thetaData - thetaMean)/thetaStd

print(thetaData_standard)
print(dataPoisson)

np.save(thetaSave,thetaData)
np.save(theta_standardSave,thetaData_standard)
np.save(poissonSave,dataPoisson)
np.save(thetaMeanSave,thetaMean)
np.save(thetaStdSave,thetaStd)


