from modelClasses import CNF, autoEncoder
from generateDataFuncs import generateTrainingData, plots
import numpy as np
import torch
import matplotlib.pyplot as plt


dnumber = 0
device = torch.device(f"cuda:{dnumber}" if torch.cuda.is_available() else "cpu")
print(device)


EPSILON = 1e-6

dataPoisson_latent = torch.tensor(np.load("/raid/vigneshk/data/poissonEncoded.npy")).float()
latent_mean = dataPoisson_latent.mean(axis=0).to(device)
latent_std = dataPoisson_latent.std(axis=0, correction=0).to(device)

thetaData = np.load("/raid/vigneshk/data/theta_data.npy")
thetaData_unique = thetaData[::4096,:]

thetaMean = np.mean(thetaData_unique,axis=0)
thetaStd = np.std(thetaData_unique,axis = 0)


CNFModel = CNF(6,10,10,16)
ckpt_CNF  = torch.load("/raid/vigneshk/Models/CNF_checkpoint.pt", map_location=device)
CNFModel.load_state_dict(ckpt_CNF["CNF_Model"])
CNFModel.eval()
CNFModel = CNFModel.to(device)

encodeModel = autoEncoder(20,32,10)
ckpt_AE  = torch.load("/raid/vigneshk/Models/AE_checkpoint.pt", map_location=device)
encodeModel.load_state_dict(ckpt_AE)
encodeModel.eval()
encodeModel = encodeModel.to(device)



dP , tD, _ = generateTrainingData(1,1000)
dP_ten = torch.tensor(dP,device=device).float()
dP_en = encodeModel._encode(dP_ten)

dP_stan = (dP_en - latent_mean) / (EPSILON + latent_std)

with torch.no_grad():
    samples = CNFModel.flow.sample(1,context=dP_stan).cpu().numpy()
    sample_cut = samples.reshape(-1,samples.shape[-1])


thetaDist = (sample_cut * thetaStd) + thetaMean
cnfT = tD[0]

print(tD)
print(thetaDist)

N1_cnf = thetaDist[:,0]
mu1_cnf = thetaDist[:,1]
sig1_cnf = thetaDist[:,2]
N2_cnf = thetaDist[:,3]
mu2_cnf = thetaDist[:,4]
sig2_cnf = thetaDist[:,5]

minN1 = 50
maxN1 = 100

minN2 = 50
maxN2 = 70

minmu1 = 3
maxmu1 = 6

minmu2 = 12
maxmu2 = 15

minsig1 = 1
maxsig1 = 3

minsig2 = 1
maxsig2 = 3


def plotHist(data,ref_val,title,min,max):
  data = data[(data > min) & (data < max)]
  data_plot = data - ref_val
  plt.hist(data_plot, edgecolor = "black")
  plt.xlabel("relative_difference")
  plt.ylabel("counts")
  plt.title(title)
  plt.savefig("/raid/vigneshk/plots/" + title + ".png")
  plt.clf()


plotHist(N1_cnf, cnfT[0],"N1",minN1,maxN1)
plotHist(mu1_cnf, cnfT[1],"mu1",minmu1,maxmu1)
plotHist(sig1_cnf, cnfT[2],"sig1",minsig1,maxsig1)
plotHist(N2_cnf, cnfT[3],"N2",minN2,maxN2)
plotHist(mu2_cnf, cnfT[4],"mu2",minmu2,maxmu2)
plotHist(sig2_cnf, cnfT[5],"sig2",minsig2,maxsig2)



'''

testData = []

with torch.no_grad():
  for x_batch in batch_test:
    x = x_batch.to(device)
    cnfP_en = encodeModel.encode(x)
    samples = CNFModel.sample(1,context=cnfP_en).cpu().numpy()
    sample_cut = samples.reshape(-1,samples.shape[-1])
    testData.append(sample_cut)

'''



