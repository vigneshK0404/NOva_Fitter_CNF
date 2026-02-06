from modelClasses import CNF, autoEncoder
from generateDataFuncs import generateTrainingData, plots, gauss
from validatePlots import plotHist

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchinfo
import pickle


EPSILON = 1e-6

def generatePoissonData(sampleNum,N1,mu1,sig1,N2,mu2,sig2): #N1,mu1,sig1
    minX_center = 0.5
    maxX_edge = 20.5
    step = 1 # -> bin width

    rawBins = np.arange(minX_center,maxX_edge,step=step)
      
    gaussSample = step * (gauss(N1,mu1,sig1,rawBins) + gauss(N2, mu2, sig2,rawBins))
    #gaussSample = step * (gauss(N2, mu2, sig2,rawBins))
      
    rng = np.random.default_rng()
    dataPoisson = rng.poisson(lam=gaussSample,size=None)

    return dataPoisson, gaussSample
                

def valCNF(base_PATH : str):

    hyper_params = dict()
    
    dnumber = 0
    device = torch.device(f"cuda:{dnumber}" if torch.cuda.is_available() else "cpu")
    print(device) 

    latent_mean = torch.tensor(np.load("/raid/vigneshk/data/latentMean.npy")).float().to(device)
    latent_std = torch.tensor(np.load("/raid/vigneshk/data/latentStd.npy")).float().to(device) 

    thetaMean = np.load("/raid/vigneshk/data/thetaMean.npy")
    thetaStd = np.load("/raid/vigneshk/data/thetaStd.npy")


    CNFModel = CNF(n_features=6, #6
                   context_features=10,
                   n_layers = 5,
                   hidden_features = 20,
                   num_bins = 10,
                   tails = "linear",
                   tail_bound = 3.5)


    ckpt_CNF  = torch.load(base_PATH + "CNF_checkpoint.pt", map_location=device)
    CNFModel.load_state_dict(ckpt_CNF["CNF_Model"])
    CNFModel.eval()
    CNFModel = CNFModel.to(device)

    encodeModel = autoEncoder(20,32,10)
    ckpt_AE  = torch.load("/raid/vigneshk/Models/AE_checkpoint.pt", map_location=device)
    encodeModel.load_state_dict(ckpt_AE["AE_Model"])
    encodeModel.eval()
    encodeModel = encodeModel.to(device)



    dP , tD, _ = generateTrainingData(1,1)
    dP_ten = torch.tensor(dP).float()
    batch_test = DataLoader(dP_ten,batch_size=1)

    testData = []

    with torch.no_grad():
      for x_batch in tqdm(batch_test):
        x = x_batch.to(device)
        cnfP_en = encodeModel._encode(x)
        cnfP_en = (cnfP_en - latent_mean)/(latent_std + EPSILON)
        samples = CNFModel.flow.sample(10000,context=cnfP_en).cpu().numpy()
        sample_cut = samples.reshape(-1,samples.shape[-1])
        testData.append(sample_cut)

    
    thetaDist = np.concatenate(testData,axis=0)
    thetaDist = (thetaDist * thetaStd) + thetaMean

    cnfT = tD[0]
    thetaSample = list(thetaDist[0,:])
    dP1 , gT1 = generatePoissonData(1,*thetaSample)
    dPreal, gTreal = generatePoissonData(1,*cnfT)

    print(cnfT)
    print(thetaDist)

    rawBins = np.array(list(range(21)))
   
    

    plots(dP1,gT1,rawBins,base_PATH+"poissonGeneratedFromCNF.png")
    plots(dPreal,gTreal,rawBins,base_PATH+"poissonReal.png")
   

    titles = ["N1","mu1","sig1","N2","mu2","sig2"]
    #titles = ["N2","mu2","sig2"]
    
    plotHist(thetaDist,cnfT,titles,base_PATH)

#valCNF("/raid/vigneshk/Models/CNF_uniModal/")

"""
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

minVals = [minN1,minmu1,minsig1,minN2,minmu2,minsig2]
maxVals = [maxN1,maxmu1,maxsig1,maxN2,maxmu2,maxsig2]

"""
