from modelClasses import CNF, autoEncoder
from generateDataFuncs import generateTrainingData, Compare_Theta, gauss, plots, doubleGaussCDF
from validatePlots import plotHist, findMode

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchinfo
import pickle

from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL


EPSILON = 1e-4

def generatePoissonData(sampleNum,N1,N2,mu1,mu2,sig1,sig2): #N1,mu1,sig1
    minX_center = 0.5
    maxX_edge = 20.5
    step = 0.2 # -> bin width

    rawBins = np.arange(minX_center,maxX_edge,step=step)
      
    gaussSample = step * (gauss(N1,mu1,sig1,rawBins) + gauss(N2, mu2, sig2,rawBins)) 
      
    rng = np.random.default_rng()
    dataPoisson = rng.poisson(lam=gaussSample,size=None)

    return dataPoisson, gaussSample


def GenPreds(base_PATH : str, iters : int):

    dnumber = 0
    device = torch.device(f"cuda:{dnumber}" if torch.cuda.is_available() else "cpu")
    print(device) 

    latent_mean = torch.tensor(np.load("/raid/vigneshk/data/latentMean.npy")).float().to(device)
    latent_std = torch.tensor(np.load("/raid/vigneshk/data/latentStd.npy")).float().to(device) 

    dP_scaled_mean = np.load("/raid/vigneshk/data/dP_scaled_mean.npy")
    dP_scaled_std = np.load("/raid/vigneshk/data/dP_scaled_std.npy")

    thetaMean = np.load("/raid/vigneshk/data/thetaMean.npy")
    thetaStd = np.load("/raid/vigneshk/data/thetaStd.npy")


    CNFModel = CNF(n_features=6, #6
                   context_features=33, 
                   n_layers = 5,
                   hidden_features = 20,
                   num_bins = 16,
                   tails = "linear",
                   tail_bound = 3.5)


    ckpt_CNF  = torch.load(base_PATH + "CNF_checkpoint.pt", map_location=device)
    CNFModel.load_state_dict(ckpt_CNF["CNF_Model"])
    CNFModel.eval()
    CNFModel = CNFModel.to(device)


    encodeModel = autoEncoder(100,50,33)
    ckpt_AE  = torch.load("/raid/vigneshk/Models/AE_checkpoint.pt", map_location=device)
    encodeModel.load_state_dict(ckpt_AE["AE_Model"])
    encodeModel.eval()
    encodeModel = encodeModel.to(device)

    returnList = []
    
    for i in range(iters):
        dP , tD, _ = generateTrainingData(1,10000)

        #SCALING + STANDARDIZING
        dP_scaled_AT = 2 * np.sqrt(dP + 3/8)
        dP_scaled_AT = (dP_scaled_AT - dP_scaled_mean) / (dP_scaled_std + EPSILON)


        dP_ten = torch.tensor(dP_scaled_AT).float()
        batch_test = DataLoader(dP_ten,batch_size=1000)

        testData = []

        with torch.no_grad():
          for x_batch in batch_test:
            x = x_batch.to(device)
            cnfP_en = encodeModel._encode(x)
            cnfP_en = (cnfP_en - latent_mean)/(latent_std + EPSILON)
            samples = CNFModel.flow.sample(50,context=cnfP_en).cpu().numpy() 
            sample_cut = samples.reshape(-1,samples.shape[-1])
            testData.append(sample_cut)

        
        thetaDist = np.concatenate(testData,axis=0)
        thetaDist = (thetaDist * thetaStd) + thetaMean

        cnfT = tD[0]
        returnList.append([cnfT,thetaDist])        

    return returnList

                

def valCNF(base_PATH : str, iters : int):


    rawBins = np.arange(0.5,20.5,0.2)
    binEdges = np.linspace(0.4,20.4,len(rawBins)+1)
    titles = ["N1","N2","mu1","mu2","sig1","sig2"]

    infers = []
    refs = []

    dataList = GenPreds(base_PATH, iters)
    
    for data in tqdm(dataList):
        cnfT, thetaDist = data
        modeVals = findMode(thetaDist)
        dataPoisson , _ = generatePoissonData(1,*cnfT)
        cost = ExtendedBinnedNLL(dataPoisson.flatten(),binEdges,doubleGaussCDF)
        m = Minuit(cost,*modeVals)
        m.migrad()

        params = []
        for f in m.values:
            params.append(f)  
        paramsArr = np.array(params)
        
        infers.append(paramsArr)
        refs.append(cnfT)

        print(f"Real Theta : {cnfT}")
        print(f"Inferred+Fitted Theta : {paramsArr}")
    
    infersDist = np.vstack(infers)
    refsDist = np.vstack(refs)
    percDiff = (infersDist - refsDist)*100/refsDist
    
    plotHist(percDiff,titles,base_PATH)


valCNF("/raid/vigneshk/Models/CNF_BatchNormFinal/", 360)
