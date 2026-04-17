from modelClasses import CNF, autoEncoder
from generateDataFuncs import generateTrainingData, Compare_Theta, gauss, plots, doubleGaussCDF, generatePoissonData
from validatePlots import plotHist, ModeMeanShift, plot2DMarginals, ModeDBScan, findMode

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchinfo
import pickle

from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL


EPSILON = 1e-3

def GenPreds(base_PATH : str, iters : int):

    dnumber = 0
    device = torch.device(f"cuda:{dnumber}" if torch.cuda.is_available() else "cpu")
    print(device) 

    thetaMean = np.load("/raid/vigneshk/data/paramsMean.npy")
    thetaStd = np.load("/raid/vigneshk/data/paramsStd.npy") 

    dataTest = torch.tensor(np.load("/raid/vigneshk/data/dataTest.npy")).float()
    paramsTest = np.load("/raid/vigneshk/data/paramsTest.npy")

    CNFModel = CNF(n_features=int(paramsTest.shape[1]),
                   context_features=int(dataTest.shape[1]),
                   n_layers = 8,
                   hidden_features = 25,
                   num_bins = 16,
                   tails = "linear",
                   tail_bound = 3.5) 

    ckpt_CNF  = torch.load(base_PATH + "CNF_checkpoint.pt", map_location=device)
    CNFModel.load_state_dict(ckpt_CNF["CNF_Model"])
    CNFModel.eval()
    CNFModel = CNFModel.to(device)
 
    """
    trueParams = paramsTest[1]
    testData = dataTest[1].unsqueeze(0).to(device)

    print(testData.shape)
    #print(trueParams)

    with torch.no_grad():
        #enData = encodeModel._encode(testData)
        #enData = (enData - latent_mean)/(latent_std + EPSILON)
        samples = CNFModel.flow.sample(5000,context=testData).cpu().numpy()
        sample_cut = samples.reshape(-1,samples.shape[-1])
        #print(sample_cut)
        infer = ModeDBScan(sample_cut,0.5,5)
        infer = (infer * (thetaStd + EPSILON)) + thetaMean

    trueParams = (trueParams * (thetaStd + EPSILON)) + thetaMean

    return trueParams, infer

    """
    batches = DataLoader(dataTest,batch_size=10,shuffle = False)
    trueParams = (paramsTest * (thetaStd + EPSILON)) + thetaMean
    

    centerVals = []
    percDiffarr = []
    with torch.no_grad():
        for b in tqdm(batches): #batch
            x = b.to(device)
            samples = CNFModel.flow.sample(500,context=x).cpu().numpy()
            sample_cut = samples.reshape(-1,samples.shape[-1])
            infer = ModeDBScan(sample_cut,0.5,5)
            infer = (infer * (thetaStd + EPSILON)) + thetaMean
            centerVals.append(infer) 

    return trueParams, np.array(centerVals)

                

def valCNF(base_PATH : str, iters : int):

    titles = ["Delta_24","SinSq_24","SinSq_34","Theta_23","DMsq_41","DMsq_32"]    
    params, inferRet = GenPreds(base_PATH,iters)
    percDiff = (params - inferRet)*100/params

    print(params)
    print(inferRet)
    print(percDiff)

    plotHist(percDiff,titles,base_PATH)

    #plot2DMarginals(params,inferRet,titles,base_PATH)


if __name__ == "__main__":
    valCNF("/raid/vigneshk/Models/NOvACNF_LogData/", 1)


