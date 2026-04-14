from modelClasses import CNF, autoEncoder
from generateDataFuncs import generateTrainingData, Compare_Theta, gauss, plots, doubleGaussCDF, generatePoissonData
from validatePlots import plotHist, ModeMeanShift, plot2DMarginals, ModeDBScan

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchinfo
import pickle

from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL


EPSILON = 1e-4

def GenPreds(base_PATH : str, iters : int):

    dnumber = 0
    device = torch.device(f"cuda:{dnumber}" if torch.cuda.is_available() else "cpu")
    print(device) 

    #latent_mean = torch.tensor(np.load("/raid/vigneshk/data/latentMean.npy")).float().to(device)
    #latent_std = torch.tensor(np.load("/raid/vigneshk/data/latentStd.npy")).float().to(device)  

    thetaMean = np.load("/raid/vigneshk/data/paramsMean.npy")
    thetaStd = np.load("/raid/vigneshk/data/paramsStd.npy") 

    dataTest = torch.tensor(np.load("/raid/vigneshk/data/dataTest.npy")).float()
    paramsTest = np.load("/raid/vigneshk/data/paramsTest.npy")

    #print(dataTest)
    #print(paramsTest)

    CNFModel = CNF(n_features=6, #6
                   context_features=148, #TODO : change back to 49
                   n_layers = 6,
                   hidden_features = 25,
                   num_bins = 16,
                   tails = "linear",
                   tail_bound = 5) 

    ckpt_CNF  = torch.load(base_PATH + "CNF_checkpoint.pt", map_location=device)
    CNFModel.load_state_dict(ckpt_CNF["CNF_Model"])
    CNFModel.eval()
    CNFModel = CNFModel.to(device)
    
    """
    encodeModel = autoEncoder(148,74,49)
    ckpt_AE  = torch.load(base_PATH + "AE_checkpoint.pt", map_location=device)
    encodeModel.load_state_dict(ckpt_AE["AE_Model"])
    encodeModel.eval()
    encodeModel = encodeModel.to(device)

    trueParams = paramsTest[::100,:][1]
    testData = dataTest[100:200,:].to(device)

    print(testData.shape)

    with torch.no_grad():
        #enData = encodeModel._encode(testData)
        #enData = (enData - latent_mean)/(latent_std + EPSILON)
        samples = CNFModel.flow.sample(500,context=testData).cpu().numpy()
        sample_cut = samples.reshape(-1,samples.shape[-1])
        infer = ModeDBScan(sample_cut,0.5,150)
        infer = (infer * (thetaStd + EPSILON)) + thetaMean

    trueParams = (trueParams * (thetaStd + EPSILON)) + thetaMean


    return trueParams, infer

    """
    batches = DataLoader(dataTest,batch_size=100,shuffle = False)
    trueParams = (paramsTest[::100,:] * (thetaStd + EPSILON)) + thetaMean

    centerVals = []
    #idx = 0
    with torch.no_grad():
        for b in tqdm(batches):
            x = b.to(device)
            samples = CNFModel.flow.sample(300,context=x).cpu().numpy()
            sample_cut = samples.reshape(-1,samples.shape[-1])
            infer = ModeDBScan(sample_cut,0.5,50)
            infer = (infer * (thetaStd + EPSILON)) + thetaMean
            #print(trueParams[idx])
            #print(infer)
            #print("\n")
            #idx += 1

            centerVals.append(infer)  

    return trueParams, np.array(centerVals)
                

def valCNF(base_PATH : str, iters : int):

    titles = ["Delta_24","SinSq_24","SinSq_34","Theta_23","DMsq_41","DMsq_32"]    
    params, inferRet = GenPreds(base_PATH,iters)
    print(params)
    print(inferRet)

    percDiff = (params - inferRet)*100/params
    print(percDiff)

    plotHist(percDiff,titles,base_PATH)

    #plot2DMarginals(params,inferRet,titles,base_PATH)

    

    """
    rawBins = np.arange(0.5,20.5,0.2)
    binEdges = np.linspace(0.4,20.4,len(rawBins)+1)

    dataTestForDraw = dataTest[::100,:][0]
    paramsTestForDraw = paramsTest[::100,:][0]

    dataTestForDraw = dataTestForDraw.unsqueeze(0)

    percDiff = (thetaDist - paramList)*100/thetaDist
    plotHist(percDiff,titles,base_PATH)

    infers = []
    refs = []
    dataList = GenPreds(base_PATH, iters)

    counter = 0
    for data in tqdm(dataList):
        cnfT, thetaDist = data

        if counter == 0 :
            plot2DMarginals(thetaDist,titles,base_PATH)
            counter += 1
        
        modeVals = ModeMeanShift(thetaDist,smoothing=2,minRatio = 100)[0]
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
    """
    



if __name__ == "__main__":
    valCNF("/raid/vigneshk/Models/NOvACNF_NoAE/", 1)



"""

testData = []

with torch.no_grad():
  for x_batch in tqdm(batch_test):
    x = x_batch.to(device)
    cnfP_en = encodeModel._encode(x)
    cnfP_en = (cnfP_en - latent_mean)/(latent_std + EPSILON)
    samples = CNFModel.flow.sample(1,context=cnfP_en).cpu().numpy()
    sample_cut = samples.reshape(-1,samples.shape[-1])
    testData.append(sample_cut)

    
    thetaDist = np.concatenate(testData,axis=0)
    thetaDist = (thetaDist * thetaStd) + thetaMean
    paramsTest = (paramsTest * thetaStd) + thetaMean

"""
