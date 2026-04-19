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

repeatSize = 10

def GenPreds(base_PATH : str):

    middleRatio = 0.75
    compressRatio = 0.5

    dnumber = 0
    device = torch.device(f"cuda:{dnumber}" if torch.cuda.is_available() else "cpu")
    print(device) 

    thetaMean = np.load("/raid/vigneshk/data/paramsMean.npy")
    thetaStd = np.load("/raid/vigneshk/data/paramsStd.npy") 

    dataTest = torch.tensor(np.load("/raid/vigneshk/data/dataTest.npy")).float()
    paramsTest = np.load("/raid/vigneshk/data/paramsTest.npy")

    AEModel = autoEncoder(input_dim = int(dataTest.shape[1]),
                          middle_dim = int(dataTest.shape[1] * middleRatio),
                          output_dim = int(dataTest.shape[1] * compressRatio))

    CNFModel = CNF(n_features=int(paramsTest.shape[1]),
                   context_features=int(dataTest.shape[1] * compressRatio), 
                   n_layers = 8, hidden_features = 30, 
                   num_bins = 24, tails = "linear", 
                   tail_bound = 3.5) 

    ckpt = torch.load(base_PATH + "Model_checkpoint.pt", map_location=device)
    CNFModel.load_state_dict(ckpt["CNF_Model"])
    CNFModel.eval()
    CNFModel = CNFModel.to(device)

    AEModel.load_state_dict(ckpt["AE_Model"])
    AEModel.eval()
    AEModel = AEModel.to(device)
 
    batches = DataLoader(dataTest,batch_size=repeatSize,shuffle = False)
    trueParams = (paramsTest[::repeatSize,:] * (thetaStd + EPSILON)) + thetaMean
    
    print("newCode")
    centerVals = []
    percDiffarr = []
    NumSamples = 1000
    kSamples = 1000

    with torch.no_grad():
        for b in tqdm(batches): #batch
            x = b.to(device)
            x_en = AEModel(x)
            #x_en_rep = x_en.mean(dim=0)
            samples = CNFModel.flow.sample(NumSamples,context=x_en)
            sample_cut = samples.reshape(-1,samples.shape[-1])

            x_en_firstExp = x_en.unsqueeze(1).expand(repeatSize,NumSamples,-1).reshape(NumSamples*repeatSize,-1)
            firstPass = CNFModel(sample_cut,x_en_firstExp)
            topidx = firstPass.topk(kSamples).indices

            topSamples = sample_cut[topidx]
    
            sample_exp = topSamples.unsqueeze(1).expand(kSamples,repeatSize,-1).reshape(kSamples * repeatSize, -1)
            x_en_repeat = x_en.unsqueeze(0).expand(kSamples,repeatSize,-1).reshape(kSamples*repeatSize , -1)
            
            logLik = CNFModel(sample_exp,context=x_en_repeat)
            sumLog = logLik.view(-1,repeatSize).sum(dim=1)
            infer = topSamples[torch.argmax(sumLog)].cpu().numpy()

            infer = (infer * (thetaStd + EPSILON)) + thetaMean
            centerVals.append(infer) 

    return trueParams, np.array(centerVals)


                

def valCNF(base_PATH : str):

    titles = ["Delta_24","SinSq_24","SinSq_34","SinSq_23","DMsq_41","DMsq_32"]    
    params, inferRet = GenPreds(base_PATH)
    np.save(base_PATH+"inferenceResults",inferRet)
   
    print(inferRet.shape)
    percList = []
    for i in range(inferRet.shape[1]):
        x = params[:,i]
        y = inferRet[:,i]
        if i in [1,2,4]:
            x = pow(10,x)
            y = pow(10,y)

        diff = 100 * (x - y) / (np.abs(x) + np.abs(y) + 1e-12)
        percList.append(diff)

    percDiff = np.transpose(np.vstack(percList))
    plotHist(percDiff,titles,base_PATH)


    plot2DMarginals(params,inferRet,titles,base_PATH)


if __name__ == "__main__":
    valCNF("/raid/vigneshk/Models/NOvACNF_ThickerModel/")



""" 
    trueParams = paramsTest[0]
    testData = dataTest[:10,:].to(device)

    print(testData.shape)

    with torch.no_grad():
        enData = AEModel(testData)
        samples = CNFModel.flow.sample(500,context=enData).cpu().numpy()
        sample_cut = samples.reshape(-1,samples.shape[-1])
        infer = ModeDBScan(sample_cut,0.5,5)
        infer = (infer * (thetaStd + EPSILON)) + thetaMean

    trueParams = (trueParams * (thetaStd + EPSILON)) + thetaMean

    return trueParams, infer

    """
