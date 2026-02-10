from modelClasses import CNF, autoEncoder
from generateDataFuncs import generateTrainingData, Compare_Theta, gauss, plots
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
    step = 0.2 # -> bin width

    rawBins = np.arange(minX_center,maxX_edge,step=step)
      
    gaussSample = step * (gauss(N1,mu1,sig1,rawBins) + gauss(N2, mu2, sig2,rawBins)) 
      
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

    dP_scaled_mean = np.load("/raid/vigneshk/data/dP_scaled_mean.npy")
    dP_scaled_std = np.load("/raid/vigneshk/data/dP_scaled_std.npy")

    thetaMean = np.load("/raid/vigneshk/data/thetaMean.npy")
    thetaStd = np.load("/raid/vigneshk/data/thetaStd.npy")


    CNFModel = CNF(n_features=6, #6
                   context_features=33, #TODO make it 10 again
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



    dP , tD, _ = generateTrainingData(1,10000)

    #SCALING + STANDARDIZING
    dP_scaled_AT = 2 * np.sqrt(dP + 3/8)
    dP_scaled_AT = (dP_scaled_AT - dP_scaled_mean) / (dP_scaled_std + EPSILON)


    dP_ten = torch.tensor(dP_scaled_AT).float()
    batch_test = DataLoader(dP_ten,batch_size=1000)

    testData = []

    with torch.no_grad():
      for x_batch in tqdm(batch_test):
        x = x_batch.to(device)
        cnfP_en = encodeModel._encode(x)
        cnfP_en = (cnfP_en - latent_mean)/(latent_std + EPSILON)
        samples = CNFModel.flow.sample(1000,context=cnfP_en).cpu().numpy() #TODO changed the context to dP here change it back to cnfP_en
        sample_cut = samples.reshape(-1,samples.shape[-1])
        testData.append(sample_cut)

    
    thetaDist = np.concatenate(testData,axis=0)
    thetaDist = (thetaDist * thetaStd) + thetaMean

    cnfT = tD[0]
    

    print(cnfT)
    print(thetaDist)

    rawBins = np.array(list(range(100)))   

    titles = ["N1","mu1","sig1","N2","mu2","sig2"]
       
    worstTheta = plotHist(thetaDist,cnfT,titles,base_PATH)
    print(f"Worst Theta Value:{worstTheta}")
    worstPoisson, worstGauss = generatePoissonData(1,*worstTheta)
    dP_real , gT_real = generatePoissonData(1,*cnfT)
    Compare_Theta(worstGauss,gT_real,rawBins,base_PATH+"Gauss_ThetaReal_vs_WorstTheta")
    plots(dP_real,gT_real,rawBins,base_PATH+"poissonReal.png")
    plots(worstPoisson,worstGauss,rawBins,base_PATH+"poissonWorst.png")






valCNF("/raid/vigneshk/Models/CNF_AE_IncreasedBins_UpNorm/")

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
