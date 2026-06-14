from modelClasses import CNF, autoEncoder
from validatePlots import plotHist, ModeMeanShift, plot2DMarginals, DBScan
from readCNFROOT import applyStd

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchinfo
import pickle
import uproot

import global_nums


repeatSize = global_nums.repeatSize
EPSILON = global_nums.EPSILON
middleRatio = global_nums.middleRatio
compressRatio = global_nums.compressRatio
dnumber = global_nums.dnumber


def GenPreds(base_PATH : str, AEModel : autoEncoder, CNFModel : CNF , device ,thetaMean,thetaStd,dataTest,paramsTest):

    batches = DataLoader(dataTest,batch_size=repeatSize,shuffle = False)
    trueParams = (paramsTest[::repeatSize,:] * (thetaStd + EPSILON)) + thetaMean
    centerVals = []
    percDiffarr = []
    NumSamples = 1000
    kSamples = 1000 

    with torch.no_grad():
        for b in tqdm(batches): #batch
            x = b.to(device)
            x_en = AEModel(x)
            x_en_firstExp = x_en.unsqueeze(1).expand(repeatSize,NumSamples,-1).reshape(NumSamples*repeatSize,-1)
            samples = CNFModel.flow.sample(NumSamples,context=x_en)
            sample_cut = samples.reshape(-1,samples.shape[-1])
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



def generate_seeds(data_path : str ,base_PATH : str, NumSamples : int, 
                  AEModel : autoEncoder, CNFModel : CNF, device, 
                  thetaMean, thetaStd):

    file = uproot.open(data_path + "sampleData.root")
    tree = file["dataTree"]
    branches = tree.arrays()
    data = np.array(branches["data"],dtype=np.int32)
    data = torch.tensor(applyStd(data), device=device).float()

    representatives = []

    with torch.no_grad():
        x_en = AEModel(data)
        samples = CNFModel.flow.sample(NumSamples,context=x_en)
        sample_cut = samples.reshape(-1,samples.shape[-1]).cpu().numpy()

        #print(f"sample_cut : {sample_cut.shape}")
        #print(f"x_en : {x_en.shape}")

        print("Running MMS")
        #clusters = DBScan(sample_cut, clusterDist = 1 , min_samples = int(NumSamples/1000))
        clusters = ModeMeanShift(sample_cut, 0.75, 1000)
        print("Finished MMS")

        for cluster in clusters:
            kSamples = cluster.shape[0]
            cluster = torch.tensor(cluster,device=device).float()
            x_en_Exp = x_en.unsqueeze(1).expand(1,kSamples,-1).reshape(kSamples,-1)
            #print(f"x_en_Exp : {x_en_Exp.shape}")
            firstPass = CNFModel(cluster,x_en_Exp)
            infer = cluster[torch.argmax(firstPass)].cpu().numpy()
            representatives.append((infer * (thetaStd + EPSILON)) + thetaMean)   
   
    reps = np.asarray(representatives,dtype=np.float32)
    print(reps)


    if reps.ndim == 1:
        reps = reps.reshape(1, -1)

    assert reps.shape[1] == 6, reps.shape

    with uproot.recreate(f"{data_path}cnfpreds.root") as f:
        f.mktree("tree", {"reps": "6 * float32"})
        f["tree"].extend({"reps": reps})


    return

    
def singlePred(base_PATH: str,AEModel : autoEncoder, CNFModel : CNF,device,thetaMean,thetaStd,dataTest,paramsTest):

    NumSamples = 5000
    kSamples = 500

    trueParams = paramsTest[0]
    testData = dataTest[:repeatSize,:].to(device)
    trueParams = (trueParams * (thetaStd + EPSILON)) + thetaMean

    print(testData.shape)
    centerVals = []

    with torch.no_grad():
        x_en = AEModel(testData)        
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





    


                

def valCNF(base_PATH : str, AEModel : autoEncoder, CNFModel : CNF, device, thetaMean,thetaStd,dataTest,paramsTest):

    titles = ["Delta_24","SinSq_24","SinSq_34","SinSq_23","DMsq_41","DMsq_32"]    
    params, inferRet = GenPreds(base_PATH, AEModel, CNFModel, device,
                                thetaMean,thetaStd,dataTest,paramsTest)
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
    print(percDiff)
    plotHist(percDiff,titles,base_PATH)


    #plot2DMarginals(params,inferRet,titles,base_PATH)


if __name__ == "__main__":

    base_PATH = "Models/Increased_AE_Binning_2/"
    data_path = "data/"
    thetaMean = np.load("data/processed/stats/theta_mean.npy")
    thetaStd = np.load("/raid/vigneshk/data/processed/stats/theta_std.npy") 

    device = torch.device(f"cuda:{dnumber}" if torch.cuda.is_available() else "cpu")
    print(device) 

    AEModel = autoEncoder(input_dim = 148,
                          middle_dim = int(148 * middleRatio),
                          output_dim = int(148 * compressRatio))

    CNFModel = CNF(n_features=6,
                   context_features=int(148 * compressRatio), 
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
        
    generate_seeds(data_path,base_PATH, 100000, AEModel , CNFModel, device, thetaMean, thetaStd)


    #dataTest = torch.from_numpy(np.load("data/processed/testing/17_data_0.npy")[:300000])
    #paramsTest = np.load("data/processed/testing/17_theta_0.npy")[:300000]
    #valCNF(base_PATH, AEModel, CNFModel,device, thetaMean,thetaStd,dataTest,paramsTest)
    




""" 
    params , inferRet = singlePred(base_PATH,
                              AEModel,CNFModel,thetaMean,
                              thetaStd,dataTest,paramsTest)

    print(params)
    print(inferRet)

    inferRet = inferRet.squeeze()
    percList = []
    for i in range(len(inferRet)):
        x = params[i]
        y = inferRet[i]
        if i in [1,2,4]:
            x = pow(10,x)
            y = pow(10,y)

        diff = 100 * (x - y) / (np.abs(x) + np.abs(y) + 1e-12)
        percList.append(diff)
    print(percList)

    """
