from modelClasses import CNF, Encoder
from validatePlots import plotHist, ModeMeanShift, plot2DMarginals
from readCNFROOT import applyStd

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import uproot
import consts

def GenPreds(base_PATH : str, EModel : Encoder, CNFModel : CNF , device 
             ,thetaMean : np.array , thetaStd : np.array, dataTest : np.array, paramsTest : np.array):

    batches = DataLoader(dataTest,batch_size=consts.repeatSize,shuffle = False)
    trueParams = (paramsTest[::consts.repeatSize,:] * (thetaStd + consts.EPSILON)) + thetaMean
    centerVals = []
    percDiffarr = []
    NumSamples = 1000
    kSamples = 1000 

    with torch.no_grad():
        for b in tqdm(batches): #batch
            x = b.to(device)
            x_en = EModel(x)
            x_en_firstExp = x_en.unsqueeze(1).expand(consts.repeatSize,NumSamples,-1).reshape(NumSamples*consts.repeatSize,-1)
            samples = CNFModel.flow.sample(NumSamples,context=x_en)
            sample_cut = samples.reshape(-1,samples.shape[-1])
            firstPass = CNFModel(sample_cut,x_en_firstExp)
            topidx = firstPass.topk(kSamples).indices

            topSamples = sample_cut[topidx]
    
            sample_exp = topSamples.unsqueeze(1).expand(kSamples,consts.repeatSize,-1).reshape(kSamples * consts.repeatSize, -1)
            x_en_repeat = x_en.unsqueeze(0).expand(kSamples,consts.repeatSize,-1).reshape(kSamples*consts.repeatSize , -1)
            
            logLik = CNFModel(sample_exp,context=x_en_repeat)
            sumLog = logLik.view(-1,consts.repeatSize).sum(dim=1)
            infer = topSamples[torch.argmax(sumLog)].cpu().numpy()

            infer = (infer * (thetaStd + consts.EPSILON)) + thetaMean
            centerVals.append(infer) 

    return trueParams, np.array(centerVals)



def generate_seeds(data_path : str ,base_PATH : str, NumSamples : int, 
                  EModel : Encoder, CNFModel : CNF, device, 
                  thetaMean, thetaStd):

    file = uproot.open(data_path + "sampleData.root")
    tree = file["dataTree"]
    branches = tree.arrays()
    data = np.array(branches["data"],dtype=np.int32)
    data = torch.tensor(applyStd(data), device=device).float()

    rep_list = []

    with torch.no_grad():
        x_en = EModel(data)
        samples = CNFModel.flow.sample(NumSamples,context=x_en)
        sample_cut = samples.reshape(-1,samples.shape[-1]).cpu().numpy()
        
        data_bunches = np.array_split(sample_cut,len(data))
        theta_bunches = torch.split(x_en,1)
 

        for true_theta, bunch in tqdm(zip(theta_bunches, data_bunches)):
            assert len(bunch) == NumSamples, len(bunch)
            assert len(true_theta) == 1, len(true_theta)

            representatives = []
            clusters = ModeMeanShift(bunch, 0.75, 1000)
            print(f"Num Clusters : {len(clusters)}")

            for cluster in clusters:
                kSamples = cluster.shape[0]
                cluster = torch.tensor(cluster,device=device).float()
                x_en_Exp = true_theta.unsqueeze(1).expand(1,kSamples,-1).reshape(kSamples,-1)
                firstPass = CNFModel(cluster,x_en_Exp)
                infer = cluster[torch.argmax(firstPass)]
                representatives.append(infer)   
   
            reps = torch.stack(representatives)
            x_en_reps = true_theta.unsqueeze(1).expand(1,len(reps),-1).reshape(len(reps),-1)
            logRankings = CNFModel(reps,x_en_reps).argsort(descending=True)
            rep_list.append(np.asarray(reps[logRankings][:consts.topExps].cpu())) 

    final_reps = np.stack(rep_list)
    final_reps *= (thetaStd + consts.EPSILON)
    final_reps += thetaMean
    final_reps = final_reps.astype(np.float32).reshape(consts.topExps*len(data),-1)
    print(final_reps)

    with uproot.recreate(f"{data_path}cnfpreds.root") as f:
        f["tree"] = {"reps": final_reps}

   
    return

    
def singlePred(base_PATH: str,EModel : Encoder, CNFModel : CNF,device,thetaMean,thetaStd,dataTest,paramsTest):

    NumSamples = 5000
    kSamples = 500

    trueParams = paramsTest[0]
    testData = dataTest[:consts.repeatSize,:].to(device)
    trueParams = (trueParams * (thetaStd + EPSILON)) + thetaMean

    print(testData.shape)
    centerVals = []

    with torch.no_grad():
        x_en = EModel(testData)        
        samples = CNFModel.flow.sample(NumSamples,context=x_en)
        sample_cut = samples.reshape(-1,samples.shape[-1])

        x_en_firstExp = x_en.unsqueeze(1).expand(consts.repeatSize,NumSamples,-1).reshape(NumSamples*consts.repeatSize,-1)
        firstPass = CNFModel(sample_cut,x_en_firstExp)
        topidx = firstPass.topk(kSamples).indices

        topSamples = sample_cut[topidx]

        sample_exp = topSamples.unsqueeze(1).expand(kSamples,consts.repeatSize,-1).reshape(kSamples * consts.repeatSize, -1)
        x_en_repeat = x_en.unsqueeze(0).expand(kSamples,consts.repeatSize,-1).reshape(kSamples*consts.repeatSize , -1)
        
        logLik = CNFModel(sample_exp,context=x_en_repeat)
        sumLog = logLik.view(-1,consts.repeatSize).sum(dim=1)
        infer = topSamples[torch.argmax(sumLog)].cpu().numpy()

        infer = (infer * (thetaStd + consts.EPSILON)) + thetaMean
        centerVals.append(infer) 

    return trueParams, np.array(centerVals)
                

def valCNF(base_PATH : str, EModel : Encoder, CNFModel : CNF, device, thetaMean,thetaStd,dataTest,paramsTest):

    titles = ["Delta_24","SinSq_24","SinSq_34","SinSq_23","DMsq_41","DMsq_32"]    
    params, inferRet = GenPreds(base_PATH, EModel, CNFModel, device,
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

    base_PATH = "Models/NOvACNF_ThickerModel/"
    thetaMean = np.load(consts.theta_mean_path)
    thetaStd = np.load(consts.theta_std_path) 

    device = torch.device(f"cuda:{consts.dnumber}" if torch.cuda.is_available() else "cpu")
    print(device) 

    EModel = Encoder(input_dim = consts.input_dim,
                          middle_dim = consts.middle_dim,
                          output_dim = consts.output_dim)

    CNFModel = CNF(n_features = consts.n_features,
                   context_features = consts.context_features, 
                   n_layers = consts.n_layers, hidden_features = consts.hidden_features, 
                   num_bins = consts.num_bins, tails = consts.tails, 
                   tail_bound = consts.tail_bound) 

    ckpt = torch.load(base_PATH + "Model_checkpoint.pt", map_location=device)
    CNFModel.load_state_dict(ckpt["CNF_Model"])
    CNFModel.eval()
    CNFModel = CNFModel.to(device)

    EModel.load_state_dict(ckpt["AE_Model"])
    EModel.eval()
    EModel = EModel.to(device)
        
    generate_seeds(consts.base_path, base_PATH, 50000, EModel , CNFModel, device, thetaMean, thetaStd)


    #dataTest = torch.from_numpy(np.load("data/processed/testing/17_data_0.npy")[:300000])
    #paramsTest = np.load("data/processed/testing/17_theta_0.npy")[:300000]
    #valCNF(base_PATH, EModel, CNFModel,device, thetaMean,thetaStd,dataTest,paramsTest)
    








"""
 if reps.ndim == 1:
        reps = reps.reshape(1, -1)

    assert reps.shape[1] == 6, reps.shape

    with uproot.recreate(f"{data_path}cnfpreds.root") as f:
        f.mktree("tree", {"reps": "6 * float32"})
        f["tree"].extend({"reps": reps})

"""
