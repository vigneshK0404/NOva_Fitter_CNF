from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN

def findMode(thetaDist : np.array):

    iterations = thetaDist.shape[1]
    modeVals = []

    for i in range(iterations):
        data = thetaDist[:,i]
        hist, bin_edges = np.histogram(data)
        idx = np.argmax(hist)
        mode = (bin_edges[idx] + bin_edges[idx+1])/2
        modeVals.append(mode.item())

    return np.array(modeVals)

def DBScan(thetaDist : np.array, clusterDist : float, min_samples : int):
    db = DBSCAN(eps = clusterDist, min_samples = min_samples).fit(thetaDist)
    uniqueClusters = set(db.labels_)
    
   
    clusters = []

    for i in uniqueClusters:
        mask = db.labels_ == i
        clusters.append(thetaDist[mask])
                 

    return clusters


def plot2DMarginals(truth : np.array, thetaDist : np.array, titles : list, base_PATH : str):

    iterations = thetaDist.shape[1]
    
    pdf = FPDF()
    
    for i in range(iterations):
        x = thetaDist[:,i]
        trueX = truth[i]
        titleX = titles[i]
        for j in range(i+1,iterations):
            y = thetaDist[:,j]
            trueY = truth[j]
            titleY = titles[j]
            title = f"{titleX} vs {titleY}"
            imagePath = base_PATH + title + ".png"
            plt.figure()            
            H, xeds, yeds, _ = plt.hist2d(x,y,bins=100,cmap='viridis')
            plt.axvline(x=trueX, linestyle='--',color='red')
            plt.axhline(y=trueY, linestyle='--',color='red')
            idx = np.unravel_index(np.argmax(H),H.shape)
            modeXCol = (xeds[idx[0]] + xeds[idx[0]+1])/2
            modeYRow = (yeds[idx[1]] + yeds[idx[1]+1])/2
            print("")
            print(f"idx : {idx}")
            print(f"{titleX} : [{xeds[0]}, {xeds[-1]}] -> {modeXCol}")
            print(f"{titleY} : [{yeds[0]}, {yeds[-1]}] -> {modeYRow}")
            print("")

            plt.colorbar(label='Frequency of points')
            plt.xlabel(titleX)
            plt.ylabel(titleY)
            plt.title(title)
            plt.locator_params(axis='both', nbins=10)
            plt.savefig(imagePath)
            plt.close()

            pdf.add_page()
            pdf.image(imagePath,x=10,y=60,w=200,h=170)
    pdf.output(base_PATH + "2DMarginals.pdf", "F")
        



def plotHist(thetaDist : np.array , titles : list, base_PATH : str):

    iterations = thetaDist.shape[1]

    
    PATH = base_PATH + "hP.bin"
    with open(PATH, 'rb') as handle:
        hyper_params = pickle.load(handle)

    full_string = str()
   
    for key,value in hyper_params.items():
        full_string += key
        full_string += " : "
        full_string += str(value)
        full_string += "\n"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.multi_cell(w=0,h=10,txt=full_string,border=1)
    
    for i in range(iterations):
        data_plot = thetaDist[:,i]
        outOfRange = data_plot[(data_plot > 100) | (data_plot < -100)]
        print(f"{titles[i]} Out of Range : {outOfRange}")

        imagePath = base_PATH + titles[i] + ".png"
        pdf.add_page()

        plt.figure()
        _, bins, _ = plt.hist(data_plot, edgecolor = "black")
        bin_diffs = np.diff(bins)
        if np.all(bin_diffs - bin_diffs[0]) == 0 :
            print(f"uniform binning : {titles[i]} - {bin_diffs[0]}")
        else :
            print(f"non-uniform binnins : {titles[i]} - {bin_diffs}")

        plt.xlabel("relative_difference %")
        plt.ylabel("counts")
        plt.title(titles[i])
        plt.savefig(imagePath)
        plt.close()
        pdf.image(imagePath,x=10,y=60,w=200,h=170)
    
    pdf.output(base_PATH + "ThetaPlots.pdf","F")



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

