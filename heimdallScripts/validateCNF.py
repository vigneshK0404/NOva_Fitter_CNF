from modelClasses import CNF, autoEncoder
from generateDataFuncs import generateTrainingData, plots, gauss
import numpy as np
import torch
import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages
from fpdf import FPDF


def generatePoissonData(sampleNum,N1,mu1,sig1,N2,mu2,sig2):
  minX_center = 0.5
  maxX_edge = 20.5
  step = 1 # -> bin width

  rawBins = np.arange(minX_center,maxX_edge,step=step)
  
  gaussSample = step * (gauss(N1,mu1,sig1,rawBins) + gauss(N2, mu2, sig2,rawBins))
  
  rng = np.random.default_rng()
  dataPoisson = rng.poisson(lam=gaussSample,size=None)

  return dataPoisson, gaussSample



EPSILON = 1e-6

def plotHist(thetaDist, ref_vals, titles, minVals, maxVals):
    iterations = thetaDist.shape[1]

    imagePath_Base = "/raid/vigneshk/plots/thetaPlots/"

    CNFn_layers = 2
    CNF_layerW = 20
    layerType = "Masked Autoregressive Transform"
    GradClip = "True"
    Epoch = 7
    Batch_Size = 1024
    Optim = "Adam"

    AE_layerW = 32
    AEn_layers = " Encode: 3 Decode: 3"
    AE_type = "AutoEncoder"
    AE_Optim = "Adam"


    full_string = f"CNF \n\n Layer_Info:-{layerType}: N-{CNFn_layers} W-{CNF_layerW} \n Optim, GradClip:{Optim},{GradClip} \n Training:- Epoch:{Epoch}, Batch:{Batch_Size} \n\n AE \n\n Type:{AE_type},{Optim} \n Layers, hidden_width:{AEn_layers},{AE_layerW} \n"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.multi_cell(w=0,h=10,txt=full_string,border=1)
    
    #with PdfPages("/raid/vigneshk/plots/thetaPlots/ThetaPlots.pdf") as pdf:
    for i in range(iterations):
        data = thetaDist[:,i]
        data = data[(data > minVals[i]) & (data < maxVals[i])]
        data_plot = abs(data - ref_vals[i])*100/ref_vals[i]

        imagePath = imagePath_Base + titles[i] + ".png"
        pdf.add_page()

        plt.figure()
        plt.hist(data_plot, edgecolor = "black")
        plt.xlabel("relative_difference %")
        plt.ylabel("counts")
        plt.title(titles[i])
        plt.savefig(imagePath)
        plt.close()

        pdf.image(imagePath,x=10,y=60,w=200,h=170)
    
    pdf.output(imagePath_Base + "ThetaPlots.pdf","F")

            

def valCNF():

    dnumber = 0
    device = torch.device(f"cuda:{dnumber}" if torch.cuda.is_available() else "cpu")
    print(device)


    
    dataPoisson_latent = torch.tensor(np.load("/raid/vigneshk/data/poissonEncoded.npy")).float()
    latent_mean = dataPoisson_latent.mean(axis=0).to(device)
    latent_std = dataPoisson_latent.std(axis=0, correction=0).to(device)

    thetaData = np.load("/raid/vigneshk/data/theta_data.npy")
    thetaData_unique = thetaData[::4096,:]

    thetaMean = np.mean(thetaData_unique,axis=0)
    thetaStd = np.std(thetaData_unique,axis = 0)


    CNFModel = CNF(n_features=6,
                   context_features=10,
                   n_layers = 2,
                   hidden_features = 20)


    ckpt_CNF  = torch.load("/raid/vigneshk/Models/CNF_checkpoint.pt", map_location=device)
    CNFModel.load_state_dict(ckpt_CNF["CNF_Model"])
    CNFModel.eval()
    CNFModel = CNFModel.to(device)

    encodeModel = autoEncoder(20,32,10)
    ckpt_AE  = torch.load("/raid/vigneshk/Models/AE_checkpoint.pt", map_location=device)
    encodeModel.load_state_dict(ckpt_AE["AE_Model"])
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

    #print(cnfT)
    #print(thetaDist)

    cnfTList = list(cnfT)

    thetaSample = list(thetaDist[0,:])
    dP1 , gT1 = generatePoissonData(1,*thetaSample)
    dPreal, gTreal = generatePoissonData(1,*cnfTList)

    rawBins = np.array(list(range(20)))
   

    plots(dP1,gT1,rawBins,"/raid/vigneshk/plots/poissonGeneratedFromCNF.png")
    plots(dPreal,gTreal,rawBins,"/raid/vigneshk/plots/poissonReal.png")

    
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

    titles = ["N1","mu1","sig1","N2","mu2","sig2"]
    minVals = [minN1,minmu1,minsig1,minN2,minmu2,minsig2]
    maxVals = [maxN1,maxmu1,maxsig1,maxN2,maxmu2,maxsig2]

    plotHist(thetaDist,cnfT,titles,minVals,maxVals)



#ToDo: Refactor this Script and Make it neater, too much boilerplate

def drawLatent():
    dataPoisson_latent = torch.tensor(np.load("/raid/vigneshk/data/poissonEncoded.npy")).float()
    latent_mean = dataPoisson_latent.mean(dim=0)
    latent_std = dataPoisson_latent.std(dim=0, correction=0)

    dP_std = (dataPoisson_latent - latent_mean) / (latent_std + EPSILON)

    print(latent_std)
    
    for i in range(10):
        data = dP_std[:,i]
        plt.hist(data)
        plt.savefig("/raid/vigneshk/plots/PoissonPlots/" + "LatentPoissonBin_Std"+str(i)+".png")
        plt.clf()
    


valCNF()  


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



