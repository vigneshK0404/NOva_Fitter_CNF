from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np



def plotHist(thetaDist, ref_vals, titles):
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
        #data = data[(data > minVals[i]) & (data < maxVals[i])]
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
    

