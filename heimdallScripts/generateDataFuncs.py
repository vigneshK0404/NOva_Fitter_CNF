import numpy as np
import matplotlib.pyplot as plt

alpha = 100

minN1 = alpha*50
maxN1 = alpha*1000

minN2 = alpha*50
maxN2 = alpha*700

minmu1 = 3
maxmu1 = 6

minmu2 = 12
maxmu2 = 15

minsig1 = 1
maxsig1 = 3

minsig2 = 1
maxsig2 = 3

def gauss(N,mu,sig,x):
    N_t = N.reshape(-1,1)
    mu_t = mu.reshape(-1,1)
    sig_t = sig.reshape(-1,1)

    term1 = N_t / (sig_t * np.sqrt(2 * np.pi))
    term2 = np.exp(-0.5 * np.square((x - mu_t) / sig_t))
    return term1 * term2

def generatePrior(sampleSize):
    N1 = np.random.uniform(minN1,maxN1,sampleSize)
    N2 = np.random.uniform(minN2,maxN2,sampleSize)

    mu1 = np.random.uniform(minmu1,maxmu1,sampleSize)
    mu2 = np.random.uniform(minmu2,maxmu2,sampleSize)

    sig1 = np.random.uniform(minsig1,maxsig1,sampleSize)
    sig2 = np.random.uniform(minsig2,maxsig2,sampleSize)

    return N1,N2,mu1,mu2,sig1,sig2

def generateTrainingData(uniqueNum, sampleNum):
  minX_center = 0.5
  maxX_edge = 20.5
  step = 0.2 # -> bin width

  rawBins = np.arange(minX_center,maxX_edge,step=step)
  genP = generatePrior(uniqueNum)
  N1,N2,mu1,mu2,sig1,sig2 = genP

  gaussSample = step * (gauss(N2, mu2, sig2,rawBins) + gauss(N1,mu1,sig1,rawBins))
  gaussSample_expanded = gaussSample[:,None,:]

  rng = np.random.default_rng()
  dataPoisson = rng.poisson(lam=gaussSample_expanded,size=(uniqueNum, sampleNum, gaussSample.shape[-1]))

  
  thetaData = np.column_stack(genP)
  fullthetaData = np.repeat(thetaData,repeats=sampleNum,axis=0)

  dataPoisson = dataPoisson.reshape(uniqueNum*sampleNum, gaussSample.shape[-1])
  
  return dataPoisson, fullthetaData, gaussSample



def plots(dP, fG, binEdges, address, bin_width):
    dP = dP.flatten()
    fG = fG.flatten()

    plt.bar(binEdges,fG,width=bin_width,label="Gaussian",edgecolor="black") 
    plt.bar(binEdges,dP,width=bin_width,label="Poisson",edgecolor="black",alpha=0.5) 

    plt.legend()
    plt.savefig(address)
    plt.clf()

def Compare_Theta(theta_gen, fG, binEdges, address, bin_width):
    theta_gen = theta_gen.flatten()
    fG = fG.flatten()

    print(theta_gen.shape)
    print(fG.shape)
    print(len(binEdges))

    plt.bar(binEdges,fG,width=bin_width,label="Theta_Real",edgecolor="black") 
    plt.bar(binEdges,theta_gen,width=bin_width,label="Theta_Gen",edgecolor="black",alpha=0.5) 

    plt.legend()
    plt.savefig(address)
    plt.clf()




