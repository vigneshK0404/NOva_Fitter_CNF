import numpy as np
import matplotlib.pyplot as plt


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

def gauss(N,mu,sig,x):
    N_t = N.reshape(-1,1)
    mu_t = mu.reshape(-1,1)
    sig_t = sig.reshape(-1,1)

    term1 = N_t / (sig_t * np.sqrt(2 * np.pi))
    term2 = np.exp(-0.5 * np.square((x - mu_t) / sig_t))
    return np.array(term1 * term2)

def generatePrior(sampleSize):
    N1 = np.random.uniform(minN1,maxN1,sampleSize)
    N2 = np.random.uniform(minN2,maxN2,sampleSize)

    mu1 = np.random.uniform(minmu1,maxmu1,sampleSize)
    mu2 = np.random.uniform(minmu2,maxmu2,sampleSize)

    sig1 = np.random.uniform(minsig1,maxsig1,sampleSize)
    sig2 = np.random.uniform(minsig2,maxsig2,sampleSize)

    return N1,mu1,sig1,N2,mu2,sig2

def generateTrainingData(uniqueNum, sampleNum):
  minX_center = 0.5
  maxX_edge = 20.5
  step = 1 # -> bin width

  rawBins = np.arange(minX_center,maxX_edge,step=step)
  genP = generatePrior(uniqueNum)
  N1,mu1,sig1,N2,mu2,sig2 = genP

  gaussSample = step * (gauss(N1,mu1,sig1,rawBins) + gauss(N2, mu2, sig2,rawBins))
  fullGaussMatrix = np.repeat(gaussSample,repeats=sampleNum,axis=0)

  thetaData = np.column_stack(genP)
  fullthetaData = np.repeat(thetaData,repeats=sampleNum,axis=0)

  rng = np.random.default_rng()
  dataPoisson = rng.poisson(lam=fullGaussMatrix,size=None)

  return dataPoisson, fullthetaData, fullGaussMatrix



def plots(dP, fG, binEdges, address):
  plt.plot(binEdges[1:], fG.flatten()[:-1])
  plt.hist(binEdges[:-1], binEdges, weights=fG.flatten()[:-1],
           edgecolor='black', label='gauss')
  plt.hist(binEdges[:-1], binEdges, weights=dP.flatten()[:-1],edgecolor='black', label='poisson')
  plt.legend()
  plt.savefig(address)
  plt.clf()

