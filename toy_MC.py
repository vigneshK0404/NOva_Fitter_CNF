{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "A100",
      "machine_shape": "hm",
      "authorship_tag": "ABX9TyMO/GEmsqFk2b5pHUjKEAAJ"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "!pip install torchinfo\n",
        "!pip install nflows"
      ],
      "metadata": {
        "id": "6k71TZgr1wJ9"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "import torchinfo\n",
        "import numpy as np\n",
        "import sklearn\n",
        "import matplotlib.pyplot as plt\n",
        "import tqdm\n",
        "from torch.utils.data import DataLoader\n",
        "import time"
      ],
      "metadata": {
        "id": "hEV9mG-mOFo8"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from nflows.flows.base import Flow  # a container for full Flow\n",
        "from nflows.distributions.normal import StandardNormal  # Gaussian latent space distribution\n",
        "from nflows.transforms.base import (\n",
        "    CompositeTransform,\n",
        ")  # a wrapper to stack simpler transformations to form a more complex one\n",
        "from nflows.transforms.autoregressive import (\n",
        "    MaskedAffineAutoregressiveTransform,\n",
        ")  # the basic transformation, which we will stack several times\n",
        "from nflows.transforms.autoregressive import (\n",
        "    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,\n",
        ")  # the basic transformation, which we will stack several times\n",
        "from nflows.transforms.permutations import ReversePermutation # a layer that simply reverts the order of outputs"
      ],
      "metadata": {
        "id": "BI0xrzb40TXx"
      },
      "execution_count": 88,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(torch.cuda.is_available())\n",
        "print(torch.cuda.device_count())\n",
        "dnumber = 0\n",
        "device = torch.device(f\"cuda:{dnumber}\" if torch.cuda.is_available() else \"cpu\")\n",
        "print(device)\n",
        "device_name = torch.cuda.get_device_name(dnumber)\n",
        "print(device_name)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "795Sn-feCNZK",
        "outputId": "a547f8c4-8b9d-45ad-ba68-3cec823b97ff"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "True\n",
            "1\n",
            "cuda:0\n",
            "NVIDIA A100-SXM4-80GB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def gauss(N,mu,sig,x):\n",
        "    N_t = N.reshape(-1,1)\n",
        "    mu_t = mu.reshape(-1,1)\n",
        "    sig_t = sig.reshape(-1,1)\n",
        "\n",
        "    term1 = N_t / (sig_t * np.sqrt(2 * np.pi))\n",
        "    term2 = np.exp(-0.5 * np.square((x - mu_t) / sig_t))\n",
        "    return np.array(term1 * term2)\n",
        "\n",
        "\n",
        "def generatePrior(sampleSize):\n",
        "    N1 = np.random.uniform(10,50,sampleSize)\n",
        "    N2 = np.random.uniform(10,30,sampleSize)\n",
        "\n",
        "    mu1 = np.random.uniform(1,3,sampleSize)\n",
        "    mu2 = np.random.uniform(5,9,sampleSize)\n",
        "\n",
        "    sig1 = np.random.uniform(1,3,sampleSize)\n",
        "    sig2 = np.random.uniform(5,9,sampleSize)\n",
        "\n",
        "    return N1,mu1,sig1,N2,mu2,sig2\n",
        "\n",
        "\n",
        "def generateTrainingData(sampleNumber):\n",
        "  N1,mu1,sig1,N2,mu2,sig2 = generatePrior(sampleNumber)\n",
        "\n",
        "  raw = np.arange(0.5,10,1) #startbinCenter, endBinEdge, StepSize [0.5,1.5...9.5]\n",
        "  gaussTotal = gauss(N1,mu1,sig1,raw) + gauss(N2,mu2,sig2,raw)\n",
        "\n",
        "  binNumber = len(gaussTotal)\n",
        "  dataPoisson = np.random.poisson(lam=gaussTotal,size=None)\n",
        "\n",
        "  #thetaData = np.column_stack((N1,mu1,sig1,N2,mu2,sig2))\n",
        "  return dataPoisson\n",
        "\n",
        "dataPoisson = generateTrainingData(10000000)"
      ],
      "metadata": {
        "id": "XUvfNJ-U2Eet"
      },
      "execution_count": 89,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "I am going to normalize the data using min-max I can preserve the relationships while keeping it ideal for sigmoid to take part perhaps. I also found appraches involving log, if this does not work I will look into that."
      ],
      "metadata": {
        "id": "qjNNuwT3LsKq"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "max_vals = torch.tensor(np.max(dataPoisson, axis = 0))\n",
        "min_vals = torch.tensor(np.min(dataPoisson, axis = 0))\n",
        "input_p = torch.tensor(dataPoisson,dtype=torch.float32)"
      ],
      "metadata": {
        "id": "Ngyxhfg0Dzqs"
      },
      "execution_count": 90,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "We may have to rethink structure of data storage for CNF later. But for now the autoencoder simply needs to take the 10-D output in and minimize loss."
      ],
      "metadata": {
        "id": "_vYSvuGE3dgd"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "Datasets = DataLoader(input_p, batch_size =4096, shuffle=True)\n",
        "\n",
        "#for data in Datasets:\n",
        "#  print(data.shape)"
      ],
      "metadata": {
        "id": "IhTp8tjKNQZK"
      },
      "execution_count": 91,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "class autoEncoder(torch.nn.Module):\n",
        "  def __init__(self, input_dim, hidden_dim):\n",
        "    super().__init__()\n",
        "\n",
        "    #self.max = torch.nn.Parameter(max_vals,requires_grad=False)\n",
        "    #self.min = torch.nn.Parameter(min_vals,requires_grad=False)\n",
        "\n",
        "    self.register_buffer(\"min\", min_vals)\n",
        "    self.register_buffer(\"max\", max_vals)\n",
        "\n",
        "\n",
        "\n",
        "    self.encoder = torch.nn.Sequential(\n",
        "        torch.nn.Linear(input_dim,32),\n",
        "        torch.nn.ReLU(),\n",
        "        torch.nn.Linear(32,hidden_dim),\n",
        "    )\n",
        "\n",
        "    self.decoder = torch.nn.Sequential(\n",
        "        torch.nn.Linear(hidden_dim,32),\n",
        "        torch.nn.ReLU(),\n",
        "        torch.nn.Linear(32,input_dim)\n",
        "    )\n",
        "\n",
        "  def forward(self,x):\n",
        "    x = (x-self.min)/(self.max - self.min + 1e-8)\n",
        "\n",
        "    encode = self.encoder(x)\n",
        "    decode = self.decoder(encode)\n",
        "\n",
        "    y = decode*(self.max - self.min) + self.min\n",
        "    return y"
      ],
      "metadata": {
        "id": "jVeDFxHo3ohZ"
      },
      "execution_count": 92,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "lr = 1e-4\n",
        "inputDim = 10\n",
        "hiddenDim = 6\n",
        "compress = autoEncoder(inputDim, hiddenDim).to(device)\n",
        "optim = torch.optim.Adam(compress.parameters(), lr=lr)\n",
        "loss_fn = torch.nn.MSELoss()\n",
        "torchinfo.summary(compress)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "e1PYEvN36bgS",
        "outputId": "3779db9c-e37b-4540-9604-c9137b7cf187"
      },
      "execution_count": 93,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "=================================================================\n",
              "Layer (type:depth-idx)                   Param #\n",
              "=================================================================\n",
              "autoEncoder                              --\n",
              "├─Sequential: 1-1                        --\n",
              "│    └─Linear: 2-1                       352\n",
              "│    └─ReLU: 2-2                         --\n",
              "│    └─Linear: 2-3                       198\n",
              "├─Sequential: 1-2                        --\n",
              "│    └─Linear: 2-4                       224\n",
              "│    └─ReLU: 2-5                         --\n",
              "│    └─Linear: 2-6                       330\n",
              "=================================================================\n",
              "Total params: 1,104\n",
              "Trainable params: 1,104\n",
              "Non-trainable params: 0\n",
              "================================================================="
            ]
          },
          "metadata": {},
          "execution_count": 93
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "max_iter = 5\n",
        "losses = []\n",
        "\n",
        "\n",
        "for iter in tqdm.tqdm(range(max_iter)):\n",
        "  iter_losses = []\n",
        "  for x_batch in Datasets:\n",
        "      x_batch = x_batch.to(device, non_blocking=True)\n",
        "      optim.zero_grad()\n",
        "      y_pred = compress(x_batch)\n",
        "      loss = loss_fn(y_pred, x_batch)\n",
        "      iter_losses.append(loss.item())\n",
        "      loss.backward()\n",
        "      optim.step()\n",
        "\n",
        "  losses.append(np.mean(np.array(iter_losses)))\n",
        "\n",
        "print(len(losses))\n",
        "plt.plot(losses)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 485
        },
        "id": "tlJ1RSQ_Le-1",
        "outputId": "5ca8cf8f-b413-4873-8bd7-0209de8b6e57"
      },
      "execution_count": 94,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 5/5 [03:12<00:00, 38.47s/it]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "5\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[<matplotlib.lines.Line2D at 0x7b09a45fb650>]"
            ]
          },
          "metadata": {},
          "execution_count": 94
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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XocPH9ZvfvMbu0uJiNGOz2nfw+LiYlVVVamlpUX/+7//q2effVZ79uwZ8RevE4UzRqedv9raWr344ovatWtXTN9k6ohgMXnyZLndbl24cGHI/gsXLig/P3/Y1+Tn54d1vJ3GMr7rJSUlafHixTp58mQkSoy6kc6fz+fThAkTbKoq8u65556Y/2X9wgsvaPv27dq7d6+mTp1602Od9D0MCWd814v172FycrJmzpwpSbrzzju1f/9+/eAHP9Drr79+w7FOPHdSeGO8XqyfvwMHDqixsVFLliwZ2Nfb26u9e/fqn//5nxUIBOR2u4e8xo7z6IipkOTkZN1555169913B/YFg0G9++67I86dLV26dMjxkrRr166bzrXZZSzju15vb6+qq6tVUFAQqTKjyknnz0pVVVUxew5N09QLL7ygLVu26L333tP06dNv+RonncexjO96TvseBoNBBQKBYX/mpHN3Mzcb4/Vi/fw98sgjqq6uVlVV1cB21113ad26daqqqrohVEg2nceI3RZqsc2bN5ter9fctGmTWVNTY3796183J06caDY0NJimaZpf+cpXzJdeemng+A8++MD0eDzm3/3d35lHjx41X375ZTMpKcmsrq62awg3Fe74/uZv/sbcuXOneerUKfPAgQPmH/zBH5gpKSnmkSNH7BrCTbW2tpqVlZVmZWWlKcn83ve+Z1ZWVppnzpwxTdM0X3rpJfMrX/nKwPGfffaZmZqaav75n/+5efToUfPVV1813W63uWPHDruGcEvhjvH73/++WVFRYZ44ccKsrq42X3zxRdPlcpnvvPOOXUO4qeeff97MzMw0d+/ebdbX1w9sHR0dA8c4+Xs4lvE56Xv40ksvmXv27DFPnz5tfvLJJ+ZLL71kGoZh/upXvzJN09nnLiTcMTrp/I3k+qdCYuE8OiZYmKZp/tM//ZN52223mcnJyeY999xj7tu3b+BnDz74oPnss88OOf6///u/zdmzZ5vJycnmvHnzzLfeeivKFYcnnPFt2LBh4Ni8vDzzySefNA8ePGhD1aMTerTy+i00pmeffdZ88MEHb3jNokWLzOTkZPOOO+4w33jjjajXHY5wx/jd737XnDFjhpmSkmJOmjTJXL58ufnee+/ZU/woDDc2SUPOi5O/h2MZn5O+h3/0R39kTps2zUxOTjZzcnLMRx55ZOAXrmk6+9yFhDtGJ52/kVwfLGLhPNI2HQAAWMYR91gAAABnIFgAAADLECwAAIBlCBYAAMAyBAsAAGAZggUAALAMwQIAAFiGYAEAACxDsAAAAJYhWAAAAMsQLAAAgGUIFgAAwDL/H4ujwzf0jgoEAAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(losses[-1])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "q4SMSYRUcMcA",
        "outputId": "c0851b12-3084-475f-f05e-b6c9699d6894"
      },
      "execution_count": 95,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "0.6010078921072021\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "testP = torch.tensor(generateTrainingData(1), dtype = torch.float32)\n",
        "testP = testP.to(device)\n",
        "decodePtest = compress(testP)"
      ],
      "metadata": {
        "id": "ZAQ8V0kmlxyu"
      },
      "execution_count": 97,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "decodePtest = compress(testP).to(\"cpu\").detach().numpy().flatten()\n",
        "testP_CPU = testP.to(\"cpu\").detach().numpy().flatten()"
      ],
      "metadata": {
        "id": "zu7CDDf9mwFv"
      },
      "execution_count": 98,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(decodePtest)\n",
        "print(testP_CPU)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tuvMDxcjmd6e",
        "outputId": "0c242ce6-b7a5-4d01-d773-1a2d4350c32d"
      },
      "execution_count": 99,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[3.9778435 3.9989376 9.026543  8.03295   6.922861  2.1522017 1.8487747\n",
            " 1.4326153 1.228493  1.1112852]\n",
            "[4. 4. 9. 8. 7. 3. 1. 1. 0. 0.]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "bins = np.array(range(11))\n",
        "plt.hist(bins[:-1], bins,weights=testP_CPU,color='blue',edgecolor='black',alpha=0.5,label=\"TrueBin\")\n",
        "plt.hist(bins[:-1], bins,weights=decodePtest,color='red',edgecolor='black',alpha=0.4,label=\"decodedBin\")\n",
        "plt.legend()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 448
        },
        "id": "00wMnq1mnFGM",
        "outputId": "df5d99b2-57ad-470b-e121-53783f36a43c"
      },
      "execution_count": 100,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.legend.Legend at 0x7b0acd183b90>"
            ]
          },
          "metadata": {},
          "execution_count": 100
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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69atuvLKKyVJL7zwgvr06XPG73z//ffroYcektfrldvt1tChQ7Vo0aLzvk8+n0/Lly9XamqqJOm2227T2rVrm/Uy74QFACDiHDx4UPX19UH/Y+/QoYMuu+wySdKuXbvk9XrVq1evoOd5PB5lZmZK+naPxQ033HDO129oaNCIESMCyxISEjRkyBDt3r1bkrR79+4z9hgMGzYs6P6nn36qzz77LOjjDb/fL5/Pp/Lycu3bt0/x8fH64Q9/GPh57969z/qNksLCQk2aNEl+v18VFRV64IEHNG7cOG3YsEFxcXFn/T26d+8eiApJys7O/t69IReKsAAARB2n06m4uDjt2LHjjI3udweItsSBjk6nU/fcc4+mTZt2xs+6deumffv2hfxaHTt2VI8ePSR9u+dj8eLFGjZsmD788EONGTPmrM9JSEgIuh8TEyOfzxfGbxA+jrEAAEScSy+9VAkJCdqyZUtgWVVVVWBDPXDgQHm9XlVWVqpHjx5Bty5dukiSBgwYoLVr157z9RMTE/XRRx8FljU0NGjbtm3q27evJKlPnz7aunVr0PM2b94cdH/QoEH6/PPPz5ihR48eSkxMVO/evdXY2KgdO3YEnrN3796QzkXxXTDV1dV972NbEmEBAIg4KSkpuvPOO1VYWKgPPvhAZWVlmjRpkmJjv92s9erVSxMmTNDtt9+u119/XeXl5dq6dauKi4v197//XZI0e/Zsbdu2Tb/61a/02Wefac+ePSopKdGJEyfUrl073XvvvSosLNS7776rzz//XHfffbdcLpfuvPNOSdIvf/lL7d+/X4WFhdq7d69Wrlyp5cuXB815//336+OPP9bUqVNVWlqq/fv368033wwcvHnZZZfpmmuu0T333KMtW7Zox44duuuuu866N6WmpkbHjh3T0aNHtXXrVhUWFiorK0vDhw9vxnc6fHwUAgA4q9MuV6tezxNPPCGn06nx48crNTVVv/nNb1RdXR34+bJly/TYY4/pN7/5jQ4fPqyOHTvq3/7t33TddddJ+jY+3n//fT3wwAMaMmSI7Ha7hg4dqltuuUWStGDBAvl8Pt12222qqanR4MGD9d577wVOSNWtWze99tprmjFjhp555hkNGTJE8+fP1+TJkwMzDBgwQOvXr9eDDz6oH/3oR/L7/br00kt10003Bc1511136eqrr1bnzp312GOPqaio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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "HMXPAjSNp4k8"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}