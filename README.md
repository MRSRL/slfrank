# SLfRank: Shinnar-Le-Roux Pulse Design using Rank Factorization

This repo contains code for the SLfRank pulse design algorithm. The Shinnar-Le-Roux (SLR) algorithm is widely used to design frequency selective pulses with large flip angles. We improve its design process to generate pulses with lower energy (by as much as 26%) and more accurate phase profiles.

The code depends on SigPy and CVXPy.
You can install dependent libraries by running

     pip install -r requirements.txt

To get started, you can run the demo notebook `demo.ipynb` locally or `demo_colab.ipynb` through Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MRSRL/slfrank/blob/master/demo_colab.ipynb)
