# Applicabiliy domain for protein language models
This repository contains the code and datasets developed for my Bachelor’s Thesis, completed within the Bachelor’s Degree in Bioinformatics (BDBI) at ESCI-UPF in Barcelona, in collaboration with Nostrum Biodiscovery.


# Project Overview
This work explores the applicability domain of UniRef dataset-trained protein language models (pLMs). With ESM2 Transformer models of different sizes (8M, 35M, 150M parameters), we examine the influence of model size, dataset redundancy (UniRef50 vs UniRef90), and training time on generalization. MGnify sequences grouped into homology bins (40–60%, 60–80%, etc.) are the test cases used.

We trained the pLMs on masked language modeling objectives and obtained embeddings from parallel GPU jobs. We calculated the Cosine similarity, Fréchet ChemNet Distance, Mahalanobis distance, Perplexity score and the autoencoder-based unfamiliarity score to determine whether the test sample falls in the applicability domain of the model.
