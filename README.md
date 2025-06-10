# Applicabiliy domain for protein language models
This repository contains the code and datasets developed for my Bachelor’s Thesis, completed within the Bachelor’s Degree in Bioinformatics (BDBI) at ESCI-UPF in Barcelona, in collaboration with Nostrum Biodiscovery.


# Project Overview
This work explores the applicability domain of ESM2 protein language models (pLM) trained with UniRef dataset and tested with MGnify sequences grouped into homology bins (40–60%, 60–80%, etc.).Our objective is assesing the influence of model size (8M,35M,150M and 650M parameters), dataset redundancy (UniRef50 vs UniRef90) and training time. To evaluate it, different metrics have been used: Cosine similarity, Fréchet ChemNet Distance, Mahalanobis distance, Perplexity score and the autoencoder-based unfamiliarity score.

# Setup

**Requirements**
Install the required dependencies from requirements.yaml:
conda env create -f requirements.yaml

**Repository structure**
models/: Baseline and our models
data/: Subset datasets used in our research.
code/: Code for the autoencoder, embeddings obtention, esm2 model, metrics obtention and plots.

# Workflow

Pipeline for metrics obtention for 8M our model

**1. Train 8M esm model**
   
   1.Log into your Weights and Biases account to track model loss:
   
   ```ruby
     wandb login <your-API-key>
   ```
   
   2.Execute esm2_partitioned_execution_8M.sh as executes esm2_job_splitter that splits the model execution into 10        splits. Then each job will execute the esm2_model.py

   ```ruby
      sbatch esm2_partitioned_execution_8M.sh
   ```

**2. Obtain train embeddings**

   1. Execute count_sequences.py to know the number of sequences you need to specify in                                     parallelize_embedding_extraction_8M_our_model.sh

      ```ruby     
         python count_sequences.py
      ```
        
   3. Execute parallelize_embedding_extraction_8M_our_model.sh as has all the hyperparameters prepared to extract           the training embeddings by splitting dataset into different jobs.

      ```ruby
         sbatch parallelize_embedding_extraction_8M_our_model.sh
      ```
   5. Execute per_job_compress_train_embeddings_8M_our_model.sh as has all the hyperparameters to merge all the             embeddings created by 8M our model into 10 numpy arrays.

      ```ruby
         sbatch per_job_compress_train_embeddings_8M_our_model.sh
      ```
      
   7. Execute joining_jobs_compress_train_embeddings_8M.sh as has all the hyperparameters to merge the 10 numpy             arrays into 1

      ```ruby
         sbatch joining_jobs_compress_train_embeddings_8M.sh
      ```
      
**3. Obtain validation and test embeddings**

   1. Execute validation_testing_embedding_8M.sh with the hyperparameters it has.

      ```ruby
         sbatch validation_testing_embedding_8M.sh
      ```

**4. Train the autoencoder**

   1. Once you have the array that contains all the embeddings, run autoencoder_parallelized_8M.sh to train the          autoencoder with training embeddings.

      ```ruby
         sbatch autoencoder_parallelized_8M.sh
      ```
      
**5. Compute metrics**

   1. Execute FCD_embeddings.sh. Note: You need to change the input dimension to the corresponding embedding             dimension. For example for 8M is 320.

      ```ruby
         sbatch FCD_embeddings.sh
      ```   
   
   3. Execute perplexity_calculation_our_model.sh

      ```ruby
         sbatch perplexity_calculation_our_model.sh
      ```
   
   In both output files you will obtain the metrics results for 8M our model.
