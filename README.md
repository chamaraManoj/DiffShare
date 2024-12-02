# DiffShare
Official repository of the DiffShare paper. This code repository is still under maintenance. There are two main phases in this repo development. During the first phase, we update the repository including the codes and data used in the experiments in DiffShare development. We share the code base used for crypto dataset and this is the same code base used for other dataset with slight modifications to the features used. In the second phase, we provide generic workflow of DiffShare that can be used as a tool on given multivariate time-series data. 

## Phase 1 - Initial implementation of the DiffShare
This phase contains the initial implementation of the DiffShare, including the codes and data of the experiements. The folders/files given contains the following details.

1. requirments.txt: required libraries for the code development
2. data/synth: This folder contains the synthetic data for *Crypto* dataset including the two main classes **normal** and **rugpull** representing non-scam and scame token types respectively in chunk form. These are the direct output of the DNN model, therefore in normalized domain. We have provided the corresponding minmax distribution for those token to reconstruct the data to its original value
3. diffusion\_main\_crypto: Folder contains the DiffShare main data generation.
   - data: `.csv` files containing the token information for both **normal** and **rugpull** data. It includes, token name, chunk id, relative chunk position, previous chunk id which are used by the data loader in pytorch implementation.
   - models: trained models for **normal** adn **rugpull** generation
   - synth\_minmax: minmax data used which is taken as a conditional input for chunk data generation
   - `.py` scripts: all the python scripts corresponding for the data generation
4. diffusion\_minmax\_main\_crypto: codes related to the minmax generation. Corresponding real data is in data/synth folder.
   - trained\_models: trained light weight DM model for **normal** (given as **Norma**) adn **rugpull** (given as **Normal\_rug**) data. Models are given for each feature separartely.
   - `.py` scripts: all the python scripts corresponding for the minmax data generation
5. post-processing steps: contains the codes for post processing steps (chunk mergaing and converting normalized data to the original values)
    
