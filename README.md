# Deep Variational Koopman Models
Source code for "Deep Variational Koopman Models: Inferring Koopman Observations for Uncertainty-Aware Dynamics Modeling and Control" from IJCAI 2019. The paper can be found [here](https://arxiv.org/pdf/1902.09742.pdf).

## Overview
A description of the individual files is given below.
### ```training``` Directory
* ```koopman_model.py``` - script for defining architecture of and constructing Deep Koopman models for training.
* ```train_koopman.py``` - training script for Deep Koopman models.
* ```bayes_filter.py``` - script for defining architecture of and constructing Deep Variational Bayes Filter models.
* ```train_bayes_filter.py``` - training script for Deep Variational Bayes Filter models.
* ```dataloader.py``` - script for loading and processing data prior to training.
* ```utils.py``` - contains functions for evaluating trained models.
* ```find_matrices.py``` - script to load a trained neural network model and determine the B-matrix, action normalization parameters, and goal state encoding.
* ```find_dynamics.py``` - script to load a trained neural network model and output the current state encoding and the A-matrix based on the previous sequence of observed states and actions.
