# Physics-informed neural network with compartment model differential equation

## Purpose
This module is an implementation of physics-informed neural network approach for SEIR compartment model calibration.
It contains neural network abstraction and usage example.

## Description of content

* "seirpinn" folder contains source code of neural network
* "example.ipynb" contains usage example
* "requirements.txt" contains dependencies list

## Setup
* python3 -m venv venv
* source venv/bin/activate
* pip install -r requirements.txt

## Data

Data contains daily statistics of the first COVID-19 wave in Saint-Petersburg divided into compartments. First wave starts at 2020-03-04 and finished at 2020-06-30.

## Theory

### Physics-informed neural networks

Physics-informed neural network (PINN) [1] is a neural
network which is trained to solve supervised learning tasks
respecting physics laws described by general nonlinear partial
differential equations. Partial differential equations usage is
facilitating the learning algorithm to capture the right solution
even with a low amount of training examples.
The network then seeks to minimize the mean squared error
of the loss function by utilizing Adam optimization method
used in conjunction with PyTorch software.

## Neural network parameters

|Option|Description|
|------|-----------|
|NUM_EPOCHS | Amount of learning iterations|
|LEARNING_RATE| Sensitivity of optimization algorithm|
|PARAMETERS_RANGE| Calibration parameter limits |
|NETWORK.nodes| Amount of nodes in neural network per layer |
|NETWORK.hidden_layers_amount| Amount of hidden layers |
|NETWORK.activation_function | Activation function of nodes| 
|LOSS.threshold| Limit of loss function |
|LOSS.balance_delta_U | Balancing factor of NN output |
|LOSS.balance_delta_F | Balancing factor of ODE residuals |


## Usage
Look at example.ipynb

## References

[1] Raissi M., Perdikaris P., Karniadakis G.E., Physics-informed neural
networks: A deep learning framework for solving forward and inverse
problems involving nonlinear partial differential equations // Journal of
Computational Physics. — 2019.— Vol. 378.