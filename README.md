# COVID_AI

Libraries for COVID-19 modeling by means of coupling the classical approaches with AI algorithms.

The detailed description of each algorithm along with the examples is placed into the folders, please check corresponding readme files for more information.

Short description of folders:

Data preparation - includes the script for restoring data on disease incidence from several sources. 

The data mining method was developed and implemented based on statistical models such as the moving average model (MAM). The use of relatively simple approaches was caused by the need for a generalized approach that does not rely on the subject area and the specifics of collecting specific data arrays. A test example of using the preliminary data analysis method consists of processing of COVID-19 incidence data. Due to discrepancies in the number of new active cases of COVID-19 received from different sources (Internet resource "Stopcoronovirus.rf" and reports of the Interdepartmental City Coordinating Council), a new array of incidence data is generated based on two available ones, which is used as an alternative when calibrating models. The points of the new array are chosen as the maxima of the values of the two original morbidity trajectories, while the data from the reports of the Interdepartmental City Coordinating Council are preliminarily smoothed.

MCMC_calibration - includes MCMC calibration algorithms for SEIR models.

This module is an implementation of Markov chain Monte Carlo methods via Metropolis-Hastings procedure for a deterministic SEIRD model and worked on time-dependent infection rate functions to analyze the evolution of the COVID-19 outbreak in different parts of the world. 

The Metropolis-Hastings procedure is an iterative algorithm. To carry out the Metropolis-Hastings algorithm, we need to draw random samples from the following distributions. For this experiment, pymcmcstat, A Python Package for Bayesian Inference Using Delayed Rejection Adaptive Metropolis is used. This module is intended to be the main class from which to run these Markov Chain Monte Carlo type simulations. The user will create an MCMC object, initialize data, simulation options, model settings and parameters.

The notebook "Multiple covid phases SPB.ipynb" contains the proposed deterministic SEIRD model which analyzes the evolution of the COVID-19 outbreak in Saint Petersburg, Russia during multiple phases. and the notebook "Multiple covid phases-world.ipynb" analyzes the evolution of the COVID-19 outbreak in 3 different nations (Russia, India, Bolivia) and for the whole world data during multiple phases.

PINN_calibration - includes algorithms for model calibration using PINN approach.

Thу module is an implementation of physics-informed neural network approach for SEIR compartment model calibration. It contains neural network abstraction and usage example. Physics-informed neural network (PINN) is a neural network which is trained to solve supervised learning tasks respecting physics laws described by general nonlinear partial differential equations. Partial differential equations usage is facilitating the learning algorithm to capture the right solution even with a low amount of training examples. The network then seeks to minimize the mean squared error of the loss function by utilizing Adam optimization method used in conjunction with PyTorch software. The test file is named example.ipynb.

Surrogate modeling - contains scripts for surrogate modeling of compartmental and multiagent models.

In this project we demonstrate the possibility of using Gaussian Process Regression as a surrogate modeling method to assess the dynamics of COVID-19 propagation. The models were trained on data obtained by multiple launches of SEIRD and ABM models.

Gaussian Process Regression (GPR) is used as a surrogate to minimize the simulation time, as well as to demonstrate the possibility of estimating the dynamics of COVID-19 propagation with different sets of input parameters.

Gaussian Process Regression is a class of supervised machine learning algorithm, for which it is sufficient to use a small number of parameters to make a prediction. Covariance functions are an important component of GPR models because these functions weigh the contribution of training points to the predicted test targets according to the kernel distance between the observed training points and test points
