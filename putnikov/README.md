# Physics-informed neural network with compartment model differential equation

## Purpose
This module is an implementation of physics-informed neural network approach for SEIR compartment model calibration.
It contains neural network abstraction and usage example.

## Data

Data contains daily statistics of the first COVID-19 wave in Saint-Petersburg divided into compartments. First wave starts at 2020-03-04 and finished at 2020-06-30.


## Description of content

* "seirpinn" folder contains source code of neural network
* "example.ipynb" contains usage example
* "requirements.txt" contains dependencies list

## Setup venv
* python3 -m venv venv
* source venv/bin/activate
* pip install -r requirements.txt

## Usage
Look at example.ipynb