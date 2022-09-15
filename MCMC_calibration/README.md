# Calibrating compartmental models of infection dynamics using Markov chain Monte Carlo methods

## Purpose
This module is an implementation of Markov chain Monte Carlo methods for a deterministic SEIRD model and worked on time-dependent infection rate functions to analyze the evolution of the COVID-19 outbreak in different parts of the world, and in Saint Petersburg, Russia. 

## Data
To put the model to the test, we employed two separate datasets. One file has data records for COVID-19 epidemics in Saint Petersburg, Russia, and the second file has data records for COVID-19 epidemics in 225 countries.

### Saint Petersburg 
The data file "spb.combined.daily.txt" forms of COVID-19 pandemic records from St. Petersburg, Russia. As of 2022-03-17, this file has daily numbers of covid-19 data [1].
The data file contains tab-delimited values for:
* "TIME" : text string in YYYY-MM-DD format;
* "CONFIRMED" : integer, number of confirmed COVID-19 cases for a given day;
* "RECOVERED" : integer, number of recovered for a given day;
* "DEATHS" : integer, number of deaths for a given day;
* "ACTIVE" : integer, number of active cases (derived from cumulative CONFIRMED - (cumulative RECOVERED + cumulative DEATHS)) for a given day;
* "CONFIRMED.spb" : integer, number of confirmed cases according to the City Council for a given day;
* "HOSPITALIZED.today" : integer, number of hospitaized for a given day;
* "PCR.tested" : integer, number of PCR-tests performed for a given day;
* "v1.CS" : integer, number of vaccines administered (1st dose), cumulated sum to date;
* "v2.CS" : integer, number of vaccines administered (2nd dose), cumulated sum to date;
* "Yandex.ACTIVITY.points" : numeric, Yandex overall activity point for a given day;

### World
The data file "worldometer_coronavirus_daily_data.csv" contains daily numbers of covid-19 data for each of the 225 countries. From 2020-2-15 to 2022-05-14, all countries have records (820 days per country). With the exception of China, which has data dating from 2020-1-22 to 2022-05-14 (844 days per nation), and Palau, which has records dating from 2021-8-25 to 2022-05-14 (844 days per country) (263 days per country) [2].
The data contains comma separated values for:
* "date": Date of observation of the row’s data in YYYY-MM-DD format.
* "country": Country in which the the row’s data was observed.
* "cumulative total cases": integer, Cumulative number of confirmed cases as of the row’s date, for the row’s country.
* "daily new cases": integer, daily new number of confirmed cases on the row’s date, for the row’s country.
* "active cases": integer, Number of active cases (i.e., confirmed cases that still didn’t recover nor die) on the row’s date, for the row’s country.
* "cumulative total deaths": integer, Cumulative number of confirmed deaths as of the row’s date, for the row’s country.
* "daily new deaths": integer, daily new number of confirmed deaths on the row’s date, for the row’s country.


## Setup
* pip install -r requirements.txt
Install the required libraries using requirments.txt file

## Description
The Metropolis-Hastings procedure is an iterative algorithm. To carry out the
Metropolis-Hastings algorithm, we need to draw random samples from the following
distributions. For this experiment, pymcmcstat, A Python Package for Bayesian Inference
Using Delayed Rejection Adaptive Metropolis is used [3]. This module is intended to be
the main class from which to run these Markov Chain Monte Carlo type simulations. The
user will create an MCMC object, initialize data, simulation options, model settings and
parameters [4].

The notebook "Multiple covid phases SPB.ipynb" contains the proposed deterministic SEIRD model which analyzes the evolution of the COVID-19 outbreak in Saint Petersburg, Russia during multiple phases. and the notebook "Multiple covid phases-world.ipynb" analyzes the evolution of the COVID-19 outbreak in 3 different nations (Russia, India, Bolivia) and for the whole world data during multiple phases.

## References
[1] Kouprianov, Alexei. — COVID-19.SPb. Coronavirus epidemics in St.
Petersburg, Russia: data and scripts, 2022. — data and R code.
https://github.com/alexei-kouprianov/COVID-19.SPb.

[2]ASSAKER, JOSEPH. — Covid-19 Global Dataset, 2022.
https://www.kaggle.com/datasets/josephassaker/covid19-globaldataset?select=worldometer_coronavirus_ daily_data.csv.

[3] Miles, Paul R. pymcmcstat: A Python Package for Bayesian Inference Using
Delayed Rejection Adaptive Metropolis / Paul R. Miles // Journal of Open Source
Software. — 2019. — Vol. 4, no. 38. — P. 1417. https://doi.org/10.21105/joss.
01417.

[4] pymcmcstat package, url = https://pymcmcstat.readthedocs.io/en/latest/pymcmcstat.hturldate = 2018.


