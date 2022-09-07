## Application of Gaussian process regression as a surrogate modeling method to assess the dynamics of COVID-19 propagation

### Gaussian Process Regression

Gaussian Process Regression (GPR) is used as a surrogate to minimize the time costs of the agent-based model, as well as to demonstrate the possibility of estimating the dynamics of COVID-19 propagation with different sets of input parameters.

Gaussian Process Regression is a class of supervised machine learning algorithm, for which it is sufficient to use a small number of parameters to make a prediction. Covariance functions are an important component of GPR models because these functions weigh the contribution of training points to the predicted test targets according to the kernel distance between the observed training points and test points [1]. The following covariance functions were considered in this project [2]:

•	Rational Quadratic (RQ) kernel

•	Squared Exponential (SE) / Radial-basis Function (RBF) kernel 

•	Multidimentional Product of RQ and RBF

•	Additive kernel (combination of RQ and RBF)

Squared Exponential Covariance function is the Gaussian kernel. SE defined by the formula [2]:

$$k_{ SE }(x,x')=\sigma^2(exp(-\dfrac{(x-x')^2}{2l^2})$$
where:

$l$ is the lengthscale determines the distance you have to move in input space before the function value can change significantly [3]

$\sigma^2$ – the output variance determines the average distance of your function away from its mean

$x$ – observed training points

$x'$ – test points 

Rational Quadratic Covariance function can be seen as a scale mixture (an infinite sum) of squared exponential covariance functions with different characteristic length-scales [3]. RQ is defined by formula:

$$k_{ RQ }(x,x')=\sigma^2(1+(\dfrac{(x-x')^2}{2{\alpha}l^2})^{-\alpha}$$

### Synthetic population and sampling

‘Synthetic population’ is a synthesized, spatially explicit human agent database representing the population of a city, a region or a country. By its cumulative characteristics, this database is equivalent to the real population but its records are not correspondent to real people [4]. The initial synthetic population of St. Petersburg used in the simulation consists of several files described in Table 2.

Table 2. File structure of a synthetic population for Saint Petersburg.
| File        | Contents           |
| ------------- |:-------------:|
| people_workplaces.txt     | Records for each person, along with their age and gender and information about workplaces for each agent |
| households.txt      | The location and descriptive attributes for each household      |
| schools.json  | Records for each school (dictionary)      |

![image](https://user-images.githubusercontent.com/59513334/188823077-6b6e301a-92c2-4c0b-b366-263dea6416bd.png 'Fig. 1. Schema of tables and field dependencies.')

Fig. 1. Schema of tables and field dependencies.

Sampling algorithm:
1. From the list of all agents “people_workplaces.txt” randomly select the required number of agents
2. From the file “households.txt” select the houses corresponding to the agent IDs selected in the first paragraph
3. Similarly, from the “schools.json” file, select the schools that the agents attend.

### GPR for SEIRD-model

The SEIRD model (Susceptible-Exposed-Infected-Recovered-Deceased) is the extended version of the SEIR model [7]. The populations’ dynamics is described by the following system of differential equations [8]:

$$\begin{cases}
      \dfrac{dS(t)}{dt}=\dfrac{-{\beta}S(t)I(t)}{N}\\
      \dfrac{dE(t)}{dt}=\dfrac{-{\beta}S(t)I(t)}{N}-{\beta}E(t)\\
      \dfrac{dI(t)}{dt}={\delta}E(t)-(\gamma+\mu)I(t)\\
      \dfrac{dR(t)}{dt}={\gamma}I(t)\\
      \dfrac{dD(t)}{dt}={\mu}I(t)
    \end{cases}\,$$
    
where $S(t)$ – the number of susceptible individuals at time $t$;

$I(t)$ – the number of infected individuals at time $t$;

$R(t)$ – the number of recovered individuals at time $t$;

$E(t)$ – the number of exposed individuals at time $t$;

$E(t)$ – the number of dead individuals at time $t$;

$\beta$ – effective contact rate;

$\mu$ – mortality rate;

$γ$ – recovery rate.


### GPR for ABM-model

Multiagent models, also known as agent-based models (ABM), have some limitations despite all their advantages, in particular, high-level complexity of parameters, long execution time, and complexity of model analysis. As the complexity of agent-based models increases, the number of parameters required to be assessed on real data grows. Due to the presence of stochastic processes for model calibration, as well as a need for uncertainty and sensitivity analysis [5], it is necessary to conduct many simulation launches, which leads to increased time consumption.

In this work an agent-based framework from [6] was used. The basic principle of simulation is as follows: each agent in the population potentially interacts with other agents if they attend the same school (for schoolchildren), workplace (for working age adults), or lives in the same household. The infectivity of each agent depends on their day of infection [4]. The modeling step of this model is equal to one day. Agents are randomly selected from the general population and are assigned an infectious status at the beginning of the simulation. Step by step algorithm is described in [4].
The main input parameters of the model which are important for uncertainty and sensitivity analysis are introduced in Table 1.


### References
[1] Sankaran, Sethuraman, and Alison L. Marsden. (2011) "A stochastic collocation method for uncertainty quantification and propagation in cardiovascular simulations." Journal of biomechanical engineering 133(3).

[2] The Kernel Cookbook: Advice on Covariance functions [Online]. – URL: https://www.cs.toronto.edu/~duvenaud/cookbook/ (accessed: 20.06.2022).

[3] Rasmussen, Carl Edward. "Gaussian processes in machine learning." Summer school on machine learning. Springer, Berlin, Heidelberg, 2003.

[4] Leonenko, Vasiliy, Sviatoslav Arzamastsev, and Georgiy Bobashev. (2020) "Contact patterns and influenza outbreaks in Russian cities: A proof-of-concept study via agent-based modeling." Journal of Computational Science 44: 101156.

[5] Perumal, Rylan, and Terence L. van Zyl. "Surrogate assisted methods for the parameterisation of agent-based models." 2020 7th International conference on soft computing & machine intelligence (ISCMI). IEEE, 2020.

[6] Influenza_spatial: A spatial model for the spread of influenza [Online]. – URL: https://github.com/vnleonenko/Influenza_spatial (accessed: 15.01.2022).

[7] Li M. Y. An introduction to mathematical modeling of infectious diseases. – Cham : Springer, 2018. – Vol. 2. – P. 34.

[8] Manik S. et al. Impact of climate on COVID-19 transmission: A study over Indian states //Environmental Research. – 2022. – Vol. 211. – P. 113110.


