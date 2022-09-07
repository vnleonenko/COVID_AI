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

### GPR for ABM-model

### References
[1] Sankaran, Sethuraman, and Alison L. Marsden. (2011) "A stochastic collocation method for uncertainty quantification and propagation in cardiovascular simulations." Journal of biomechanical engineering 133(3).

[2] The Kernel Cookbook: Advice on Covariance functions [Online]. – URL: https://www.cs.toronto.edu/~duvenaud/cookbook/ (accessed: 20.06.2022).

[3] Rasmussen, Carl Edward. "Gaussian processes in machine learning." Summer school on machine learning. Springer, Berlin, Heidelberg, 2003.

[4] Leonenko, Vasiliy, Sviatoslav Arzamastsev, and Georgiy Bobashev. (2020) "Contact patterns and influenza outbreaks in Russian cities: A proof-of-concept study via agent-based modeling." Journal of Computational Science 44: 101156.

