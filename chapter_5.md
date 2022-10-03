extracted from  [Probabilistic Programming and Bayesian Methods for Hackers Chapter 5][1]

### Loss Functions 
A loss function is a function of the true parameter, and an estimate of that parameter:

$$L( \theta, \hat{\theta} ) = f( \theta, \hat{\theta} )$$

The important point of loss functions is that it measures how bad our current estimate is: the larger the loss, the worse the estimate is according to the loss function. A simple, and very common, example of a loss function is the squared-error loss:

$$L( \theta, \hat{\theta} ) = ( \theta -  \hat{\theta} )^2$$

another very popular loss function is the log-loss function
$$L( \theta, \hat{\theta} ) = -\theta\log( \hat{\theta} ) - (1- \theta)\log( 1 - \hat{\theta} ), \; \; \theta \in {0,1}, \; \hat{\theta} \in [0,1]
$$

### in the real world 

In Bayesian Inference we assume that every parameter is a random variable with a prior and a posterior, therefore there is nothing like a "true value" just a few "realizations". Given that we will be focused in estimating the `Expected Loss` rather than a Loss estimation based in a point estimate (or just one sample from the posterior).





[//]: # (References)
[1]: <https://nbviewer.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter5_LossFunctions/Ch5_LossFunctions_PyMC3.ipynb#Loss-Functions>