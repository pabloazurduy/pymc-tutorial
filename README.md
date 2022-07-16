# Bayesian Methods for hackers - PYMC4 snippets  
This is a collection of some snippets founded in the book [Probabilistic Programming and Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) ported to `pymc4` (`pymc==4.1.2`). The last version of the book (available [online](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers#pymc3)) was implemented on `pymc3`, but the library has suffered strong changes in the classes and implementations since then. I have a physical copy of the book and is impossible to follow. I use the `pymc3` notebooks available in the github repository. 

## Notes in Changelog from pymc3 to pymc4

this are some changes in the API from pymc3 to pymc4 that I've being using to port the code from the book:

- `testval` parameter is replaced by `initval`
- `theano.tensor` (`tt`) its replaced by  `aesara.tensor` (`at`)
- `test_value` is (apparently) not longer working. No idea if there is a way to get a value without sampling first. 
- to get values from `trace` from `trace = pm.sample(10000, **args)`  use `trace.posterior['var'].values[0]` - old way `trace['var'][15000:]` - 
- try not to use `pm.Metropolis()` [results are not very consistent] use instead `pm.NUTS()` is a different sampling algorithm, but seems to be more consistent than the other algorithm. 


## Some Bayesian Inference concepts

The main goal of the bayesian inference is to estimate a `posterior` $f(\theta|d)$ given that we have a `prior` distribution $f(\theta)$ (over the parameters) and `likelihood` distribution too $f(d|\theta)$ and a `evidence` or `average likelihood` $f(d)$

$$f(\theta|d) =\frac{f(d|\theta)*f(\theta)}{f(d)}$$

Where $f(d) = \sum_{\theta}f(d|\theta)*f(\theta)$ which is a "normalization" value over the posterior "space". We use $f()$ to avoid the use of $IP()$ but, conceptually is just a probability (not exactly* but "almost"). 

One difference from the bayesian inference to the "classical" statistical inference is that we are looking into the "distribution" of the parameter $f(\theta|d)$(`posterior`) and not assuming that there is a "real" value $\theta$ and we are looking into an estimator of that "real parameter" $\hat{\theta} \sim \theta$. Therefore our posterior will provide us with a distribution of the parameter, rather than just an estimator. 

Usually in the bayesian inference problems we have a random process and we follow the next steps:
1. we have some prior beliefs on the distributions that generate that process and some priors over the parameters. we call all of them $f(\theta)$ our `priors` (this priors can be nested)
2. given our "model" of the reality we can estimate a `likelihood` function over the data $f(d|\theta)$ this usually describe how likely is to get our data given that we have a prior from our priors $f(\theta)$
3. We usually don't know the shape or form of the `posterior` distribution, except when we have [conjugate distributions](https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions) with the `prior` and `likelihood`.If we  don't have an analytical solution we still can sample from it using `mcmc` #magic. 
4. we don't care much about the "`evidence`" function mainly because is hard to estimate and there are some properties over `mcmc` that allow us to sample from a "proportionally similar" function rather than the original distribution too and it will [guarantee convergency too](https://github.com/dccuchile/CC6104#part-iii-bayesian-inference). 



We can identify this components in a simple implementation with `pymc` 

```python
import pymc as pm
# ======================= #
# Simple model declaration 
# ======================= #
with pm.Model() as model:
    theta = pm.Exponential("poisson_param", 1.0, initval=0.5) # prior
    data_generator = pm.Poisson("data_generator", theta) # likelihood function (the poisson will describe the data distribution given the parameter value) 
    # we sometimes (usually) don't know the shape of the posterior 
    # but we can sample from it using mcmc 
    trace = pm.sample(1000) # samples of the posterior using MCMC 

```

Even when we set a prior distribution and that assumes some model of the reality, the posterior can look very different -it can be shaped over the weight of the "data" or evidence - and it will be converge too, maybe more slow, but it will be shaped too.  

Actually, in bayesian inference the process of using new data to re-estimate the posterior is called [`bayesian update`](https://github.com/dccuchile/CC6104/blob/master/slides/3_1_ST-bayesian.pdf), and this process can be run indefinitely using the last estimated `posterior` as the new `prior`, and generate a `new posterior` base on it. 

<img src="img/bayesian_updating.png" alt="drawing" width="400"/>