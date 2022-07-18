# Bayesian Methods for hackers - PYMC4 snippets  
This is a collection of some snippets founded in the book [Probabilistic Programming and Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) ported to `pymc4` (`pymc==4.1.2`). The last version of the book (available [online](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers#pymc3)) was implemented on `pymc3`, but the library has suffered strong changes in the classes and implementations since then. I have a physical copy of the book and is impossible to follow. I use the `pymc3` notebooks available in the github repository. 

## Notes in Changelog from pymc3 to pymc4

this are some changes in the API from pymc3 to pymc4 that I've being using to port the code from the book:

- `testval` parameter is replaced by `initval`
- `theano.tensor` (`tt`) its replaced by  `aesara.tensor` (`at`)
- `test_value` is (apparently) not longer working. No idea if there is a way to get a value without sampling first. 
- to get values from `trace` from `trace = pm.sample(10000, **args)`  use `trace.posterior['var'].values[0]` - old way `trace['var'][15000:]` - 
- try not to use `pm.Metropolis()` [results are not very consistent] use instead `pm.NUTS()` is a different sampling algorithm, but seems to be more consistent than the other algorithm. 


# Some Bayesian Inference concepts

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

<p align="center"><img src="img/bayesian_updating.png" alt="drawing" width="400"/></p>


### Notes about doing inference with samples (extracted from [Ch2-3](https://github.com/dccuchile/CC6104/blob/master/slides/3_2_ST-posterior.pdf)) 

If we know the posterior shape, for example after using a conjugate priors, we can estimate easily the areas under the curve using the formula, Although, Given that usually we can't know for sure the shape of the posterior we have to make some hypothesis test over the **sampling** of the posterior, a few common tests are the followings: 

1. What is the probability than $\theta \le \theta_{ub}$ ? $\mathbb{P}(\theta \le \theta_{ub})$. 
    
    We count the number of elements in the `trace` that are lower than $\theta_{ub}$ and divided by the elements in the `trace`. Therefore `p ~ len(trace<=ub)/len(trace)`

2. What is the probability that $\theta_{lb} \le \theta \le \theta_{ub}$ ?. $\mathbb{P}(\theta \in [\theta_{lb},\theta_{ub}])$.
 
    We use the estimation of (1) to estimate $\mathbb{P}(\theta \le \theta_{ub})$ and $\mathbb{P}(\theta \le \theta_{lb})$ and then our new estimation $\mathbb{P}(\theta \in [\theta_{lb},\theta_{ub}])$ will be simply the difference between both probabilities

    $$\mathbb{P}(\theta \in [\theta_{lb},\theta_{ub}]) = \mathbb{P}(\theta \le \theta_{ub}) - \mathbb{P}(\theta \le \theta_{lb}) $$


3. What is the $\theta_{ub}$ that we know with $p$ probability that $\theta \le \theta_{ub}$ ?. Find $\theta_{ub}: \mathbb{P}(\theta \le \theta_{ub}) = p$

    We calculate the `np.quantile` where `q=p` for the trace, and that's our $\theta_{ub}$. `qub ~ np.quantile(trace, q=p)`

4. In the bayesian world the **credible intervals** are the equivalent of the *confidence intervals*  in the classical world. They have a very different interpretation thought:

    -  A confidence interval is a region (that will vary depending on the sample). that after infinitely repeating the data sampling experiments will contain the true parameter  $(1-\alpha)$ percentage of the time. 

    - In contrast, a **credible interval** is a range of values that **we believe** our parameter can take with a certain probability according to both our prior beliefs and the evidence given by the data.

    The determination procedure for a **credible interval** varies given the posterior shape, if the shape is "symmetrical" using an **equally tailored interval** will be enough. 

    An  **equally tailored interval**  is estimated using the procedure in (3). we basically get both bounds of the interval estimating the tails sizes $\alpha/2$. For example, if $90\% = 1-\alpha$ is our *confidence (credible?) level* ($\alpha = 10\%$, $\alpha/2 = 5\%$ ). therefore we estimate $\theta_{lb}$ = `np.quantile(trace, q=0.05)` and our  $\theta_{ub}$ = `np.quantile(trace, q=0.95)`. Therefore our **credible interval** $CRI = [\theta_{lb}, \theta_{ub}]$
     
    Nevertheless, sometimes (and very often), our distribution is non-symmetrical or very skew, and is possible that a symmetrical **credible interval** might not even contain the most likely value of $\theta$ (for example when true $\theta$ is in a extreme). In this case we use the **Highest Posterior Density Interval (HPDI)**. The concept behind that interval is basically get *all the possible intervals* with area $1-\alpha$ and choose the one with the narrower interval ($\theta_{ub}-\theta_{lb}:argmin$). 

    when the `posterior` is not skew, HDPI and symmetrical intervals are very similar.

5. What is the mode ?. The highest probability value, also known as the maximum a posteriori (MAP)
    
    `mp = pymc.find_MAP()` apparently this should be called after the `pm.sample()` altho I could't find a "correct" example, apparently this is another "minor" change in the API. Also [other alternatives](https://stackoverflow.com/q/22284502/5318634)

6. What is the mean ? 
    
    `mean = np.mean(trace)`

7. What is the median ? 
    
    `mean = np.median(trace)`
   
   When the posterior is gaussian (or similar), the three indicators (`mean`, `mode`, `median`) tends to be very aligned. that usually don't happen when the distribution is skew. 

    <p align="center"><img src="img/mean_mode_median.png" alt="drawing" width="400"/></p>


### Generating "Predictions"

Usually we want to generate simulated data ($\widetilde{d}$) in order to validate our model, and additionally, to get predictions. We want to use our posterior information ($f(\theta|d)$) to do so.

A simple approach would be to use a point estimator of $\theta \sim f(\theta|d)$, for example MAP ($\theta_{MAP}$), insert it on the `likelihood` function $f(d|\theta_{MAP})$, and finally, using that distribution to generate the new data $\widetilde{d}$. The problem with that procedure is that we will ignore the uncertainty in $\theta$ that we have learn from `posterior`. we would like to include that uncertainty in our simulation.  

What we can do is to use many $\theta_{i}$ values , and use them to simulate data using the `likelihood` function $\widetilde{d_i} \leftarrow f(d|\theta_{i})$, then, we would **weight** those simulations $\widetilde{d_i}$ with the probability of getting those  $\theta_i$ from the `posterior`. We call this new distribution the **posterior predictive distribution**, mathematically will be something like $f(\widetilde{d}|d) =\int_{\theta_i} f(\widetilde{d}|\theta_i)f(\theta_i|d) \partial \theta_i$. We basically have an estimation on how likely we can obtain a value $\widetilde{d_i}$ given our posterior. 

An alternative process to estimate the **posterior predictive distribution** is to use a `trace` from the `posterior` (a vector of $\theta_i$'s), and use them to get samples from the `likelihood` function $\widetilde{d_i} \leftarrow f(d|\theta_{i})$ and then simply average them, given that the $\theta_i$ values where already draw using the posterior. 