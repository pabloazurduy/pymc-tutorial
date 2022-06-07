import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
plt.pause(1) # fix plots 

# ======================= #
# Simple model declaration 
# ======================= #
with pm.Model() as model:
    parameter = pm.Exponential("poisson_param", 1.0, initval=0.5)
    data_generator = pm.Poisson("data_generator", parameter)
    data_plus_one = data_generator + 1
    # idata = pm.sample(1000, tune=1000, cores=1)

# plot results 
# az.plot_trace(idata, combined=True)
# az.summary(idata, round_to=2)

with pm.Model() as model:
    lambda_1 = pm.Exponential("lambda_1", 1.0)
    lambda_2 = pm.Exponential("lambda_2", 1.0)
    tau = pm.DiscreteUniform("tau", lower=0, upper=10)
    new_deterministic_variable = lambda_1 + lambda_2
    # idata = pm.sample(1000, tune=1000, cores=1)

# =============================== #
# Deterministic tracking activate 
# =============================== #
def subtract(x, y):
    return x - y

with pm.Model() as model:
    stochastic_1 = pm.Uniform("U_1", 0, 1)
    stochastic_2 = pm.Uniform("U_2", 0, 1)
    det_1 = pm.Deterministic("Delta", subtract(stochastic_1, stochastic_2))
    idata = pm.sample(1000, tune=1000, cores=1)
az.plot_trace(idata, combined=True)

# ===================== #
# use of aesara tensor 
# ===================== #
import aesara.tensor as at

with pm.Model() as theano_test:
    p1 = pm.Uniform("p", 0, 1)
    p2 = 1 - p1
    p = at.stack([p1, p2]) 
    assignment = pm.Categorical("assignment", p) #pair of probabilities for our two categories p=(p1,p2)
    #idata = pm.sample(1000, tune=1000, cores=1)
#az.plot_trace(idata, combined=True)

# We're using some fake data here
import numpy as np
data = np.array([10, 25, 15, 20, 35])
with pm.Model() as model:
    lambda_1 = pm.Exponential("lambda_3", 1.0)
    lambda_2 = pm.Exponential("lambda_4", 1.0)
    idx = np.arange(len(data)) # Index
    tau = pm.DiscreteUniform("tau2", lower=0, upper=len(data) - 1)
    lambda_ = pm.math.switch(tau >= idx, lambda_1, lambda_2)
    obs = pm.Poisson("obs", lambda_, observed=data)
    #idata = pm.sample(1000, tune=1000, cores=1)
#az.plot_trace(idata, combined=True)
