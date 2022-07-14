# Bayesian Methods for hackers - PYMC4 snippets  
This are some of the replications of the code finded in the book [Probabilistic Programming and Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) the last version of the book was adapted to pymc3, but the library has suffered strong changes in the classes and implementations. I have the physical copy and is impossible to follow. I use the `pymc3` notebooks available in the github repository. I left some of the general findings of changes in the following list:

## Notes in Changelog from pymc3 to pymc4

this are some changes in the book from pymc3 to pymc4

- `testval` parameter is replaced by `initval`
- `theano.tensor` (`tt`) its replaced by  `aesara.tensor` (`at`)
- `test_value` is (apparently) not longer working. No idea if there is a way to get a value without sampling first. 
- to get values from `trace` from `trace = pm.sample(10000, **args)`  use `trace.posterior['var'].values[0]` - old way `trace['var'][15000:]` - 
- try not to use `pm.Metropolis()` [results are not very consistent] use instead `pm.NUTS()` is a different sampling algorithm, but seems to be more consistent than the other algorithm. 


