# TO-DOs:
Last edit: 30.01.26 Friday
- Model: y = \beta0 + \beta1 * CF_k^n + \sigma \nu to -> f = \beta0 + \beta1 * CF_k^n with x=CF \theta = {\beta_0, \beta_1, k, n} with \nu ~N(0,1))

- I either need to do a grid search to establish these parameters
- or second option: gradient descent


10.02.2026
- Implement z() till next week.
- You need to modify the cochlea simulation pipeline to accept .wavs, but still get PSTH, not the mean-rates.
- You need to also implement running the simlaion i.e. 100 times with different seeds
- another function to get the average of the 100 runs.
- Look for dIPC things, you need the memory there.
