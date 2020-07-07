# Cell-state variational autoencoder with temporal integration.

Variational autoencoder implementation using PyTorch for latent cell-state inference in single-cell ATAC-seq data.

Our model incorporates an additional module to integrate temporal information with the observed ATAC-seq data to inform the latent-space inference, correct measurement errors and learn a continuous ordering of cells from fewavailable timepoints. 

![Learned developmental trajectory](https://github.com/tohein/tempo/blob/master/notebooks/f1/animation.gif)



Our implementation uses stochastic layers based on existing PyTorch distribution objects, which simplify
the development of VAEs with diverse priors and likelihoods.
