# Cell-state variational autoencoder with temporal integration.

A variational autoencoder (VAE) model for latent cell-state inference from single-cell ATAC-seq (scATAC-seq) data [1].

Our implementation features an additional module to integrate temporal information with scATAC-seq counts to inform the latent-space inference, correct measurement errors and learn a continuous ordering of cells from few available timepoints. 

![Learned developmental trajectory](https://github.com/tohein/tempo/blob/master/notebooks/f1/animation.gif)

The application of Tempo to data from four Drosophila melanogaster F1 crosses profiled at three stages across embryonic development [1] can be found in the [Jupyter notebook](https://github.com/tohein/tempo/blob/master/notebooks/f1/Tempo_final.ipynb).

## References

[1] T. Heinen, S. Secchia, J. Reddington, B. Zhao, E.E.M. Furlong, O. Stegle. [scDALI: Modelling allelic heterogeneity of DNA accessibility in single-cells reveals context-specific genetic regulation](https://www.biorxiv.org/content/10.1101/2021.03.19.436142v1). Preprint, bioRxiv (2021).
