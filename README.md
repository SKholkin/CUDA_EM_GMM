# JAX Accelerated EM for Gaussian Mixture Models
Current repo provides implementation of EM algorithm in JAX using all of the nice just-in-time compilation utilities of this beautiful library, such as:

1. JIT compilation using XLA
2. Using CUDA is easy
3. Analytic Autodiff
4. Easy to use, like Numpy

## Running scripts
All of the code is contained in Jupyter notebook `jax_em.ipynb` for the ease of visualizations.
Or you can use Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11O8irggdiiuApxJCQdwd2k-YjZMmFxtu?usp=sharing)
## Checkpoints & Data
Datasets are contained in `data` folder.
