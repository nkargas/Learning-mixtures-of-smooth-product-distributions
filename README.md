# Learning Mixtures of Smooth Product Distributions: Identifiability and Algorithm

This is the implementation of the following paper \
[Learning Mixtures of Smooth Product Distributions: Identifiability and Algorithm](https://arxiv.org/abs/1904.01156) \
Nikos Kargas, Nicholas D. Sidiropoulos \
AISTATS 2019.

Demo.m: Produces Fig.5 from the paper.
Requires Tensorlab (https://www.tensorlab.net/)

### Abstract
We study the problem of learning a mixture model of non-parametric product distributions. The problem of learning a mixture model is that of finding the component distributions along with the mixing weights using observed samples generated from the mixture. The problem is well-studied in the parametric setting, i.e., when the component distributions are members of a parametric family -- such as Gaussian distributions. In this work, we focus on multivariate mixtures of non-parametric product distributions and propose a two-stage approach which recovers the component distributions of the mixture under a smoothness condition. Our approach builds upon the identifiability properties of the canonical polyadic (low-rank) decomposition of tensors, in tandem with Fourier and Shannon-Nyquist sampling staples from signal processing. We demonstrate the effectiveness of the approach on synthetic and real datasets.

If you find this code useful for your research, please cite our paper:

```
@inproceedings{KarSid2019,
    title = {Learning Mixtures of Smooth Product Distributions: Identifiability and Algorithm},
    author = {Kargas, Nikos and Sidiropoulos, Nicholas D},
    booktitle = {Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics (AISTATS)},
    pages = {388--396},
    year = {2019}
}
```
