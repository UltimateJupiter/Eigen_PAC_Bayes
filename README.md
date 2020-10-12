# Tighter PAC-Bayer Generalization Bound with Estimated Layer-wise Hessian Eigenbasis

This work is based on the previous work by [Dziugaite & Roy](https://arxiv.org/abs/1703.11008) which proposed optimizing the PAC-Bayes Bound for test error using SGD.

We applied our [new understandings of the layer-wise Hessian](https://arxiv.org/abs/2010.04261) to better align the covariance of posterior distribution with the sharp / flat directions in the loss landscape and achieved tighter generaliation bounds.

The idea of the approximation is similar to that of [KFAC](https://arxiv.org/abs/1503.05671) but on Hessians instead of FIM.
