Generative Adversarial Networks (GANs) in PyTorch
===============


## Introduction

We begin this trek to GANs-using-PyTorch here:
https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f#.sch4xgsa9

You might want to run this to get the original results:

```
./gan_pytorch.py
```

Original GAN Success Rate is 8/10. We asked ourselves the question:
**Can this be improved while the model remains parsimonious?**

## Research 2 Strategies for Success Rate Improvement

### I. Use Original Method (4 Moments) with Modifications
- Simulate Triangular Distribution, not Gaussian (Triangular has fixed tails)
- Increase Skewness Weight
- Modify optimizer settings (SGD momentum)

### II. Measure Distribution by Conditional Expectations
- Simulate Triangular Distribution, not Gaussian (Triangular has fixed tails)
- Define Conditional Expectations using Torch clamp function, a variant of ReLU. Apply weights to increase their gradient close to boundary points.
- Modify optimizer settings (SGD momentum)

