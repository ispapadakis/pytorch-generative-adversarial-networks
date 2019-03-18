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
**Can success rate be improved while the model remains parsimonious?**

## Research 2 Strategies for Success Rate Improvement

### I. Use Original Method (4 Moments) with Modifications
- Simulate Triangular Distribution, not Gaussian (Triangular has fixed tails)
- Increase Skewness Weight
- Modify optimizer settings (SGD momentum)

### II. Measure Distribution by Conditional Expectations
- Simulate Triangular Distribution, not Gaussian (Triangular has fixed tails)
- Define Conditional Expectations using Torch clamp function, a variant of ReLU. Apply weights to increase their gradient close to boundary points.
- Modify optimizer settings (SGD momentum)

## RESULTS
Measure success rate as done in our reference:
- Observe distribution of "fake" images after 10 replications (each replication ends after 5,000 epochs)
- Visually inspect "fake" image distribution histograms. 
- Report how many times "fake" distributions would pass for triangular

### I. 4-Moments Method
Success here is literally in the eye of the beholder. All resulting "fake" distributions have a mode at their center, which is close to 4. They lie far from the original of a uniform distribution and much closer to the triangular distribution.

We call the sucess rate here **9.25/10**. We deem 0.50 points as lost in replication 1, where the right tail is too fat for triangular. Also, 0.25 points are lost in replication 6, as both tails are excessively fat.

### II. Conditional Expectations
Again, the Generator part of GAN creates "fake" distributions with mode at 4, as desired. We call the success rate here **9.50/10**. In many cases, the resulting distributions show higher kurtosis than expected from a triangular distribution. Overall, Conditional Expectations appear to add to the reliability of this GAN.

