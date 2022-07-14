import tensorflow_probability as tfp
tfd = tfp.distributions
from scipy.stats import norm 
import numpy as np

x = np.linspace(0, 1, 100) 


# Define a single scalar Normal distribution.
dist = tfd.Normal(loc=0., scale=3.)

tf = dist.prob(x)
sci = norm.pdf(x, loc=0., scale=3.)

print(tf/sci)