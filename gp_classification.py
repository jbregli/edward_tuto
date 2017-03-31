"""
EDWARD TUTORIAL: Gaussian process classification
------------------------------------------------

In supervised learning, the task is to infer hidden structure from labeled data,
comprised of training examples $\{(x_n, y_n)\}$.
Classification means the output y takes discrete values.

A Gaussian process is a powerful object for modeling nonlinear relationships between
pairs of random variables. It defines a distribution over (possibly nonlinear) functions,
which can be applied for representing our uncertainty around the true functional
relationship. Here we define a Gaussian process model for classification
(Rasumussen & Williams, 2006).

Formally, a distribution over functions f:\mathbb{R}^{D}→\mathbb{R} can be specified by a
Gaussian process,
p(f)=GP(f∣0,k(x,x′)),
whose mean function is the zero function,
and whose covariance function is some kernel which describes dependence between any set of
inputs to the function.

Given a set of input-output pairs {x_{n} \in \mathbb{R}^{D}, y_{n} \in \mathbb{R}}, the
likelihood can be written as a multivariate normal,
p(y)=Normal(y∣0,K)
where K is a covariance matrix given by evaluating k(x_{n},x_{m}) for each pair of inputs
in the data set.

The above applies directly for regression where y is a real-valued response, but not for
(binary) classification, where y is a label in {0,1}. To deal with classification, we
interpret the response as latent variables which is squashed into [0,1]. We then draw from
a Bernoulli to determine the label, with probability given by the squashed value.

Define the likelihood of an observation (x_{n},y_{n}) as,
p(y_{n}∣z,x_{n})=Bernoulli(y_{n}∣logit−1(x_{n}^{⊤}z)).

Define the prior to be a multivariate normal
p(z)=Normal(z∣0,K),
with covariance matrix given as previously stated

NOTE:
Uses crabs data set:
https://github.com/blei-lab/edward/tree/master/examples/data
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import edward as ed

from edward.models import Bernoulli, MultivariateNormalCholesky, Normal
from edward.util import rbf

plt.style.use('ggplot')
ed.set_seed(42)

# ===== Data =====
data = np.loadtxt('data/crabs_train.txt', delimiter=',')
data[data[:, 0] == -1, 0] = 0  # replace -1 label with 0 label

N = data.shape[0]  # number of data points
D = data.shape[1] - 1  # number of features

X_train = data[:, 1:]
y_train = data[:, 0]

print("Number of data points: {}".format(N))
print("Number of features: {}".format(D))

# ===== Model =====
X = tf.placeholder(tf.float32, [N, D])
f = MultivariateNormalCholesky(mu=tf.zeros(N), chol=tf.cholesky(rbf(X)))
y = Bernoulli(logits=f)

# ==== Inference =====
qf = Normal(mu=tf.Variable(tf.random_normal([N])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([N]))))

inference = ed.KLqp({f: qf}, data={X: X_train, y: y_train})
inference.run(n_iter=5000)
