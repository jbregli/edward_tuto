"""
EDWARD TUTORIAL: Latent space models
-------------------------------------------

What we can learn about nodes in a network from their connectivity patterns?
We can begin to study this using a latent space model (Hoff, Raftery, & Handcock, 2002).
Latent space models embed nodes in the network in a latent space, where the likelihood of
forming an edge between two nodes depends on their distance in the latent space.

Each neuron n is a node in the network and is associated with a latent position
z_{n} \in \mathbb{R}^{K}. We place a Gaussian prior on each of the latent positions.
The log-odds of an edge between node i and j is proportional to the Euclidean distance
between the latent representations of the nodes |z_{i}−z_{j}|.
Here, we model the weights (Y_{ij}) of the edges with a Poisson likelihood.
The rate is the reciprocal of the distance in latent space.
The generative process is as follows:
    1) For each node n=1,…,N,
        z_{n} \sim Normal(0,I)
    2) For each edge (i,j) \in \{1,…,N\}×\{1,…,N\},
        Y_{ij} \sim Poisson(\frac{1}{|zi−zj|}).


NOTE:
Uses crabs data set:
https://github.com/blei-lab/edward/tree/master/examples/data
"""
import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal, Poisson


# ===== Data =====
x_train = np.load('data/celegans_brain.npy')

N = x_train.shape[0]  # number of data points
K = 3  # latent dimensionality


# ===== Model =====
z = Normal(mu=tf.zeros([N, K]), sigma=tf.ones([N, K]))

# Calculate N x N distance matrix.
# 1. Create a vector, [||z_1||^2, ||z_2||^2, ..., ||z_N||^2],
#  and tile it to create N identical rows.
xp = tf.tile(tf.reduce_sum(tf.pow(z, 2), 1, keep_dims=True), [1, N])

# 2. Create a N x N matrix where entry (i, j) is
# ||z_i||^2 + ||z_j||^2 - 2 z_i^T z_j.
xp = xp + tf.transpose(xp) - 2 * tf.matmul(z, z, transpose_b=True)

# 3. Invert the pairwise distances and make rate along diagonals to
# be close to zero.
xp = 1.0 / tf.sqrt(xp + tf.diag(tf.zeros(N) + 1e3))

# Note Edward doesn't currently support sampling for Poisson.
# Hard-code it to 0's for now
# It isn't used during inference, but just required for instanciating
# some variables.
x = Poisson(lam=xp, value=tf.zeros_like(xp))


# ===== Inference =====
# MAP:
inference_map = ed.MAP([z], data={x: x_train})
print("--- MAP exact inference: ---")
inference_map.run(n_iter=2500)

# Variational approximation:
q_z = Normal(mu=tf.Variable(tf.random_normal([N, K])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([N, K]))))
inference_vi = ed.KLqp({z: q_z}, data={x: x_train})
print("--- Variational approximate inference: ---")
inference_vi.run(n_iter=2500)