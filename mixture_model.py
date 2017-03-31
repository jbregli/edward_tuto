"""
EDWARD TUTORIAL: Mixture models
-------------------------------

In unsupervised learning, the task is to infer hidden structure from unlabeled data,
comprised of training examples |{x_{n}\​}.

A mixture model is a model typically used for clustering. It assigns a mixture component
to each data point, and this mixture component determines the distribution that the data point
is generated from. A mixture of Gaussians uses Gaussian distributions to generate this data
(Bishop, 2006).

For a set of N data points, the likelihood of each observation x_{n} is
p(x_{n}∣\pi,\mu\sigma) = \sum_{k=1}^{K} \pi_{k} Normal(x_{n}∣\mu_{k}, \sigma_{k})
Where the latent variable \pi is a K-dimensional probability vector which mixes individual
Gaussian distributions, each characterized by a mean \mu_{k} and standard deviation
\sigma_{k}.

Define the prior on \pi \in [0,1] such that \sum_{k}^{K}\pi_{k}=1 to be
p(\pi)=Dirichlet(\pi∣\alpha_{K}).

Define the prior on each component \mu_{k} \in \matbb{R}^{D} to be
p(\mu_{k})=Normal(\mu_{k}∣0,2 \diag() \sigma^{2}).

Define the prior on each component \sigma_{k} \in \mathbb{R}^{D} to be
p(\sigma_{k})=InverseGamma(\sigma_{k}∣a, b).
"""

import edward as ed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import six
import tensorflow as tf

from edward.models import Categorical, InverseGamma, Mixture, MultivariateNormalDiag, Normal

plt.style.use('ggplot')
ed.set_seed(42)


# ===== Utils =====
def build_toy_dataset(N):
    """
    x \in \mathbb{R}^{2}
    It's either generated from N([1,1], [0.1,0.1]) with probability pi
    or N([-1,-1], [0.1,0.1]) with probability 1-pi
    :param N: Number of samples
    :return: x
    """
    pi = np.array([0.4, 0.6])
    mus = [[1, 1], [-1, -1]]
    stds = [[0.1, 0.1], [0.1, 0.1]]
    x = np.zeros((N, 2), dtype=np.float32)

    for n in range(N):
        k = np.argmax(np.random.multinomial(1, pi))
        x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))

    return x


# ===== Data =====
N = 500  # number of data points
K = 2  # number of components
D = 2  # dimensionality of data

x_train = build_toy_dataset(N)


# ===== Visualization (1) =====
# Visualizing the dataset
plt.scatter(x_train[:, 0], x_train[:, 1])
plt.axis([-3, 3, -3, 3])
plt.title("Simulated dataset")
plt.show()


# ===== Model =====
# Standard mixture of gaussian:
mu = Normal(mu=tf.zeros([K, D]), sigma=tf.ones([K, D]))
sigma = InverseGamma(alpha=tf.zeros([K, D]), beta=tf.ones([K, D]))
c = Categorical(logits=tf.zeros([N, K]))
x = Normal(mu=tf.gather(mu, c), sigma=tf.gather(sigma, c))

# Collapsed mixture: Marginalizes out the mixture assignments
mu = Normal(mu=tf.zeros([K, D]), sigma=tf.ones([K, D]))
sigma = InverseGamma(alpha=tf.ones([K, D]), beta=tf.ones([K, D]))
c = Categorical(logits=tf.zeros([N, K]))
components = [
    MultivariateNormalDiag(mu=tf.ones([N, 1]) * mu[k],
                           diag_stdev=tf.ones([N, 1]) * sigma[k])
    for k in range(K)]
x = Mixture(cat=c, components=components)

# ==== Inference =====
q_mu = Normal(mu=tf.Variable(tf.random_normal([K, D])),
              sigma=tf.nn.softplus(tf.Variable(tf.zeros([K, D]))))
q_sigma = InverseGamma(alpha=tf.nn.softplus(tf.Variable(tf.random_normal([K, D]))),
                       beta=tf.nn.softplus(tf.Variable(tf.random_normal([K, D]))))

inference = ed.KLqp({mu: q_mu, sigma: q_sigma}, data={x: x_train})
inference.initialize(n_samples=20, n_iter=4000)

sess = ed.get_session()
tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
    info_dict = inference.update()
    inference.print_progress(info_dict)
    t = info_dict['t']
    if t % inference.n_print == 0:
        print("\nInferred cluster means:")
        print(sess.run(q_mu.mean()))

# ===== Criticism =====
# Calculate likelihood for each data point and cluster assignment,
# averaged over many posterior samples.
# x_post has shape (N, 100, K, D).
mu_sample = q_mu.sample(100)
sigma_sample = q_sigma.sample(100)
x_post = Normal(mu=tf.ones([N, 1, 1, 1]) * mu_sample,
                sigma=tf.ones([N, 1, 1, 1]) * sigma_sample)
x_broadcasted = tf.tile(tf.reshape(x_train,
                                   [N, 1, 1, D]),
                        [1, 100, K, 1])

# Sum over latent dimension, then average over posterior samples.
# log_liks ends up with shape (N, K).
log_liks = x_post.log_prob(x_broadcasted)
log_liks = tf.reduce_sum(log_liks, 3)
log_liks = tf.reduce_mean(log_liks, 1)

# Choose the cluster with the highest likelihood for each data point.
clusters = tf.argmax(log_liks, 1).eval()

plt.scatter(x_train[:, 0], x_train[:, 1], c=clusters, cmap=cm.bwr)
plt.axis([-3, 3, -3, 3])
plt.title("Predicted cluster assignments")
plt.show()


