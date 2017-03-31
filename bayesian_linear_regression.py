"""
EDWARD TUTORIAL: bayesian linear regression
-------------------------------------------

In supervised learning, the task is to infer hidden structure from labeled data,
comprised of training examples $\{(x_n, y_n)\}$.
Regression typically means the output yy takes continuous values.

Posit the model as Bayesian linear regression (Murphy, 2012). It assumes a linear
relationship between the inputs $\mathbf{x} \in \mathbb{R}^{D}$​ and the outputs
$y \in \mathbb{R}$.

For a set of $N$ data points $(\mathbf{X},\mathbf{y})=\{(\mathbf{x}_n, y_n)\}$,
the model posits the following distributions:
p(\mathbf{w}) = \text{Normal}(\mathbf{w} | \mathbf{0}, \sigma_w^2\mathbf{I})
p(b) = \text{Normal}(b | 0, \sigma_b^2)
p(\mathbf{y} | \mathbf{w}, b, \mathbf{X}) =
    \prod_{n=1}^N \text{Normal}(y_n \mid \mathbf{x}_n^\top\mathbf{w} + b, \sigma_y^2).
​​
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import edward as ed
from edward.models import Normal

plt.style.use('ggplot')
ed.set_seed(42)


# ===== Utils =====
def build_toy_dataset(N=50, noise_std=0.1):
    """
    y = \cos(x) + \epsilon
    :param N: Number of data points
    :param noise_std: Noise level
    :return: x, y
    """
    x = np.linspace(-3, 3, num=N)
    y = np.cos(x) + np.random.normal(0, noise_std, size=N)
    x = x.astype(np.float32).reshape((N, 1))
    y = y.astype(np.float32)
    return x, y


def neural_network(x, W_0, W_1, b_0, b_1):
    """
    2L neural network

    :param x:
    :param W_0:
    :param W_1:
    :param b_0:
    :param b_1:
    :return:
    """
    h = tf.tanh(tf.matmul(x, W_0) + b_0)
    h = tf.matmul(h, W_1) + b_1
    return tf.reshape(h, [-1])


# ===== Data =====
N = 50  # Number of data points
D = 1  # Number of features

w_true = np.random.randn(D)
x_train, y_train = build_toy_dataset(N)
x_test, y_test = build_toy_dataset(N)

# ===== Model =====
# Two layer Bayesian NN as approximator:
W_0 = Normal(mu=tf.zeros([D, 2]), sigma=tf.ones([D  , 2]))
W_1 = Normal(mu=tf.zeros([2, 1]), sigma=tf.ones([2, 1]))
b_0 = Normal(mu=tf.zeros(2), sigma=tf.ones(1))
b_1 = Normal(mu=tf.zeros(1), sigma=tf.ones(1))

x = x_train

y = Normal(mu=neural_network(x, W_0, W_1, b_0, b_1),
           sigma=0.1 * tf.ones(N))

# Normal variational approximation:
# Note: the parameters of each q are defined as tf.Variable,
# they can thus vary (and be learnt)
# Applying softplus to sigma is a trick to constrain them to
# positive values
qW_0 = Normal(mu=tf.Variable(tf.zeros([D, 2])),
              sigma=tf.nn.softplus(tf.Variable(tf.zeros([D, 2]))))
qW_1 = Normal(mu=tf.Variable(tf.zeros([2, 1])),
              sigma=tf.nn.softplus(tf.Variable(tf.zeros([2, 1]))))
qb_0 = Normal(mu=tf.Variable(tf.zeros(2)),
              sigma=tf.nn.softplus(tf.Variable(tf.zeros(2))))
qb_1 = Normal(mu=tf.Variable(tf.zeros(1)),
              sigma=tf.nn.softplus(tf.Variable(tf.zeros(1))))

# ===== Visualization (1) =====
# Visualizing the prior
# Sample functions from variational model to visualize fits.
rs = np.random.RandomState(0)
inputs = np.linspace(-5, 5, num=400, dtype=np.float32)
x = tf.expand_dims(inputs, 1)
mus = tf.stack(
    [neural_network(x, qW_0.sample(), qW_1.sample(),
                    qb_0.sample(), qb_1.sample())
     for _ in range(10)])

sess = ed.get_session()
tf.global_variables_initializer().run()
outputs = mus.eval()

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Iteration: 0")
ax.plot(x_train, y_train, 'ks', alpha=0.5, label='(x, y)')
ax.plot(inputs, outputs[0].T, 'r', lw=2, alpha=0.5, label='prior draws')
ax.plot(inputs, outputs[1:].T, 'r', lw=2, alpha=0.5)
ax.set_xlim([-5, 5])
ax.set_ylim([-2, 2])
ax.legend()
plt.show()

# ==== Inference =====
inference = ed.KLqp({W_0: qW_0, b_0: qb_0, W_1: qW_1, b_1: qb_1},
                    data={y: y_train})
inference.run(n_iter=500)

# ===== Visualization (2) =====
# Visualizing the posterior
outputs = mus.eval()

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Iteration: 500")
ax.plot(x_train, y_train, 'ks', alpha=0.5, label='(x, y)')
ax.plot(inputs, outputs[0].T, 'r', lw=2, alpha=0.5, label='posterior draws')
ax.plot(inputs, outputs[1:].T, 'r', lw=2, alpha=0.5)
ax.set_xlim([-5, 5])
ax.set_ylim([-2, 2])
ax.legend()
plt.show()
