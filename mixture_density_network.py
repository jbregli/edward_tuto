"""
EDWARD TUTORIAL: Mixture Density Networks
-----------------------------------------

Mixture density networks (MDN) (Bishop, 1994) are a class of models obtained by
combining a conventional neural network with a mixture density model.

"""
import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from edward.models import Categorical, Mixture, Normal
from tensorflow.contrib import slim
from scipy import stats
from sklearn.model_selection import train_test_split

plt.style.use('ggplot')
ed.set_seed(42)


# ===== Utils =====
def plot_normal_mix(pis, mus, sigmas, ax, label='', comp=True):
    """
    Plots the mixture of Normal models to axis=ax
    comp=True plots all components of mixture model

    :param pis:
    :param mus:
    :param sigmas:
    :param ax:
    :param label:
    :param comp:
    :return: plot
    """
    x = np.linspace(-10.5, 10.5, 250)
    final = np.zeros_like(x)
    for i, (weight_mix, mu_mix, sigma_mix) in enumerate(zip(pis, mus, sigmas)):
        temp = stats.norm.pdf(x, mu_mix, sigma_mix) * weight_mix
        final = final + temp
        if comp:
            ax.plot(x, temp, label='Normal ' + str(i))
    ax.plot(x, final, label='Mixture of Normals ' + label)
    ax.legend(fontsize=13)


def sample_from_mixture(x, pred_weights, pred_means, pred_std, amount):
    """
    Draws samples from mixture model.

    :param x:
    :param pred_weights:
    :param pred_means:
    :param pred_std:
    :param amount:
    :return: 2 d array with input X and sample from prediction of mixture model.
    """
    samples = np.zeros((amount, 2))
    n_mix = len(pred_weights[0])
    to_choose_from = np.arange(n_mix)
    for j, (weights, means, std_devs) in enumerate(
            zip(pred_weights, pred_means, pred_std)):
        index = np.random.choice(to_choose_from, p=weights)
        samples[j, 1] = np.random.normal(means[index], std_devs[index], size=1)
        samples[j, 0] = x[j]

        if j == amount - 1:
            break
    return samples


def build_toy_dataset(N):
    """
    x = 7*sin(0.75*y) + y/2 + r

    Build the dataset
    :param N: Number of observation
    :return: X_train, X_test, y_train, y_test
    """
    y_data = np.random.uniform(-10.5, 10.5, N)
    r_data = np.random.normal(size=N)  # random noise
    x_data = np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + r_data * 1.0
    x_data = x_data.reshape((N, 1))
    return train_test_split(x_data, y_data, random_state=42)


def neural_network(X):
    """
    2 hidden layers with 15 hidden units and 3 output layers
    :param X:
    :return:  mu, sigma, logits
    """
    # 2 hidden layers with 15 hidden units
    hidden1 = slim.fully_connected(X, 15)
    hidden2 = slim.fully_connected(hidden1, 15)

    mus = slim.fully_connected(hidden2, K, activation_fn=None)
    sigmas = slim.fully_connected(hidden2, K, activation_fn=tf.exp)
    logits = slim.fully_connected(hidden2, K, activation_fn=None)

    return mus, sigmas, logits


# ===== Data =====
N = 5000  # number of data points
D = 1  # number of features
K = 20  # number of mixture components

X_train, X_test, y_train, y_test = build_toy_dataset(N)
print("Size of features in training data: {}".format(X_train.shape))
print("Size of output in training data: {}".format(y_train.shape))
print("Size of features in test data: {}".format(X_test.shape))
print("Size of output in test data: {}".format(y_test.shape))


# ===== Visualization (1) =====
# Visualizing the dataset
sns.regplot(X_train, y_train, fit_reg=False)
plt.show()

# ===== Model =====
X_ph = tf.placeholder(tf.float32, [None, D])
y_ph = tf.placeholder(tf.float32, [None])

mus, sigmas, logits = neural_network(X_ph)
cat = Categorical(logits=logits)
components = [Normal(mu=mu, sigma=sigma) for mu, sigma
              in zip(tf.unstack(tf.transpose(mus)),
                     tf.unstack(tf.transpose(sigmas)))]

y = Mixture(cat=cat, components=components, value=tf.zeros_like(y_ph))
# Note: A bug exists in Mixture which prevents samples from it to have
# a shape of [None]. For now fix it using the value argument, as
# sampling is not necessary for MAP estimation anyways.

# ==== Inference =====
# MAP estimation
# There are no latent variables to infer. Thus inference is concerned
# with only training model parameters, which are baked into how we
# specify the neural networks.
inference = ed.MAP(data={y: y_ph})

# Example of how to manually control training:
# First initialize then recursively call update()
inference.initialize(var_list=tf.trainable_variables())
sess = ed.get_session()
tf.global_variables_initializer().run()

n_epoch = 1000
train_loss = np.zeros(n_epoch)
test_loss = np.zeros(n_epoch)
for i in range(n_epoch):
    info_dict = inference.update(feed_dict={X_ph: X_train,
                                            y_ph: y_train})
    train_loss[i] = info_dict['loss']
    test_loss[i] = sess.run(inference.loss,
                          feed_dict={X_ph: X_test,
                                     y_ph: y_test})
    inference.print_progress(info_dict)

# Test set predictions:
pred_weights, pred_means, pred_std = \
    sess.run([tf.nn.softmax(logits), mus, sigmas], feed_dict={X_ph: X_test})

# ===== Visualization (2) =====
# Visualizing the training loss
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 3.5))
plt.plot(np.arange(n_epoch), -test_loss / len(X_test), label='Test')
plt.plot(np.arange(n_epoch), -train_loss / len(X_train), label='Train')
plt.legend(fontsize=20)
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Log-likelihood', fontsize=15)
plt.show()

# ===== Criticism =====
# 1) Look at how individual examples perform.
# NOTE: this is an inverse problem we can't get the answer correct,
# but we can hope that the truth lies in area where the model has
# high probability.
# In this plot the truth is the vertical grey line while the blue line is
# the prediction of the mixture density network.
# As you can see, we didn't do too bad.
obj = [0, 4, 6]
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 6))

plot_normal_mix(pred_weights[obj][0], pred_means[obj][0],
                pred_std[obj][0], axes[0], comp=False)
axes[0].axvline(x=y_test[obj][0], color='black', alpha=0.5)

plot_normal_mix(pred_weights[obj][2], pred_means[obj][2],
                pred_std[obj][2], axes[1], comp=False)
axes[1].axvline(x=y_test[obj][2], color='black', alpha=0.5)

plot_normal_mix(pred_weights[obj][1], pred_means[obj][1],
                pred_std[obj][1], axes[2], comp=False)
axes[2].axvline(x=y_test[obj][1], color='black', alpha=0.5)
plt.show()

# 2) Check the ensemble by drawing samples of the prediction and plotting
# the density of those.
# The MDN has learned what we'd like it to learn.
a = sample_from_mixture(X_test, pred_weights, pred_means,
                        pred_std, amount=len(X_test))
sns.jointplot(a[:, 0], a[:, 1], kind="hex", color="#4CB391",
              ylim=(-10, 10), xlim=(-14, 14))
plt.show()

