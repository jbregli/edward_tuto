"""
EDWARD TUTORIAL: Linear Mixed Effects Models
--------------------------------------------

With linear mixed effects models, we wish to model a linear relationship for data points
with inputs of varying type, categorized into subgroups, and associated to a
real-valued output.

in this setup, one makes an independence assumption where each data point regresses
with a constant slope among each other. In our setting, the observations come from sets
of groups which may have varying slopes and intercepts. Thus we’d like to build a model
that can capture this behavior.

In a linear mixed effects model, we add an additional term $\mathbf{Z}\eta$ where
$\mathbf{Z}$ corresponds to random effects with coefficients $\eta$ to the linear
regression.
The model takes the form:
\eta &\sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})
\mathbf{y} &= \mathbf{X}\beta + \mathbf{Z}\eta + \epsilon​​

Given data, the goal is to infer $\beta$, $\eta$, and $\sigma^{2}$, where:
$\beta$ are model parameters (“fixed effects”)
$\eta$ are latent variables (“random effects”)
$\sigma^{2}$ is a variance component parameter.

Given data, we aim to infer the model's fixed and random effects.
In this analysis, we use variational inference with the KL(q∥p) divergence measure.
We specify fully factorized normal approximations for the random effects and pass in
all training data for inference.
Under the algorithm, the fixed effects will be estimated under a variational EM scheme.

"""
import edward as ed
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from edward.models import Normal

plt.style.use('ggplot')
ed.set_seed(42)

# ===== Data =====
# Use InstEval data set:
#  https://github.com/blei-lab/edward/blob/master/examples/data/insteval.csv
data = pd.read_csv('./data/insteval.csv')
data['dcodes'] = data['d'].astype('category').cat.codes
data['deptcodes'] = data['dept'].astype('category').cat.codes
data['s'] -= 1

train = data.sample(frac=0.8)
test = data.drop(train.index)

train.head()

s_train = train['s'].values.astype(int)
d_train = train['dcodes'].values.astype(int)
dept_train = train['deptcodes'].values.astype(int)
y_train = train['y'].values.astype(float)
service_train = train['service'].values.astype(int)
n_obs_train = train.shape[0]

s_test = test['s'].values.astype(int)
d_test = test['dcodes'].values.astype(int)
dept_test = test['deptcodes'].values.astype(int)
y_test = test['y'].values.astype(float)
service_test = test['service'].values.astype(int)
n_obs_test = test.shape[0]

n_s = 2972  # number of students
n_d = 1128  # number of instructors
n_dept = 14  # number of departments
n_obs = train.shape[0]  # number of observations

# ===== Model =====
# Set up placeholders for the inputs.
s_ph = tf.placeholder(tf.int32, [None])
d_ph = tf.placeholder(tf.int32, [None])
dept_ph = tf.placeholder(tf.int32, [None])
service_ph = tf.placeholder(tf.float32, [None])

# Set up fixed effects.
mu = tf.Variable(tf.random_normal([]))
service = tf.Variable(tf.random_normal([]))

sigma_s = tf.sqrt(tf.exp(tf.Variable(tf.random_normal([]))))
sigma_d = tf.sqrt(tf.exp(tf.Variable(tf.random_normal([]))))
sigma_dept = tf.sqrt(tf.exp(tf.Variable(tf.random_normal([]))))

# Set up random effects.
eta_s = Normal(mu=tf.zeros(n_s), sigma=sigma_s*tf.ones(n_s))
eta_d = Normal(mu=tf.zeros(n_d), sigma=sigma_s*tf.ones(n_d))
eta_dept = Normal(mu=tf.zeros(n_dept), sigma=sigma_s*tf.ones(n_dept))

yhat = tf.gather(params=eta_s, indices=s_ph) + \
    tf.gather(params=eta_d, indices=d_ph) + \
    tf.gather(params=eta_dept, indices=dept_ph) + \
    mu + service * service_ph
y = Normal(mu=yhat, sigma=tf.ones(n_obs))

# ===== Inference =====
q_eta_s = Normal(mu=tf.Variable(tf.random_normal([n_s])),
                 sigma=tf.nn.softplus(tf.Variable(tf.random_normal([n_s]))))
q_eta_d = Normal(mu=tf.Variable(tf.random_normal([n_d])),
                 sigma=tf.nn.softplus(tf.Variable(tf.random_normal([n_d]))))
q_eta_dept = Normal(mu=tf.Variable(tf.random_normal([n_dept])),
                    sigma=tf.nn.softplus(tf.Variable(tf.random_normal([n_dept]))))

latent_vars = {eta_s: q_eta_s,
               eta_d: q_eta_d,
               eta_dept: q_eta_dept}
data = {y: y_train,
        s_ph: s_train,
        d_ph: d_train,
        dept_ph: dept_train,
        service_ph: service_train}

inference = ed.KLqp(latent_vars=latent_vars,
                    data=data)

# Prediction for test set:
yhat_test = ed.copy(yhat,
                    dict_swap={eta_s: q_eta_s.mean(),
                               eta_d: q_eta_d.mean(),
                               eta_dept: q_eta_dept.mean()})

# ===== Training =====
inference.initialize(n_print=20, n_iter=500)
tf.global_variables_initializer().run()

plt.ion()
for _ in range(inference.n_iter):
    # Update and print progress of the algorithm.
    info_dict = inference.update()
    inference.print_progress(info_dict)

    t = info_dict['t']
    if t == 1 or t % inference.n_print == 0:
        if t > 1:
            plt.clf()

        # Make prediction on test data.
        yhat_vals = yhat_test.eval(feed_dict={s_ph: s_test,
                                              d_ph: d_test,
                                              dept_ph: dept_test,
                                              service_ph: service_test})

        # Form residual plot.
        plt.title("Residuals for Predicted Ratings on Test Set")
        plt.xlim(-4, 4)
        plt.ylim(0, 800)
        plt.hist(yhat_vals - y_test, 75)
        plt.pause(0.05)
