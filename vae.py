# Variational Auto-Encoder - implementation.
# Apr-Oct 2021 (v5).


import time
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm


# GPU memory hack
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


# set random seed for reproducibility
tf.random.set_seed(1)


# Hyperparameters
# model
lat_dim      = 2        # latent dimension, in principle free choice but in practice fixed by plotting code
hid_dim      = 500      # free choice, hidden dimension
kl_wt        = 1e-3     # free choice, weight given to the Kullback Leibler loss term
# training
batch_size   = 128      # free choice, limited by GPU memory, Kingma & Welling used 100
num_epochs   = 100      # free choice, number of training epochs
ann_epochs   = 0        # free choice, number of Kullback Leibler annealing epochs
l2_wt        = 0        # free choice, L2 regularization weight


# Data

(x_train, _), (x_test, y_test) = mnist.load_data()

x_train  = x_train.astype('float32') / 255.0
x_train  = x_train.reshape(x_train.shape[0],-1)
x_test   = x_test.astype('float32') / 255.0
x_test   = x_test.reshape(x_test.shape[0],-1)

train_ds = tf.data.Dataset.from_tensor_slices(x_train)
train_ds = train_ds.batch(batch_size)
val_ds   = tf.data.Dataset.from_tensor_slices(x_test)
val_ds   = val_ds.batch(batch_size)

# image dimension in pixels
img_dim = x_train.shape[1]
# KL annealing training steps
ann_stp = ann_epochs * x_train.shape[0] / batch_size
 

# Model

# The encoder converts an image into the mean u and log variance lv of a normal distn over latent dimensions.
# That is, the posterior distribution q(z|x) = N(z; u, exp(lv) o I) where o is element-wise multiplication.
# Log variance is predicted to ensure that variance is always positive.

class Encoder(layers.Layer):
    '''VAE Encoder following Kingma & Welling, 2013'''
    def __init__(self, lat_dim, hid_dim, l2_wt, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        # store parameters
        self.lat_dim = lat_dim
        self.hid_dim = hid_dim
        self.l2_wt   = l2_wt
        # l2 weight regularizer
        self.reg  = keras.regularizers.L2(l2=l2_wt)
        # input to hidden dense layer
        self.i2h  = layers.Dense(hid_dim, kernel_regularizer=self.reg, activation='tanh')
        # hidden to latent mean dense layer
        self.h2u  = layers.Dense(lat_dim, kernel_regularizer=self.reg)
        # hidden to latent log variance dense layer
        self.h2lv = layers.Dense(lat_dim, kernel_regularizer=self.reg)
        #
    def call(self, inputs):
        # inputs are images,       (batch, img_dim)
        # compute hidden layer,    (batch, hid_dim)
        h  = self.i2h(inputs)
        # compute latent means,    (batch, lat_dim)
        u  = self.h2u(h)
        # compute latent log vars, (batch, lat_dim)
        lv = self.h2lv(h)
        # return latent means, log vars
        return u, lv



# The decoder converts latent vectors z, whether sampled from the encoder distn or from
# the prior distn into Bernoulli distribution parameters, one for every pixel in the image.

class Decoder(layers.Layer):
    '''VAE Decoder following Kingma & Welling, 2013'''
    def __init__(self, img_dim, hid_dim, l2_wt, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        # store parameters
        self.img_dim = img_dim
        self.hid_dim = hid_dim
        self.l2_wt   = l2_wt
        # l2 weight regularizer
        self.reg = keras.regularizers.L2(l2=l2_wt)
        # latent to hidden dense layer
        self.l2h = layers.Dense(hid_dim, kernel_regularizer=self.reg, activation='tanh')
        # hidden to Bernoulli distribution parameter dense layer
        self.h2y = layers.Dense(img_dim, kernel_regularizer=self.reg, activation='sigmoid')
        #
    def call(self, inputs):
        # inputs (latent vectors), (batch, lat_dim)
        # compute hidden layer,    (batch, hid_dim)
        h = self.l2h(inputs)
        # compute Bernoulli parms, (batch, img_dim)
        y = self.h2y(h)
        # return Bernoulli parms
        return y



# The VAE model includes both encoder and decoder.
# Both are trained by maximizing the Variational Lower Bound by encoding training examples into
# latent variable distributions, sampling from those distributions, and decoding the samples into
# predicted images, which, since this is an autoencoder, should be similar to the training examples.
# The sampling process ensures that latent variables close to the encoding decode to similar images,
# which enforces local continuity and ultimately wider structure on the latent space.

class VAE(keras.Model):
    def __init__(self, img_dim, lat_dim, hid_dim, ann_stp, kl_wt, l2_wt, **kwargs):
        super(VAE, self).__init__(**kwargs)
        # store parameters
        self.img_dim = img_dim
        self.lat_dim = lat_dim
        self.hid_dim = hid_dim
        self.ann_stp = tf.constant(ann_stp, dtype=tf.float32)
        self.kl_wt   = tf.constant(kl_wt,  dtype=tf.float32)
        self.l2_wt   = l2_wt
        # encoder & decoder
        self.encoder = Encoder(lat_dim, hid_dim, l2_wt)
        self.decoder = Decoder(img_dim, hid_dim, l2_wt)
        # initialize the training steps counter to zero
        self.trn_stp = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        #
    def vae_loss(self, x, y, u, lv):
        # We want to maximize the variational lower bound, which includes maximizing p(x|z) where x is
        # sampled from the data and the probability is computed using the model.  Maximizing the log
        # probability is the same thing as minimizing the cross entropy, see Deep Learning, p129.  So
        # the loss to be minimized is binary_crossentropy(true, pred):
        xent_loss = tf.keras.metrics.binary_crossentropy(x, y)
        # The other term in the variational lower bound is -DKL(q(z|x)||p(z)).  The loss is minus
        # this and is given, see Kingma and Welling, 2013, by
        kl_loss   = -0.5 * tf.reduce_sum(1 + lv - tf.square(u) - tf.exp(lv), axis=-1)
        # Apply KL cost weighting, and annealing as described in Akuzawa, 2018 (from Bowman, 2016)
        kl_weight = self.kl_wt * tf.minimum((self.trn_stp + 1.0) / (self.ann_stp + 1.0), 1.0)
        # Both the loss terms above have shape (batch, ) so take the mean over batch here.
        return tf.reduce_mean(xent_loss + kl_weight * kl_loss)
    #
    def sample(self, u, lv):
        # first sample epsilon from N(0, I),                        (batch, lat_dim)
        epsilon = tf.random.normal([tf.shape(u)[0], self.lat_dim])
        # shift and scale epsilon to sample from N(u, exp(lv) o I), (batch, lat_dim)
        return u + epsilon * tf.exp(0.5 * lv)
    #
    def call(self, inputs, training):
        # inputs are images,    (batch, img_dim)
        # training is a Python boolean
        #
        # encode each image into a normal distribution over latent dimensions N(u, exp(lv) o I)
        # u and lv shapes       (batch, lat_dim)
        u, lv = self.encoder(inputs)
        # sample from each normal distribution, (batch, lat_dim)
        z     = self.sample(u, lv)
        # decode an image from each sampled z,  (batch, img_dim)
        y     = self.decoder(z)
        # compute the VAE loss
        loss  = self.vae_loss(inputs, y, u, lv)
        # add the VAE loss as the layer's loss
        self.add_loss(loss)
        # if training increment the training step counter
        if training:
            self.trn_stp.assign_add(1.0)
        # return u and lv which may be used to plot progress
        return u, lv


# instantiate the model
vae = VAE(img_dim, lat_dim, hid_dim, ann_stp, kl_wt, l2_wt)

# define optimizer
optimizer = tf.keras.optimizers.RMSprop()

# define accumulators
train_loss = tf.keras.metrics.Mean()
val_loss   = tf.keras.metrics.Mean()

# setup tensorboard logging directories
# run "tensorboard --logdir logs" in shell
# point browser at http://localhost:6007/
current_time         = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir        = 'logs/' + current_time + '/train'
val_log_dir          = 'logs/' + current_time + '/val'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer   = tf.summary.create_file_writer(val_log_dir)

# train step
signature = [tf.TensorSpec(shape=(None, img_dim), dtype=tf.float32)]

@tf.function(input_signature = signature)
def train_step(d):
    with tf.GradientTape() as tape:
        # call the model, computing the loss and the encodings
        u, lv = vae(d, training=True)
        # obtain the loss
        loss  = sum(vae.losses)
    #
    # compute and apply the gradients
    gradients = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
    # accumulate the loss
    train_loss(loss)
    # return encodings
    return u, lv


# val step
@tf.function(input_signature = signature)
def val_step(d):
    # call the model
    _, _ = vae(d, training=False)
    # obtain the loss
    loss  = sum(vae.losses)
    # accumulate the loss
    val_loss(loss)


ul = []
sl = []

# training loop
for epoch in range(num_epochs):
    start = time.time()
    #
    # reset accumulators
    train_loss.reset_states()
    val_loss.reset_states()
    #
    for d in train_ds:
        u, lv = train_step(d)
        # for the first epoch log the mean and stdev of the 1st example in each batch
        if epoch == 0:
            ul.append(u.numpy()[0])
            sl.append(tf.exp(0.5 * lv).numpy()[0])
    #
    print(f'epoch {epoch+1:3d} train loss {train_loss.result():.4f}, ', end='')
    # tensorboard log
    with train_summary_writer.as_default():
        _ = tf.summary.scalar('loss', train_loss.result(), step=epoch)
    # validate epoch
    for d in val_ds:
        val_step(d)
    #
    print(f'val loss {val_loss.result():.4f}, ', end='')
    # tensorboard log
    with val_summary_writer.as_default():
        _ = tf.summary.scalar('loss', val_loss.result(), step=epoch)
    #
    print(f'time taken {time.time() - start:.2f}s')
    

# save model
vae.save("trained_model")


# now use the model generatively
# produce a 15x15 grid of images

n = 15
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, xi in enumerate(grid_x):
    for j, yj in enumerate(grid_y):
        z_sample = np.array([[xi, yj]], dtype=np.float32)
        z_image  = vae.decoder(z_sample)
        digit    = tf.reshape(z_image, [digit_size, digit_size])
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10,10))
plt.imshow(figure, cmap='Greys_r')
plt.tick_params(
    axis='both',
    which='both',
    bottom=False,
    left=False,
    labelbottom=False,
    labelleft=False)
plt.show()


# now use ul and sl to plot q(z|x) as the model learns over the 1st epoch
# in each case plot the mean surrounded by an ellipse at 1 standard deviation

plt.figure(figsize=(10,10))
plt.axes()
ax = plt.gca()
ax.set_aspect('equal')
lim=10
plt.xlim([-lim,lim])
plt.ylim([-lim,lim])
cmap = plt.get_cmap('rainbow').reversed()
for i in range(len(ul)):
    colour = cmap(i / len(ul))
    _ = ax.add_patch(matplotlib.patches.Ellipse(ul[i], 2*sl[i][0], 2*sl[i][1], fc='none', ec=colour))

# add a circle for the prior
ax.add_patch(matplotlib.patches.Ellipse((0.0,0.0), 2.0, 2.0, fc='none', ec='black'))
plt.title('q(z|x) posterior distributions, 1 example per batch for 1st training epoch\n'
          'colourmap is rainbow, with red at the start of epoch, violet at the end\n'
          'ellipses are drawn at 1 standard deviation\n'
          'black circle is the prior, N(0,I)')
plt.xlabel('latent dim 0')
plt.ylabel('latent dim 1')
plt.show()

