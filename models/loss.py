import tensorflow as tf
import keras
from keras.activations import softmax


LARGE_NUM = 1e9


def keras_loss(hidden_norm=True, temperature=1.0,weights=1.0):
  def loss(y_true, hidden):
    y_pred = add_contrastive_loss(hidden, hidden_norm, temperature, weights)
    return keras.losses.categorical_crossentropy(y_true, y_pred)
  return loss


def cos_similarity(hidden):
    hidden1, hidden2 = tf.split(hidden, 2, 0)
    x = K.l2_normalize(hidden1, axis=-1)
    y = K.l2_normalize(hidden2, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)



def add_contrastive_loss(hidden,
                         hidden_norm=True,
                         temperature=1.0,
                         weights=1.0):
  """Compute loss for model.
  Args:
    hidden: hidden vector (`Tensor`) of shape (2 * bsz, dim).
    hidden_norm: whether or not to use normalization on the hidden vector.
    temperature: a `floating` number for temperature scaling.
    tpu_context: context information for tpu.
    weights: a weighting number or vector.
  Returns:
    A loss scalar.
    The logits for contrastive prediction task.
    The labels for contrastive prediction task.
  """
  # Get (normalized) hidden1 and hidden2.
  if hidden_norm:
    hidden = tf.math.l2_normalize(hidden, -1)
  hidden1, hidden2 = tf.split(hidden, 2, 0)
  batch_size = tf.shape(hidden1)[0]


  hidden1_large = hidden1
  hidden2_large = hidden2
  labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
  masks = tf.one_hot(tf.range(batch_size), batch_size)

  logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
  logits_aa = logits_aa - masks * LARGE_NUM
  logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
  logits_bb = logits_bb - masks * LARGE_NUM
  logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
  logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature

  loss_a = tf.compat.v1.losses.softmax_cross_entropy(
      labels, tf.concat([logits_ab, logits_aa], 1), weights=weights)
  loss_b = tf.compat.v1.losses.softmax_cross_entropy(
      labels, tf.concat([logits_ba, logits_bb], 1), weights=weights)
  loss = loss_a + loss_b

  part1 = softmax(tf.concat([logits_ab, logits_aa], 1))
  part2 = softmax(tf.concat([logits_ba, logits_bb], 1))
  output = part1 + part2# tf.concat([part1, part2], 1)

  return output #loss , logits_ab, labels


from keras import backend as K
from keras.layers import Activation

from keras.utils import get_custom_objects


class Swish(Activation):
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = "swish"


def swish(x):
    return K.sigmoid(x) * x


get_custom_objects().update({"swish": Swish(swish)})
import keras
from keras.activations import softmax


class SoftmaxCosineSim(keras.layers.Layer):
    """Custom Keras layer: takes all z-projections as input and calculates
    output matrix which needs to match to [I|O|I|O], where
            I = Unity matrix of size (batch_size x batch_size)
            O = Zero matrix of size (batch_size x batch_size)
    """

    def __init__(self, batch_size, feat_dim, **kwargs):
        super(SoftmaxCosineSim, self).__init__()
        self.batch_size = batch_size
        self.feat_dim = feat_dim
        self.units = (batch_size, 4 * feat_dim)
        self.input_dim = [(None, feat_dim)] * (batch_size * 2)
        self.temperature = 0.1
        self.LARGE_NUM = 1e9

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "batch_size": self.batch_size,
                "feat_dim": self.feat_dim,
                "units": self.units,
                "input_dim": self.input_dim,
                "temperature": self.temperature,
                "LARGE_NUM": self.LARGE_NUM,
            }
        )
        return config

    def call(self, inputs):
        z1 = []
        z2 = []

        for index in range(self.batch_size):
            # 0-index assumes that batch_size in generator is equal to 1
            z1.append(tf.math.l2_normalize(inputs[index], -1))
            z2.append(
                tf.math.l2_normalize(inputs[self.batch_size + index], -1)
            )

        # Gather hidden1/hidden2 across replicas and create local labels.
        z1_large = z1
        z2_large = z2

        masks = tf.one_hot(tf.range(self.batch_size), self.batch_size)

        # Products of vectors of same side of network (z_i), count as negative examples
        # Values on the diagonal are put equal to a very small value
        # -> exclude product between 2 identical values, no added value
        logits_aa = tf.matmul(z1, z1_large, transpose_b=True) / self.temperature
        logits_aa = logits_aa - masks * self.LARGE_NUM

        logits_bb = tf.matmul(z2, z2_large, transpose_b=True) / self.temperature
        logits_bb = logits_bb - masks * self.LARGE_NUM

        # Similarity between two transformation sides of the network (z_i and z_j)
        # -> diagonal should be as close as possible to 1
        logits_ab = tf.matmul(z1, z2_large, transpose_b=True) / self.temperature
        logits_ba = tf.matmul(z2, z1_large, transpose_b=True) / self.temperature

        part1 = softmax(tf.concat([logits_ab, logits_aa], 1))
        part2 = softmax(tf.concat([logits_ba, logits_bb], 1))
        output = tf.concat([part1, part2], 1)

        return output
