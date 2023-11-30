import tensorflow as tf
import numpy as np
import datetime


def soft_dice_loss(y_true, y_pred, epsilon=1e-30):
    true_positives = 2 * tf.reduce_sum(y_pred * y_true)

    total_predicted_expected = tf.reduce_sum(tf.square(y_pred)) + tf.reduce_sum(tf.square(y_true))
    result = 1 - (true_positives + epsilon) / (total_predicted_expected + epsilon)

    return result