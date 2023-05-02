"""
Generating adversarial examples using the l2 attack as in Carlini and Walker 2016
"""

import pickle as pk
import tensorflow as tf
from tqdm import tqdm
from cnn import CNN


class L2Attack:

    def __init__(self, model, **kwargs):
        self.model = model  # NOTE the model must return logits
        self.optimizer = tf.keras.optimizers.Adam()
        self.num_epochs = 2500
        self.c = 1

    def __call__(self, x, target):
        """
        :param x: input image
        :param target: label corresponding to target classification
        :return xp: perturbed image that makes the model classify it as the target classification
        """
        w = tf.Variable(tf.random.normal(tf.shape(x)))
        for _ in tqdm(range(self.num_epochs)):
            self.train(x, target, w)
        xp = 0.5 * (tf.tanh(w) + 1)
        return xp

    def f(self, xp, target):
        """
        This is the function f that is minimized to ensure that the perturbed image
        attacks the model successfully
        """
        Z = self.model(xp)
        ret = tf.reduce_max(Z) - Z[target]
        return tf.maximum(0.0, ret)
    
    def train(self, x, target, w):
        """
        Performs one iteration of optimizing the objective function
        """
        with tf.GradientTape() as tape:
            delta = 0.5 * (tf.tanh(w) + 1) - x
            dist_loss = tf.norm(delta)**2
            model_loss = self.f(delta + x, target)
            total_loss = dist_loss + self.c * model_loss
        gradients = tape.gradient(total_loss, w)
        self.optimizer.apply_gradients(zip([gradients], [w]))
        return dist_loss, model_loss, total_loss


def main(model_filepath="../models/vanilla", input_filepath="../data/dataset.pk", output_filepath="../data/l2attack.pk"):

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x, y = x_test[0], y_test[0]

    model = tf.keras.models.load_model(model_filepath, custom_objects={"loss": CNN.loss})
    attack = L2Attack(model)
    xp = attack(x, (y+1)%10)

    with open(output_filepath, "wb") as fd:
        pk.dump(xp, fd)


if __name__ == "__main__":
    main()
