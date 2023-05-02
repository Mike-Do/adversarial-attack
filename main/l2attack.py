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
        self.c = kwargs["c"]
        self.num_epochs = kwargs["num_epochs"]
        self.threshold_dist = kwargs["threshold_dist"]
        self.threshold_f = kwargs["threshold_f"]

    def __call__(self, x, target):
        """
        :param x: input image
        :param target: integer corresponding to label target classification
        :return xp: perturbed image that makes the model classify it as the target classification
        """
        x = tf.cast(x, dtype=tf.float32)
        w = tf.Variable(tf.random.normal(tf.shape(x)), dtype=tf.float32)

        if self.num_epochs is None:
            dist_loss, f_loss, _ = self.train(x, target, w)
            epoch = 0
            while (dist_loss > self.threshold_dist) or (f_loss > self.threshold_f):
                dist_loss, f_loss, _ = self.train(x, target, w)
                pred = self.model_prediction(w)[0]
                print(f"Epoch {epoch} | dist loss {dist_loss:.3f} | f loss {f_loss:.3f} | model pred {pred}")
                epoch += 1
            xp = 0.5 * (tf.tanh(w) + 1)
            return xp
        else:
            for _ in tqdm(range(self.num_epochs)):
                dist_loss, f_loss, _ = self.train(x, target, w)
                pred = self.model_prediction(w)
                message = f"dist loss {dist_loss:.3f} | f loss {f_loss:.3f} | model pred {pred}"
                tqdm.write(message)
            xp = 0.5 * (tf.tanh(w) + 1)
            return xp

    def f(self, xp, target):
        """
        This is the function f that is minimized to ensure that the perturbed image
        attacks the model successfully

        f(x)= max(max{Z(x)i:i!=t} - Z(x)t,−κ)

        :param xp: perturbed image of size [BATCH_SIZE, WIDTH, HEIGHT, NUM_CHANNELS]
        :param target: integer corresponding to label of target classification
        """
        xp = tf.expand_dims(xp, axis=0)
        Z = self.model(xp)
        Z = tf.reshape(Z, [10])
        Zt = Z[target]
        Z = tf.concat([Z[:target], Z[target+1:]], axis=0)  # i != t
        ret = tf.reduce_max(Z) - Zt
        return tf.maximum(0.0, ret)

    def train(self, x, target, w):
        """
        Performs one iteration of optimizing the objective function
        """
        with tf.GradientTape() as tape:
            delta = 0.5 * (tf.tanh(w) + 1) - x
            dist_loss = tf.square(tf.norm(delta, ord="euclidean"))
            f_loss = self.f(delta + x, target)
            total_loss = dist_loss + self.c * f_loss
        gradients = tape.gradient(total_loss, w)
        self.optimizer.apply_gradients(zip([gradients], [w]))
        return dist_loss, f_loss, total_loss

    def model_prediction(self, w):
        """
        For debugging information. Given w, finds the model's prediction on w.
        """
        xp = 0.5 * (tf.tanh(w) + 1)
        xp = tf.expand_dims(xp, axis=0)
        pred = self.model(xp)
        pred = tf.nn.softmax(pred, axis=1)
        return tf.argmax(pred, axis=1)


def main(**kwargs):

    index = kwargs["index"]
    target = kwargs["target"]
    model_filepath = kwargs["model_filepath"]
    output_filepath = kwargs["output_filepath"]

    model = tf.keras.models.load_model(model_filepath, custom_objects={"loss": CNN.loss})

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train / 255) - 0.5
    x_train = tf.reshape(x_train, (-1, 28, 28, 1))
    x_test = (x_test / 255) - 0.5
    x_test = tf.reshape(x_test, (-1, 28, 28, 1))

    x = x_test[index]

    assert y_test[index] != target

    attack = L2Attack(model, **kwargs)
    xp = attack(x, target)

    with open(output_filepath, "wb") as fd:
        pk.dump(xp, fd)


if __name__ == "__main__":
    kwargs = {
        "index": 21,
        "target": 3,
        "c": 100,
        "num_epochs": 2500,  # If None, run until it reaches the thresholds
        "threshold_dist": 200.0,
        "threshold_f": 0.01,
        "model_filepath": "../models/vanilla",
        "output_filepath": "../data/l2attack.pk"
    }
    main(**kwargs)
