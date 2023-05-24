
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


# par exemple X_train.shape = (1000, 32 * 32 * 3)

# par exemple Y_train.shape = (1000, 3)

class MyModel:
    def get_trainable_variables(self):
        raise NotImplementedError

    def predict(self, X_batch):
        raise NotImplementedError

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray, epochs: int, batch_size: int = 100, alpha: float = 0.01):
        for epoch_id in range(epochs):
            for batch_id in range(len(X_train) // batch_size + 1):
                X_batch = X_train[batch_id * batch_size: (batch_id + 1) * batch_size]
                Y_batch_true = Y_train[batch_id * batch_size: (batch_id + 1) * batch_size]
                with tf.GradientTape() as tape:
                    Y_batch_pred = self.predict(X_batch)
                    loss = tf.reduce_mean((Y_batch_true - Y_batch_pred) ** 2)
                trainable_variables = self.get_trainable_variables()
                grads = tape.gradient(loss, trainable_variables)
                for (grad, v) in zip(grads, trainable_variables):
                    v.assign_sub(alpha * grad)


class LinearRegressionModel(MyModel):
    def __init__(self, input_dim: int):
        self.W = tf.Variable(tf.random.uniform((input_dim, 1), -1.0, 1.0), dtype=tf.float32)
        self.b = tf.Variable(0.0, tf.float32)

    def get_trainable_variables(self):
        return [self.W, self.b]

    def predict(self, X_batch):
        if len(X_batch.shape) == 1:
            X_batch = tf.expand_dims(X_batch, -1)
        return tf.squeeze(tf.matmul(X_batch, self.W) + self.b)


def main():
    points_X = []
    points_Y = []
    for i in range(100):
        x = i / 100.0
        points_X.append(x)
        points_Y.append(3 * (x * 3) ** 2 + 4 + np.random.uniform(-0.5, 0.5))

    points_X = tf.constant(points_X, dtype=tf.float32)
    points_Y = tf.constant(points_Y, dtype=tf.float32)

    model = LinearRegressionModel(1)

    predicted_y = model.predict(points_X)

    nb_iter = 50

    fig, plots = plt.subplots(nb_iter + 1)
    fig.set_size_inches(5, 250)
    plots[0].scatter(points_X, points_Y, c='b')
    plots[0].scatter(points_X, predicted_y, c='r')

    for i in range(50):
        model.fit(points_X, points_Y, 1, 10, 0.005)


        predicted_y = model.predict(points_X)

        plots[i + 1].scatter(points_X, points_Y, c='b')
        plots[i + 1].scatter(points_X, predicted_y, c='r')

    plt.show()

if __name__ == "__main__":
    main()
