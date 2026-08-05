"""TensorFlow: create the model INSIDE autotrainer.scope().
Run: autotrainer run tensorflow_scope.py
SLURM multi-node: srun autotrainer run tensorflow_scope.py (TF_CONFIG auto-generated)

The ``if __name__ == "__main__":`` guard keeps this consistent with the other
examples and safe under process-spawning launchers.
"""

import numpy as np
import tensorflow as tf

import autotrainer


def main() -> None:
    # autotrainer.scope() picks MirroredStrategy (local multi-GPU) or MultiWorkerMirroredStrategy (SLURM)
    with autotrainer.scope():
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(10),
            ]
        )
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    X = np.random.randn(2048, 32).astype("float32")
    y = np.random.randint(0, 10, 2048)

    # 1-LINE EXECUTION: scales batch size per replica, runs training, & saves model.keras!
    #
    # Options for TensorFlow in autotrainer.train():
    #   - epochs=3                    : Number of training epochs
    #   - save_path="model.keras"     : Auto-saves model via model.save()
    #   - patience=5                  : Injects EarlyStopping(restore_best_weights=True)
    autotrainer.train(model, X, y, epochs=3, save_path="model.keras")


if __name__ == "__main__":
    main()
