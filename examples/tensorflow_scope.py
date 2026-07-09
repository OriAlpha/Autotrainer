"""TensorFlow: create the model INSIDE autotrainer.scope().
Run: autotrainer run tensorflow_scope.py
SLURM multi-node: srun autotrainer run tensorflow_scope.py (TF_CONFIG auto-generated)
"""
import numpy as np
import tensorflow as tf

import autotrainer

with autotrainer.scope():                       # picks the right tf.distribute strategy
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10),
    ])
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

bs = autotrainer.scale_batch_size(64)           # per-replica 64 -> global batch
X = np.random.randn(2048, 32).astype("float32")
y = np.random.randint(0, 10, 2048)
model.fit(X, y, batch_size=bs, epochs=3)
