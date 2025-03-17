import tensorflow as tf
from tensorflow.keras import layers, Model, ops

def create_efficientnetb3(input_shape=(250,64,1)):
    base_effnet = tf.keras.applications.EfficientNetB3(
        include_top=False,
        input_shape=input_shape,
        weights=None
    )

    x = base_effnet.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_effnet.input, outputs=outputs)

    model.compile(
        optimizer="adam", 
        loss="binary_crossentropy", 
        metrics=[
            tf.keras.metrics.F1Score(average="micro", threshold=0.5, name="f1_score"),
            tf.keras.metrics.AUC(curve="PR", name="auprc")
        ]
    )

    return model


def create_model_uncompressed(input_shape=(250, 64, 1)):    
    inputs = tf.keras.Input(shape=input_shape)
    
    # First convolution block
    x = layers.Conv2D(filters=8, kernel_size=(3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.4)(x)
    
    # Second convolution block
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.4)(x)
    
    # Third convolution block
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.4)(x)

    # Fourth convolution block
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.4)(x)

    # Fifth convolution block
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.4)(x)

    # Flatten and classification head
    x = layers.Flatten()(x)
    x = layers.Dense(8, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer="adam", 
        loss="binary_crossentropy", 
        metrics=[
            tf.keras.metrics.F1Score(average="micro", threshold=0.5, name="f1_score"),
            tf.keras.metrics.AUC(curve="PR", name="auprc"),
        ]
    )
    
    return model

# Distiller class from keras.io/examples/vision/knowledge_distillation/
class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=1,
    ):
        """Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False
    ):
        teacher_pred = self.teacher(x, training=False)
        student_loss = self.student_loss_fn(y, y_pred)

        distillation_loss = self.distillation_loss_fn(
            ops.softmax(teacher_pred / self.temperature, axis=1),
            ops.softmax(y_pred / self.temperature, axis=1),
        ) * (self.temperature**2)

        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        return loss

    def call(self, x):
        return self.student(x)
