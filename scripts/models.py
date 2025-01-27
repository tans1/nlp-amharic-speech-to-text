import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Input, Conv2D, Dense, LSTM, MaxPooling2D, Bidirectional

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred

def model_1(encoder):
    input_img = layers.Input(
        shape=(288, 432, 4), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    if tf.__version__ >= '2.3':
        new_shape = (x.type_spec.shape[-3], x.type_spec.shape[-2]*x.type_spec.shape[-1])
    else:
        new_shape = (x.shape[-3], x.shape[-2]*x.shape[-1])
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = layers.Dense(
        len(encoder.classes_) + 1, activation="softmax", name="dense2"
    )(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = tf.keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    # Optimizer
    opt = tf.keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt)
    
    return model

def model_2(encoder):
    input_img = layers.Input(
        shape=(480, 640, 4), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # First conv block
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    if tf.__version__ >= '2.3':
        new_shape = (x.type_spec.shape[-3], x.type_spec.shape[-2]*x.type_spec.shape[-1])
    else:
        new_shape = (x.shape[-3], x.shape[-2]*x.shape[-1])
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    x = layers.Dense(
        len(encoder.classes_) + 1, activation="softmax", name="dense2"
    )(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = tf.keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    opt = tf.keras.optimizers.Adam()
    model.compile(optimizer=opt)
    
    return model

def model_3(encoder):
    input_img = layers.Input(
        shape=(480, 640, 4), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.BatchNormalization()(x)
    if tf.__version__ >= '2.3':
        new_shape = (x.type_spec.shape[-3], x.type_spec.shape[-2]*x.type_spec.shape[-1])
    else:
        new_shape = (x.shape[-3], x.shape[-2]*x.shape[-1])
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(96, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.BatchNormalization()(x)

    # Output layer
    x = layers.Dense(
        len(encoder.classes_) + 1, activation="softmax", name="dense2"
    )(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = tf.keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    # Optimizer
    opt = tf.keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt)
    
    return model

def model_4(encoder):
    input_img = layers.Input(
        shape=(480, 640, 4), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # First conv block
    x = layers.Conv2D(
        64,
        (5, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.BatchNormalization()(x)

    # First conv block
    x = layers.Conv2D(
        96,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)


    if tf.__version__ >= '2.3':
        new_shape = (x.type_spec.shape[-3], x.type_spec.shape[-2]*x.type_spec.shape[-1])
    else:
        new_shape = (x.shape[-3], x.shape[-2]*x.shape[-1])
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)

    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)


    x = layers.GRU(128, return_sequences=True, dropout=0.25)(x)
    x = layers.BatchNormalization()(x)
    x = layers.GRU(128, return_sequences=True, dropout=0.25)(x)
    x = layers.BatchNormalization()(x)

    # Output layer
    x = layers.TimeDistributed(layers.Dense(
        len(encoder.classes_) + 1, activation="softmax", name="dense3"
    ))(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = tf.keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    # Optimizer
    opt = tf.keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt)
    
    return model