# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 13:48:24 2023

@author: Yue
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Bidirectional,
    Conv1D,
    TimeDistributed,
    Concatenate,
    Input,
    Flatten,
    MaxPooling1D,
    Dropout
)


class attention(Layer):
    def __init__(self, **kwargs):
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight", shape=(input_shape[-1], 1), initializer="normal"
        )
        self.b = self.add_weight(
            name="att_bias", shape=(input_shape[1], 1), initializer="zeros"
        )
        super(attention, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(attention, self).get_config()
    
def load_model(model_path):
    tf.keras.backend.clear_session()
    input_shape1 = (7, 128, 1)
    input_shape2 = (10, 1)
    input_shape3 = (7, 128, 1)
    input_flow1 = Input(shape=input_shape1)
    input_flow2 = Input(shape=input_shape2)
    input_flow3 = Input(shape=input_shape3)
    
    x = TimeDistributed(Conv1D(64, 64, activation="relu"))(input_flow1)
    
    x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(Flatten())(x)
    
    lstm1 = (LSTM(32, return_sequences=False, dropout=0.4, activation="relu"))(x)
    lstm1 = Dense(32, activation="relu")(lstm1)
    
    lstm2 = Bidirectional(
        LSTM(
            64,
            return_sequences=True,
            dropout=0.4,
            activation="relu",
            recurrent_regularizer=regularizers.l2(0.01),
            bias_regularizer=regularizers.l2(0.01),
        )
    )(input_flow2)
    lstm2 = Bidirectional(
        LSTM(32, return_sequences=True, dropout=0.3, activation="relu")
    )(lstm2)
    lstm2 = attention()(lstm2)
    lstm2 = Dense(32, activation="relu")(lstm2)
    
    x = TimeDistributed(Conv1D(64, 64, activation="relu"))(input_flow3)
    x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(Flatten())(x)
    
    lstm3 = (LSTM(32, return_sequences=False, dropout=0.4, activation="relu"))(x)
    lstm3 = Dense(32, activation="relu")(lstm3)
    
    lstm = Concatenate(axis=1)([lstm1, lstm2, lstm3])
    
    out = Dense(3, activation="softmax")(lstm)
    model = Model(inputs=[input_flow1, input_flow2, input_flow3], outputs=out)
    #model.summary()
    
    # 400
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(0.0001),
        metrics=["accuracy"],
    )

    model.load_weights(model_path)
    return model