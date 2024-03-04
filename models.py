import tensorflow as tf
from tensorflow import keras
from settings import *
import numpy as np


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


class JerryModel(keras.Model):
    def __init__(self, vocab, blocksize):
        super().__init__()
        self.vocab = vocab
        self.blocksize = blocksize
        self.embedding = keras.layers.Embedding(len(vocab), 32)
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(len(vocab), activation="softmax")
        self.norm1 = keras.layers.BatchNormalization()
        self.norm2 = keras.layers.BatchNormalization()
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(0.1)
        self.dropout2 = keras.layers.Dropout(0.1)
        self.multihead = keras.layers.MultiHeadAttention(2, 8)
        self.lstm = keras.layers.LSTM(8, return_sequences=True)
        self.look_ahead_mask = create_look_ahead_mask(blocksize)

    def call(self, inputs, training=False):
        x = self.embedding(inputs, training=training)
        x = self.norm1(x, training=training)
        x = self.layernorm1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.norm2(x, training=training)
        x = self.multihead(x, x, use_causal_mask=True, training=training)
        x = self.dropout2(x, training=training)
        return self.dense2(x, training=training)

    def answer(self, question, size):
        block = question
        if block is None:
            block = list(np.random.choice(len(self.vocab), self.blocksize))
        if len(block) < self.blocksize:
            block = list(np.random.choice(len(self.vocab), self.blocksize - len(block))) + block
        elif len(block)>self.blocksize:
            block = block[len(block)-self.blocksize:]
        print('INPUT: ', ' '.join([self.vocab[w] for w in block]))
        generated = ''
        for i in range(size):
            probs = self.call(tf.convert_to_tensor([block]))[0][-1]
            new_word = np.random.choice(len(self.vocab), p=np.array(probs))
            block = block[1:] + [new_word]
            generated += ' ' + self.vocab[new_word]
        print('OUTPUT: ', generated)
