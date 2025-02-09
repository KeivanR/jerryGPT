import tensorflow as tf
from tensorflow import keras
from settings import *
import numpy as np


def sincos_encoding(blocksize, d, N):
    res = np.zeros((blocksize, d))
    idx = 2*np.arange(d//2)
    w = 1/N**(idx/d)
    wt = np.outer(np.arange(blocksize),w)
    res[:, idx] = np.sin(wt)
    idx = 2*np.arange(d//2)+1
    w = 1/N**(idx/d)
    wt = np.outer(np.arange(blocksize),w)
    res[:, idx] = np.cos(wt)
    return res

class JerryModel(keras.Model):
    def __init__(self, vocab, blocksize):
        super().__init__()
        self.vocab = vocab
        self.blocksize = blocksize
        self.token_embedding = keras.layers.Embedding(len(vocab), 32)
        self.pos_embedding = keras.layers.Embedding(blocksize, 32)
        self.sincos_enc = sincos_encoding(blocksize, 32, 10000)
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(len(vocab), activation="softmax",kernel_regularizer=keras.regularizers.l2(l=0.1))
        self.norm1 = keras.layers.BatchNormalization()
        self.norm2 = keras.layers.BatchNormalization()
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(0.2)
        self.dropout2 = keras.layers.Dropout(0.2)
        self.multihead = keras.layers.MultiHeadAttention(6, 16,kernel_regularizer=keras.regularizers.l2(l=0.1))
        # self.lstm = keras.layers.LSTM(8, return_sequences=True)

    def call(self, inputs, training=False):
        t_emb = self.token_embedding(inputs, training=training)
        #p_emb = self.pos_embedding(tf.range(self.blocksize), training=training)
        x = t_emb + self.sincos_enc
        # x = self.norm1(x, training=training)
        x = self.layernorm1(x, training=training)
        x = self.multihead(x, x, use_causal_mask=True, training=training)
        x = self.norm2(x, training=training)
        x = self.dropout2(x, training=training)
        return self.dense2(x, training=training)

    def answer(self, question, size):
        block = question
        if block is None:
            block = list(np.random.choice(len(self.vocab), self.blocksize))
        if len(block) < self.blocksize:
            block = [list(self.vocab).index('newline')]*(self.blocksize - len(block)) + block #list(np.random.choice(len(self.vocab), self.blocksize - len(block))) + block
        elif len(block)>self.blocksize:
            block = block[len(block)-self.blocksize:]
        print('INPUT: ', ' '.join([self.vocab[w] for w in block]))
        generated = []
        for i in range(size):
            probs = self.call(tf.convert_to_tensor([block]))[0][-1]
            new_word = np.random.choice(len(self.vocab), p=np.array(probs))
            block = block[1:] + [new_word]
            generated.append(self.vocab[new_word])
        return generated

    def freeze_base(self, freeze):
        trainable = not freeze
        # self.norm1.trainable = trainable
        self.layernorm1.trainable = trainable
        self.multihead.trainable = trainable
        self.norm2.trainable = trainable
        self.dropout2.trainable = trainable
