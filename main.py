import numpy as np
import time
import random
import tensorflow as tf
from tensorflow import keras

import models
from settings import *

optimizer = keras.optimizers.Adam()
loss_fn = keras.losses.CategoricalCrossentropy()

with open(text_input) as f:
    lines = f.readlines()
text = ''.join(lines)

delim = [",", ".", "'", "!", "?", ";", '"', "-", ":", "(", ")", "£", "$", "/", "\\"]


def text_to_words(text, delim):
    for d in delim:
        text = f' {d} '.join(text.split(d))
    return text.split()


text_words = text_to_words(text, delim)
vocab = np.unique([w.lower() for w in text_words])
vocab_size = len(vocab)
for v in vocab:
    print(v)
print('Vocabulary size: ', vocab_size)

token_map = [[w, i] for w, i in zip(vocab, range(vocab_size))]
token_map = dict(token_map)


def tokenize(words):
    tokens = []
    for w in words:
        if w.lower() in token_map.keys():
            tokens.append(token_map[w.lower()])
        else:
            return None
    return tokens


ohmap = np.zeros((vocab_size, vocab_size))
ohmap[np.arange(len(ohmap)), np.arange(len(ohmap))] = 1


def one_hot(tokens):
    onehot = []
    for t in tokens:
        onehot.append(ohmap[t])
    return onehot


train_words = text_words[int(val_split * len(text_words)):]
print('Train:', len(train_words))
val_words = text_words[:int(val_split * len(text_words))]
print('Val:', len(val_words))

train_tokens = tokenize(train_words)
val_tokens = tokenize(val_words)


def get_batch(b_size, t_size, training=True):
    x = []
    y = []
    if training:
        tokens = train_tokens
    else:
        tokens = val_tokens
    idx = np.random.choice(len(tokens) - t_size - 1, b_size)
    for i in idx:
        x.append(tokens[i:i + t_size])
        y.append(one_hot(tokens[i + 1:i + t_size + 1]))
    return np.array(x), np.array(y)


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value


@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    loss_value = loss_fn(y, val_logits)
    return loss_value


callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=2,
        verbose=1,
    )
]

model = models.JerryModel(vocab, block_size)
model.build(input_shape=(None, None))

model.compile(optimizer, loss_fn)
print(model.summary())

epochs = 8
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    train_loss = 0
    # Iterate over the batches of the dataset.
    for step in range(len(train_tokens) // batch_size):
        x_batch_train, y_batch_train = get_batch(batch_size, block_size, training=True)
        loss_value = train_step(x_batch_train, y_batch_train)
        train_loss += loss_value

        # # Log every 200 batches.
        # if step % 200 == 0:
        #     print(
        #         "Training loss (for one batch) at step %d: %.4f"
        #         % (step, float(loss_value))
        #     )
        #     print("Seen so far: %d samples" % ((step + 1) * batch_size))
    train_loss /= step
    print(
        "Training loss: %.4f"
        % (float(train_loss))
    )

    # Run a validation loop at the end of each epoch.
    val_loss = 0
    for step in range(len(val_tokens) // batch_size):
        x_batch_val, y_batch_val = get_batch(batch_size, block_size, training=False)
        val_loss += test_step(x_batch_val, y_batch_val)
    val_loss /= step
    print(
        "Validation loss: %.4f"
        % (float(val_loss))
    )

    print("Time taken: %.2fs" % (time.time() - start_time))

# model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), callbacks=callbacks)


model.answer(None, 20)
model.answer(None, 20)
model.answer(None, 20)
model.answer(None, 20)
question = 'Harry Potter was running around the corner'
question_words = text_to_words(question, delim)
question_tokens = tokenize(question_words)
model.answer(question_tokens, 20)
question = 'He was so tired'
question_words = text_to_words(question, delim)
question_tokens = tokenize(question_words)
model.answer(question_tokens, 20)
question = 'Why are you saying that'
question_words = text_to_words(question, delim)
question_tokens = tokenize(question_words)
model.answer(question_tokens, 20)
question = 'It had been a long time'
question_words = text_to_words(question, delim)
question_tokens = tokenize(question_words)
model.answer(question_tokens, 20)
while question != 'quit':
    question = input('Prompt?')
    question_words = text_to_words(question, delim)
    question_tokens = tokenize(question_words)
    if question_tokens is not None:
        model.answer(question_tokens, 30)
    else:
        print('One word does not exists in the vocab, please try another prompt.')
