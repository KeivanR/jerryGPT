import numpy as np
import pandas as pd
import time
import random
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras

import models
from settings import *

optimizer = keras.optimizers.Adam()
loss_fn = keras.losses.CategoricalCrossentropy()

train_lines = []
val_lines = []
for text_input in text_inputs:
    if 'WikiQA' in text_input:
        encoding = 'utf-8'
    else:
        encoding = 'unicode_escape'
    with open(text_input, encoding=encoding) as f:
        lines = f.readlines()
    train_lines += lines[int(val_split*len(lines)):]+[' newbook ']
    val_lines += lines[:int(val_split*len(lines))]+[' newbook ']
train_text = ' newline '.join(train_lines)
val_text = ' newline '.join(val_lines)

delim0 = ['" ', ' "', "'", "-", "/", "newline", "_", "—", "‘", " “", "“ "]
delimL = ["(", "£", "`", "{", "~"]
delimR = [",", ".", "!", "?", ":", ")", "$", "*", "}"]
delimRL = [";", "|", "+", "&"]
delimMaj = [' "', ".", "!", "?", "newline"]
all_delims = delim0 + delimL + delimR + delimRL


def text_to_words(text, delim):
    for d in delim:
        text = f' {d} '.join(text.split(d))
    return text.split()


def split_upper(text, d, maj=False):
    split = text.split(f' {d} ')
    if maj:
        for i, w in enumerate(split):
            if len(w) > 0:
                split[i] = w[0].upper() + w[1:]
    return split

def words_to_text(words, d0, dL, dR, dRL, dMaj):
    text = ' '.join(words)
    for d in d0:
        split = split_upper(text, d, maj=d in dMaj)
        text = f'{d}'.join(split)
    for d in dL:
        split = split_upper(text, d, maj=d in dMaj)
        text = f' {d}'.join(split)
    for d in dR:
        split = split_upper(text, d, maj=d in dMaj)
        text = f'{d} '.join(split)
    for d in dRL:
        split = split_upper(text, d, maj=d in dMaj)
        text = f' {d} '.join(split)
    text = text.replace('newline','\n')
    text = text.replace('Newline','\n')
    return text


train_words = text_to_words(train_text, all_delims)
val_words = text_to_words(val_text, all_delims)
vocab, counts = np.unique([w.lower() for w in train_words + val_words], return_counts=True)
vocab = vocab[counts>min_token_count*len(train_words + val_words)]
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
            #   print(w, 'not in vocab')
            tokens.append(vocab_size)
    return tokens


ohmap = np.zeros((vocab_size, vocab_size))
ohmap[np.arange(len(ohmap)), np.arange(len(ohmap))] = 1


def one_hot(tokens):
    onehot = []
    for t in tokens:
        onehot.append(ohmap[t])
    return onehot


print('Train:', len(train_words))
print('Val:', len(val_words))


train_tokens = tokenize(train_words)
val_tokens = tokenize(val_words)

print(len(train_tokens))


def remove_unknown(tokens, t_size):
    tokens = np.array(tokens)
    rollmax = np.array(pd.Series(tokens).rolling(t_size).max().shift(-t_size + 1))
    return tokens[rollmax<vocab_size]


train_tokens = remove_unknown(train_tokens, block_size)
val_tokens = remove_unknown(val_tokens, block_size)
print(len(train_tokens))

def get_batch(b_idx, t_size, training=True):
    x = []
    y = []
    if training:
        tokens = train_tokens
    else:
        tokens = val_tokens
    for i in b_idx:
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
    with tf.GradientTape() as tape:
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
if model_name is None:
    model = models.JerryModel(vocab, block_size)
    model.build(input_shape=(None, None))

    model.compile(optimizer, loss_fn)
    print(model.summary())

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch+1,))
        start_time = time.time()

        train_loss = 0
        batch_idx = np.arange(len(train_tokens) - block_size - 1)
        random.shuffle(batch_idx)
        # Iterate over the batches of the dataset.
        for step in tqdm(range(len(batch_idx) // batch_size)):
            idx = batch_idx[batch_size * step + np.arange(batch_size)]
            x_batch_train, y_batch_train = get_batch(idx, block_size, training=True)
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
        batch_idx = np.arange(len(val_tokens) - block_size - 1)
        random.shuffle(batch_idx)
        for step in range(len(batch_idx) // batch_size):
            idx = batch_size * step + np.arange(batch_size)
            x_batch_val, y_batch_val = get_batch(idx, block_size, training=False)
            val_loss += test_step(x_batch_val, y_batch_val)
        val_loss /= step
        print(
            "Validation loss: %.4f"
            % (float(val_loss))
        )

        print("Time taken: %.2fs" % (time.time() - start_time))

    # model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), callbacks=callbacks)
    model.save(f'Models/{int(time.time())}')
else:
    model = models.JerryModel(vocab, block_size)
    model.build(input_shape=(None, None))
    model.compile(optimizer, loss_fn)
    model.summary()
    model.load_weights(f'Models/{model_name}')
answer = model.answer(None, 20)
print('OUTPUT: ', words_to_text(answer, delim0, delimL, delimR, delimRL, delimMaj))
answer = model.answer(None, 20)
print('OUTPUT: ', words_to_text(answer, delim0, delimL, delimR, delimRL, delimMaj))
answer = model.answer(None, 20)
print('OUTPUT: ', words_to_text(answer, delim0, delimL, delimR, delimRL, delimMaj))
answer = model.answer(None, 20)
print('OUTPUT: ', words_to_text(answer, delim0, delimL, delimR, delimRL, delimMaj))
question = 'Harry Potter was running around the corner'
question_words = text_to_words(question, all_delims)
question_tokens = tokenize(question_words)
answer = model.answer(question_tokens, 20)
print('OUTPUT: ', words_to_text(answer, delim0, delimL, delimR, delimRL, delimMaj))
question = 'He was so tired'
question_words = text_to_words(question, all_delims)
question_tokens = tokenize(question_words)
answer = model.answer(question_tokens, 20)
print('OUTPUT: ', words_to_text(answer, delim0, delimL, delimR, delimRL, delimMaj))
question = 'Why are you saying that'
question_words = text_to_words(question, all_delims)
question_tokens = tokenize(question_words)
answer = model.answer(question_tokens, 20)
print('OUTPUT: ', words_to_text(answer, delim0, delimL, delimR, delimRL, delimMaj))
question = 'It had been a long time'
question_words = text_to_words(question, all_delims)
question_tokens = tokenize(question_words)
answer = model.answer(question_tokens, 20)
print('OUTPUT: ', words_to_text(answer, delim0, delimL, delimR, delimRL, delimMaj))
while question != 'quit':
    question = input('Prompt?')
    n = input("How many tokens?")
    if n == '':
        n = 30
    else:
        n = int(n)
    question_words = text_to_words(question, all_delims)
    question_tokens = tokenize(question_words)
    if question_tokens is not None:
        answer = model.answer(question_tokens, n)
        print('OUTPUT: ', words_to_text(answer, delim0, delimL, delimR, delimRL, delimMaj))
    else:
        print('One word does not exists in the vocab, please try another prompt.')
