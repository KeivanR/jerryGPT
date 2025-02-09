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
    text = text.replace('newline', '\n')
    text = text.replace('Newline', '\n')
    return text


def tokenize(words, vocabulary):
    tokens = []
    vocab_size = len(vocabulary)
    token_map = [[w, i] for w, i in zip(vocabulary, range(vocab_size))]
    token_map = dict(token_map)
    for w in words:
        if w.lower() in token_map.keys():
            tokens.append(token_map[w.lower()])
        else:
            #   print(w, 'not in vocab')
            tokens.append(vocab_size)
    return tokens


def one_hot(tokens, vocab_size):
    ohmap = np.zeros((vocab_size, vocab_size))
    ohmap[np.arange(len(ohmap)), np.arange(len(ohmap))] = 1
    onehot = []
    for t in tokens:
        onehot.append(ohmap[t])
    return onehot


def remove_unknown(tokens, t_size, vocab_size):
    tokens = np.array(tokens)
    rollmax = np.array(pd.Series(tokens).rolling(t_size).max().shift(-t_size + 1))
    return tokens[rollmax < vocab_size]


def words_from_texts(inputs, split=.2):
    train_lines = []
    val_lines = []
    for text_input in inputs:
        if 'WikiQA' in text_input:
            encoding = 'utf-8'
        else:
            encoding = 'unicode_escape'
        with open(text_input, encoding=encoding) as f:
            lines = f.readlines()
        train_lines += lines[int(split * len(lines)):] + [' newbook ']
        val_lines += lines[:int(split * len(lines))] + [' newbook ']
    train_text = ' newline '.join(train_lines)
    val_text = ' newline '.join(val_lines)

    train_w = text_to_words(train_text, all_delims)
    val_w = text_to_words(val_text, all_delims)
    vocabulary, counts = np.unique([w.lower() for w in train_w + val_w], return_counts=True)
    vocabulary = vocabulary[counts > min_token_count * len(train_w + val_w)]
    print('Vocabulary size: ', len(vocabulary))
    print('Train:', len(train_w))
    print('Val:', len(val_w))
    return train_w, val_w, vocabulary


def tokens_from_words(words, vocabulary):
    tokens = tokenize(words, vocabulary)
    print(len(tokens))
    tokens = remove_unknown(tokens, block_size, len(vocabulary))
    print(len(tokens))
    return tokens


def get_batch(b_idx, t_size, tokens):
    x = []
    y = []
    for i in b_idx:
        x.append(tokens[i:i + t_size])
        y.append(one_hot(tokens[i + 1:i + t_size + 1], len(vocab)))
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


def train_model(model, train_tok, val_tok, epochs, vocabulary):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch + 1,))
        start_time = time.time()
        train_loss = 0
        batch_idx = np.arange(len(train_tok) - block_size - 1)
        random.shuffle(batch_idx)
        # Iterate over the batches of the dataset.
        for step in tqdm(range(len(batch_idx) // batch_size)):
            idx = batch_idx[batch_size * step + np.arange(batch_size)]
            x_batch_train, y_batch_train = get_batch(idx, block_size, train_tok)
            loss_value = train_step(x_batch_train, y_batch_train)
            train_loss += loss_value
        train_loss /= step
        print(
            "Training loss: %.4f"
            % (float(train_loss))
        )
        if val_tok is not None:
            # Run a validation loop at the end of each epoch.
            val_loss = 0
            batch_idx = np.arange(len(val_tok) - block_size - 1)
            random.shuffle(batch_idx)
            for step in range(len(batch_idx) // batch_size):
                idx = batch_size * step + np.arange(batch_size)
                x_batch_val, y_batch_val = get_batch(idx, block_size, val_tok)
                val_loss += test_step(x_batch_val, y_batch_val)
            val_loss /= step
            print(
                "Validation loss: %.4f"
                % (float(val_loss))
            )
        print("Time taken: %.2fs" % (time.time() - start_time))
    model.save(f'Models/{int(time.time())}')


train_words, val_words, vocab = words_from_texts(text_inputs, split=val_split)
train_tokens = tokens_from_words(train_words, vocab)
val_tokens = tokens_from_words(val_words, vocab)

if model_name is None:
    model = models.JerryModel(vocab, block_size)
    model.build(input_shape=(None, None))
    model.compile(optimizer, loss_fn)
    model.summary()
    train_model(model, train_tokens, val_tokens, n_epochs, vocab)
else:
    model = models.JerryModel(vocab, block_size)
    model.build(input_shape=(None, None))
    model.compile(optimizer, loss_fn)
    model.summary()
    model.load_weights(f'Models/{model_name}')

if fine_tuning is not None:
    ft_words, _, _ = words_from_texts(fine_tuning, split=0)
    ft_tokens = tokens_from_words(ft_words, vocab)
    model.freeze_base(True)
    train_model(model, ft_tokens, None, n_epochs_fine_tuning, vocab)

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
question_tokens = tokenize(question_words, vocab)
answer = model.answer(question_tokens, 20)
print('OUTPUT: ', words_to_text(answer, delim0, delimL, delimR, delimRL, delimMaj))
question = 'He was so tired'
question_words = text_to_words(question, all_delims)
question_tokens = tokenize(question_words, vocab)
answer = model.answer(question_tokens, 20)
print('OUTPUT: ', words_to_text(answer, delim0, delimL, delimR, delimRL, delimMaj))
question = 'Why are you saying that'
question_words = text_to_words(question, all_delims)
question_tokens = tokenize(question_words, vocab)
answer = model.answer(question_tokens, 20)
print('OUTPUT: ', words_to_text(answer, delim0, delimL, delimR, delimRL, delimMaj))
question = 'It had been a long time'
question_words = text_to_words(question, all_delims)
question_tokens = tokenize(question_words, vocab)
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
    question_tokens = tokenize(question_words, vocab)
    if question_tokens is not None:
        answer = model.answer(question_tokens, n)
        print('OUTPUT: ', words_to_text(answer, delim0, delimL, delimR, delimRL, delimMaj))
    else:
        print('One word does not exists in the vocab, please try another prompt.')
