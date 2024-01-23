# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import numpy as np
import os
import random
import sys

from argparse import ArgumentParser
from collections import Counter
from random import randrange
from sklearn.linear_model import LinearRegression
from tiktoken import Encoding
from tqdm import tqdm

from tiktoken_ext.openai_public import ENCODING_CONSTRUCTORS


def main(argv):
    parser = ArgumentParser()

    parser.add_argument('--test')
    parser.add_argument('--train', nargs='+', required=True)

    parser.add_argument('--split-fraction', type=float, default=0.2,
                        help='If no test data is provided, use this random fraction of the training data instead.')

    parser.add_argument('--out-coeffs', required=True)
    parser.add_argument('--out-eval')

    args = parser.parse_args(argv)

    tokenizer = load_tokenizer()

    if args.test:
        train_x, train_y = load_data(args.train, tokenizer)
        test_x, test_y = load_data(args.test, tokenizer)
    else:
        train_x, train_y = load_data(args.train, tokenizer)
        train_x, train_y, test_x, test_y = split_data(train_x, train_y, args.split_fraction)

    print(f'Train: {train_y.shape[0]} samples')
    print(f'Test: {test_y.shape[0]} samples')

    bytes_model = train_regression_model(train_x['bytes'], train_y)
    chars_model = train_regression_model(train_x['chars'], train_y)
    words_model = train_regression_model(train_x['words'], train_y)

    print_stats(bytes_model, test_x['bytes'], test_y, name='bytes')
    print_stats(chars_model, test_x['chars'], test_y, name='chars')
    print_stats(words_model, test_x['words'], test_y, name='words')

    write_coeffs(bytes_model, args.out_coeffs)

    write_eval([
        test_y,
        bytes_model.predict(test_x['bytes']),
        chars_model.predict(test_x['chars']),
        words_model.predict(test_x['words']),
    ], args.out_eval)


def load_tokenizer():
    return Encoding(**ENCODING_CONSTRUCTORS['cl100k_base']())


def load_data(paths: 'list[str]', tokenizer: Encoding, *, sample: bool = True):
    encoded_x = {
        'bytes': [], # Histogram
        'chars': [], # Total count
        'words': [], # Total count
    }

    encoded_y = []

    for prompt in load_prompts(paths):
        if sample:
            prompt = sample_from_prompt(prompt)

        byte_counts = Counter(prompt.encode('utf-8'))

        encoded_x['bytes'].append(np.array([byte_counts[i] for i in range(0x100)], np.int32))
        encoded_x['chars'].append([len(prompt)])
        encoded_x['words'].append([len(prompt.split())])

        encoded_y.append(len(tokenizer.encode(prompt, disallowed_special=())))

    encoded_x = {k: np.stack(v) for k, v in encoded_x.items()}
    encoded_y = np.stack(encoded_y)

    return encoded_x, encoded_y


def load_prompts(paths: 'list[str]'):
    for path in tqdm(paths, desc='Input files'):
        with open(path, 'rb') as f:
            with tqdm(desc='Input data', total=os.stat(f.fileno()).st_size, unit='B', unit_scale=True, leave=None) as progress:
                for line in f:
                    try:
                        yield json.loads(line)['prompt']
                    except json.JSONDecodeError:
                        pass

                    progress.n = f.tell()
                    progress.refresh()


# Slice out a random part of the prompt.
# We don't take the whole thing because prompt lengths are too correlated
# (e.g., all the prompts in a dataset having lengths in a range 2000-2100),
# making the resulting model rely too much on the average.
def sample_from_prompt(prompt: str):
    start = randrange(0, len(prompt)//2)
    end = randrange(len(prompt)//2, len(prompt) + 1)

    return prompt[start:end]


def split_data(x, y, test_fraction: float):
    indices = list(range(y.shape[0]))
    random.shuffle(indices)
    split = int(len(indices) * test_fraction)

    train_x = {k: v[indices[split:]] for k, v in x.items()}
    train_y = y[indices[split:]]

    test_x = {k: v[indices[:split]] for k, v in x.items()}
    test_y = y[indices[:split]]

    return train_x, train_y, test_x, test_y


def train_regression_model(x, y):
    model = LinearRegression(fit_intercept=False, positive=True)
    model.fit(x, y)

    return model


def print_stats(model: LinearRegression, test_x, test_y, *, name=''):
    pred_y = model.predict(test_x)

    print()
    print(f'=====')
    print(f'Model {name}')
    print(f'=====')
    print()
    print(f'Coefficients:', model.coef_, sep='\n')
    print(f'---')
    print(f'Min: {test_y.min()}')
    print(f'Avg: {int(test_y.mean())}')
    print(f'Max: {test_y.max()}')
    print(f'---')
    print(f'R^2:  {model.score(test_x, test_y)}')
    print(f'MSE:  {np.square(pred_y - test_y).mean()}')
    print(f'MAE%: {100*(np.abs(pred_y - test_y)/test_y).mean()}')
    print(f'---')
    print(f'Min AE: {np.abs(pred_y - test_y).min()}')
    print(f'Max AE: {np.abs(pred_y - test_y).max()}')
    print(f'---')
    print(f'Min AE%: {100*(np.abs(pred_y - test_y)/test_y).min()}')
    print(f'Max AE%: {100*(np.abs(pred_y - test_y)/test_y).max()}')


def write_coeffs(model: LinearRegression, path: str):
    with open(path, 'w') as f:
        json.dump(model.coef_.tolist(), f, indent=4)


def write_eval(y, path: str):
    if not path:
        return

    with open(path, 'w') as f:
        for row in zip(*y):
            print(*map(int, row), sep=',', file=f)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

