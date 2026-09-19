import math
import os

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt

from main import Model

class Classification(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 2)

    def __call__(self, x: mx.array): return self.proj(x)

def cola(filepath: str):
    data = []

    try:
        with open(filepath, 'r', encoding = 'utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 4: data.append((parts[3].encode('utf-8'), int(parts[1])))

    except FileNotFoundError: pass
    
    return data

def mcc(tp, tn, fp, fn):
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    score = (tp * tn - fp * fn) / denominator if denominator != 0 else 0.0
    return score * 100

def rollout(model: Model, b_s: bytes, dummies: list[mx.array]):
    model.reset()

    final = None
    for b in b_s:
        enc = model.encoder(mx.array(b))
        x = enc

        for j, layer in enumerate(model.layers):
            x, state, _ = layer(enc, x, dummies[j])
            layer.states = mx.stop_gradient(state)

        final = model.layers[-1].states

    if final is not None:
        mx.eval(final)
    return final

def run(path: str, data: str = 'CoLA/original/raw/in_domain_train.tsv', epochs: int = 3, dev: float = 0.1):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found at {path!r}. Train a model first "
            f"(python main.py --mode train) or pass --path to an existing .safetensors file."
        )

    model = Model(dim = 512, layers = 16, temp = 0.75, lr = 5e-4)
    model.load(path)
    model.freeze()

    head = Classification(model.dim)
    headopt = opt.AdamW(learning_rate = 1e-3)

    rows = cola(data)

    if rows == []:
        print(f'invalid CoLA dataset at {data!r}.')
        print('Download it from https://nyu-mll.github.io/CoLA/ (e.g. CoLA.zip -> CoLA/original/raw/in_domain_train.tsv).')
        return

    if not 0.0 <= dev < 1.0:
        raise ValueError(f'dev must be in [0.0, 1.0), got {dev!r}.')
    if len(rows) < 2:
        raise ValueError(f'Need at least 2 usable CoLA rows, got {len(rows)}.')

    split = int(len(rows) * (1.0 - dev))
    if not 1 <= split < len(rows):
        raise ValueError(
            f'dev={dev!r} leaves no usable split for {len(rows)} rows.'
        )
    train, heldout = rows[:split], rows[split:]

    def lossfn(params, state: mx.array, target: int):
        head.update(params)
        choice = head(state)

        loss = nn.losses.cross_entropy(choice[None, :], mx.array([target])).mean()
        return loss, choice

    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}')

        dummies = [mx.zeros((model.dim, )) for _ in range(model.layercount)]
        tp, tn, fp, fn = 0, 0, 0, 0
        score = 0.0

        for i, (b_s, label) in enumerate(train):
            if len(b_s) == 0:
                continue
            final = rollout(model, b_s, dummies)
            if final is None:
                continue

            (_, choice), grads = mx.value_and_grad(lossfn, argnums = 0)(head.trainable_parameters(), final, label)

            headopt.update(head, grads)
            mx.eval(head.parameters(), headopt.state)

            predicted_class = mx.argmax(choice).item()
            if predicted_class == 1 and label == 1: tp += 1
            elif predicted_class == 0 and label == 0: tn += 1
            elif predicted_class == 1 and label == 0: fp += 1
            elif predicted_class == 0 and label == 1: fn += 1

            score = mcc(tp, tn, fp, fn)

            if i > 0 and i % 500 == 0: print(f'{i}: T+ {tp}, T- {tn}, F+ {fp}, F- {fn} ({score:.4f})')

        print(f'train {len(train)}: T+ {tp}, T- {tn}, F+ {fp}, F- {fn} ({score:.4f})')

        # Frozen dev evaluation (no head updates).
        dtp, dtn, dfp, dfn = 0, 0, 0, 0
        for b_s, label in heldout:
            if len(b_s) == 0:
                continue
            final = rollout(model, b_s, dummies)
            if final is None:
                continue
            choice = head(final)
            predicted_class = mx.argmax(choice).item()
            if predicted_class == 1 and label == 1: dtp += 1
            elif predicted_class == 0 and label == 0: dtn += 1
            elif predicted_class == 1 and label == 0: dfp += 1
            elif predicted_class == 0 and label == 1: dfn += 1
        print(f'dev {len(heldout)}: T+ {dtp}, T- {dtn}, F+ {dfp}, F- {dfn} ({mcc(dtp, dtn, dfp, dfn):.4f})')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CoLA probe for a frozen TMT backbone.')
    parser.add_argument('--path', default='experimental-4.5m.safetensors')
    parser.add_argument('--data', default='CoLA/original/raw/in_domain_train.tsv')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--dev', type=float, default=0.1)
    args = parser.parse_args()

    run(args.path, args.data, args.epochs, args.dev)