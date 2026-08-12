import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import mlx.utils as util

from main import Model

class Classification(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 2)

    def __call__(self, x: mx.array): return self.proj(x)

def cola(filepath: str):
    data = []
    with open(filepath, 'r', encoding = 'utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 4: data.append((parts[3].encode('utf-8'), int(parts[1])))
    return data

def mcc(tp, tn, fp, fn):
    import math
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denominator if denominator != 0 else 0.0

def run():
    model = Model(dim = 512, layers = 16, temp = 0.75, lr = 5e-4)
    model.load('smaller-4.5m.safetensors')
    model.freeze()

    head = Classification(model.dim)
    headopt = opt.AdamW(learning_rate = 1e-3)

    data = cola('CoLA/original/raw/in_domain_train.tsv')

    def l(params, state: mx.array, target: int):
        head.update(params)
        choice = head(state)

        loss = nn.losses.cross_entropy(choice[None, :], mx.array([target])).mean()
        return loss, choice

    for epoch in range(3):
        print(f'\nEpoch {epoch + 1}')

        dummies = [mx.zeros((model.dim, )) for _ in range(model.layercount)]
        tp, tn, fp, fn = 0, 0, 0, 0
        
        for i, (bytes, label) in enumerate(data):
            final = None
            for b in bytes:
                x = model.encoder(mx.array(b))
                for j, layer in enumerate(model.layers):
                    x, state, _ = layer(x, dummies[j])
                    layer.states = mx.stop_gradient(state)
                
                final = model.layers[-1].states

            (_, choice), grads = mx.value_and_grad(l, argnums = 0)(head.trainable_parameters(), final, label)

            headopt.update(head, grads)
            mx.eval(head.parameters(), headopt.state)

            predicted_class = mx.argmax(choice).item()
            if predicted_class == 1 and label == 1: tp += 1
            elif predicted_class == 0 and label == 0: tn += 1
            elif predicted_class == 1 and label == 0: fp += 1
            elif predicted_class == 0 and label == 1: fn += 1

            score = mcc(tp, tn, fp, fn)

            if i > 0 and i % 500 == 0: print(f'{i + 1}: TP, TN, FP, FN | {tp}, {tn}, {fp}, {fn} ({score})')

        print(f'{i + 1}: TP, TN, FP, FN | {tp}, {tn}, {fp}, {fn} ({score})')

if __name__ == '__main__':
    run()