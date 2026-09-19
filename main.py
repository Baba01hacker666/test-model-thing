import glob
import itertools
import os
import sys
import time
from datetime import datetime

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import mlx.utils as util

class Encoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.embed = nn.Embedding(256, dim)

    def __call__(self, x: mx.array): return self.embed(x)

class Decoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.decode = nn.Linear(dim, 256)
        self.stop = nn.Linear(dim, 1)

    def __call__(self, x: mx.array): return self.decode(x), mx.sigmoid(self.stop(x))

class Layer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        
        self.decay = mx.zeros((dim, ))
        self.states = mx.zeros((dim, ))

        self.decaytrace = mx.zeros((dim, ))
        self.embedtrace = mx.zeros((256, dim))
        
        self.norm = nn.LayerNorm(dim)
        self.weights = nn.Linear(dim, dim, bias = False)
        self.silu = nn.SiLU()

    def __call__(self, enc: mx.array, x: mx.array, dummy: mx.array):
        decay = mx.sigmoid(self.decay)
        state = (decay * self.states) + enc + dummy

        return x + self.silu(self.weights(self.norm(state))), state, decay

class Model(nn.Module):
    def __init__(self, dim: int, layers: int, temp: float, lr: float):
        super().__init__()
        self.dim = dim
        self.layercount = layers
        self.temp = temp

        self.encoder = Encoder(dim)
        self.decoder = Decoder(dim)

        self.layers = [Layer(dim) for _ in range(layers)]
        self.optimizer = opt.AdamW(learning_rate = lr)

    def sample(self, output: mx.array):
        probs = mx.softmax(output)
        entropy = -mx.sum(probs * mx.log(probs + 1e-8)) / mx.log(mx.array(256.0))

        temp = mx.maximum(0.1, self.temp * (1.0 - self.temp * entropy)).item()
        return mx.random.categorical(output / temp)

    def reset(self):
        for layer in self.layers:
            layer.decay = mx.zeros((self.dim, ))
            layer.states = mx.zeros((self.dim, ))

            layer.decaytrace = mx.zeros((self.dim, ))
            layer.embedtrace = mx.zeros((256, self.dim))

        mx.eval(*[layer.states for layer in self.layers])

    def step(self, c: mx.array, dummies: mx.array):
        enc = self.encoder(c)
        x = enc
            
        states, decays = [], []

        for i, layer in enumerate(self.layers):
            x, state, decay = layer(enc, x, dummies[i])

            states.append(state)
            decays.append(decay)

        return (x, states, decays), self.decoder(x)

    def __call__(self, currb: int, nextb: int | None, end: bool, notrace: bool = False, frozen: bool = False):
        c = mx.array(currb)

        if notrace:
            _, (output, stop) = self.step(c, [mx.zeros((self.dim, )) for _ in range(self.layercount)])
            return self.sample(output).item(), stop.item()

        if frozen:
            enc = self.encoder(c)
            x = enc

            for layer in self.layers:
                x, state, _ = layer(enc, x, mx.zeros((self.dim, )))
                layer.states = mx.stop_gradient(state)

            mx.eval(*[layer.states for layer in self.layers])
            output, stop = self.decoder(x)
            return self.sample(output).item(), stop.item()

        p = self.trainable_parameters()

        def fwd(params, dummies: list[mx.array]):
            self.update(params)
            (x, states, decays), (output, stop) = self.step(c, dummies)

            loss = mx.maximum(0.0, 1.0 - mx.sqrt(mx.var(x) + 1e-4)) # variance
            if nextb is not None:
                n = mx.array(nextb)
                tgt = mx.stop_gradient(self.encoder(n))

                loss = loss + mx.mean(mx.square(x - tgt)) # pred mse
                loss = loss - output[n] + mx.logsumexp(output) # ce

                loss = loss + mx.mean(mx.square(stop - mx.array([1.0 if end else 0.0]))) # stop mse

            return loss, (states, decays, output, stop) # loss = variance loss + pred mse loss + crossentropy loss + stop mse loss

        (_, (states, decays, output, stop)), (grads, dlds_s) = mx.value_and_grad(
            fwd, argnums = (0, 1)
        )(p, [mx.zeros((self.dim, )) for _ in range(self.layercount)])

        self.update(p)

        for i, layer in enumerate(self.layers):
            dlds = dlds_s[i]

            embedtrace = (layer.embedtrace * decays[i]) + (mx.arange(256) == c)[:, None].astype(mx.float32)
            grads["encoder"]["embed"]["weight"] += dlds * (layer.embedtrace * decays[i])
            
            decaytrace = (decays[i] * layer.decaytrace) + (decays[i] * (1.0 - decays[i]) * layer.states)
            grads["layers"][i]["decay"] = dlds * decaytrace

            layer.states = mx.stop_gradient(states[i])

            layer.decaytrace = mx.stop_gradient(decaytrace)
            layer.embedtrace = mx.stop_gradient(embedtrace)
            
            mx.eval(layer.states, layer.decaytrace, layer.embedtrace)

        self.optimizer.update(self, grads)
        mx.eval(self.parameters(), self.optimizer.state)

        return self.sample(output).item(), stop.item()

    def save(self, path: str):
        data = {}
        for k, v in util.tree_flatten(self.parameters()): data[f"m.{k}"] = v
        for k, v in util.tree_flatten(self.optimizer.state): data[f"o.{k}"] = v

        for i, layer in enumerate(self.layers):
            data[f"state.{i}"] = layer.states
            data[f"decaytrace.{i}"] = layer.decaytrace
            data[f"embedtrace.{i}"] = layer.embedtrace

        tmp = 'temporary-' + path
        mx.save_safetensors(tmp, data)
        os.replace(tmp, path)

    def load(self, path: str):
        if not os.path.exists(path): return

        data, model, opts = mx.load(path), {}, {}
        
        for k, v in data.items():
            if k.startswith("m."): model[k[2:]] = v
            elif k.startswith("o."): opts[k[2:]] = v
            elif k.startswith("state."): self.layers[int(k.split('.')[1])].states = v
            elif k.startswith("decaytrace."): self.layers[int(k.split('.')[1])].decaytrace = v
            elif k.startswith("embedtrace."): self.layers[int(k.split('.')[1])].embedtrace = v
            
        if model: self.update(util.tree_unflatten(list(model.items())))
        if opts: self.optimizer.state = util.tree_unflatten(list(opts.items()))

class Runtime:
    def __init__(self, path: str, threshold: float, **kwargs):
        self.model = Model(**kwargs)
        self.path = path
        self.threshold = threshold

        self.step = 0
        self.prevtime = None

    def save(self):
        self.step += 1
        if self.step % 500 == 0: self.model.save(self.path)

    def call(self, c: int, n: int | None, end: bool, readonly: bool = False, notrace: bool = False, frozen: bool = False):
        if frozen:
            return self.model(c, n, end, frozen = True)
        outputs = self.model(c, n, end, notrace)
        if not readonly: self.save()
        return outputs

    def write(self, b: int):
        sys.stdout.buffer.write(bytes([b]))
        sys.stdout.flush()

    def chat(self, readonly: bool = False, notrace: bool = False, frozen: bool = False):
        while True:
            text = input(f'\n[{self.now()} | {0 if self.prevtime is None else time.time() - self.prevtime:.4f}s]\nUser >> ')
            self.prevtime = time.time()

            data = (text + '\n').encode('utf-8')
            
            for i, (c, n) in enumerate(itertools.pairwise(data)):
                b, _ = self.call(c, n, i == len(data) - 2, readonly, notrace, frozen)

            print(f'\n[{self.now()}]\nModel >> ', end = '', flush = True)

            b = data[-1]
            while True:
                b, stop = self.call(b, None, False, readonly, notrace, frozen)
                self.write(b)
                if stop > self.threshold:
                    print()
                    break

    def dataset(self, pattern: str = 'wikipedia_clean/**/wiki_*'):
        files = glob.glob(pattern, recursive = True)

        if not files:
            raise FileNotFoundError(
                f"No training files matched {pattern!r}. "
                "Download e.g. simplewiki XML dump, clean it into wikipedia_clean/, "
                "or pass a different pattern."
            )

        while True:
            for file in files:
                with open(file, 'r', encoding = 'utf-8', errors = 'ignore') as f:
                    for line in f:
                        data = line.encode('utf-8')
                        if len(data) < 2:
                            continue
                        for i, (c, n) in enumerate(itertools.pairwise(data)):
                            b, _ = self.call(c, n, i == len(data) - 2)
                            self.write(b)

    def now(self):
        return datetime.now().strftime('%d/%m/%Y, %H:%M:%S')

    def __call__(self, mode: str, frozen: bool = False):
        modes = ['train', 'chat', 'chatreadonly', 'chatnotrace']

        if mode not in modes:
            print(f'\nInvalid mode {mode!r}. Choose from {modes}.')
            return
        mode = modes.index(mode)

        self.model.load(self.path)
        print()

        try:
            match mode:
                case 0: self.dataset()
                case 1: self.chat(frozen = frozen)
                case 2: self.chat(readonly = True, frozen = frozen)
                case 3: self.chat(readonly = True, notrace = True)

        finally:
            if mode in (0, 1) and not frozen: self.model.save(self.path)

def count_params(dim: int, layers: int) -> int:
    per_layer = dim * dim + 3 * dim
    return 256 * dim + layers * per_layer + 256 * dim + 256 + dim + 1

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test-Model-Thing: byte-level recurrent LM with MLX.')
    parser.add_argument('--mode', choices=['train', 'chat', 'chatreadonly', 'chatnotrace'], required=True)
    parser.add_argument('--frozen', action='store_true', help='Chat without any in-memory training (weights frozen, memory still advances).')
    parser.add_argument('--path', default='experimental-4.5m.safetensors')
    parser.add_argument('--threshold', type=float, default=0.35)
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--layers', type=int, default=16)
    parser.add_argument('--temp', type=float, default=0.75)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--pattern', default='wikipedia_clean/**/wiki_*',
                        help='Glob for training files (train mode).')
    args = parser.parse_args()

    print(f'params ~= {count_params(args.dim, args.layers):,}')
    runtime = Runtime(path=args.path, threshold=args.threshold, dim=args.dim,
                      layers=args.layers, temp=args.temp, lr=args.lr)
    if args.mode == 'train':
        runtime.model.load(runtime.path)
        print()
        try:
            runtime.dataset(args.pattern)
        finally:
            runtime.model.save(runtime.path)
    else:
        runtime(args.mode, frozen = args.frozen)
