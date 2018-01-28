import torch
import numpy as np
from toolz import reduce
import codings


def test_qsgd_and_terngrad():
    n = 50
    x = torch.rand(n)
    code = codings.QSGD()
    for scheme in ['terngrad', 'qsgd']:
        code.scheme = scheme
        repeats = int(10e3)
        codes = [code.encode(x, scheme=scheme) for _ in range(repeats)]
        code.codes = codes

        approxs = [code.decode(x) for x in codes]

        data = map(lambda arg: {'y': arg[1], 'norm(y)**2': torch.norm(arg[1])**2,
                                'len(signs)': len(arg[0]['signs'])},
                   zip(codes, approxs))
        sums = reduce(lambda x, y: {k: x[k] + y[k] for k in x}, data)
        avg = {k: v / len(codes) for k, v in sums.items()}
        assert avg['norm(y)**2'] <= np.sqrt(n) * torch.norm(x)**2
        if scheme == 'qsgd':
            assert avg['len(signs)'] <= np.sqrt(n)
        rel_error = torch.norm(avg['y'] - x) / torch.norm(x)
        print(scheme, rel_error)
        assert rel_error < 0.25

if __name__ == "__main__":
    test_qsgd_and_terngrad()
