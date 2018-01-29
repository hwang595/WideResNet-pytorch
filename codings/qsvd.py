import codings
from codings import coding
import numpy as np
import copy

import torch
from toolz import reduce, partial

class QSVD(coding.Coding):
    def __init__(self, scheme='qsgd', *args, **kwargs):
        self.scheme = scheme
        self.qsgd = codings.QSGD(scheme=scheme)
        self.svd = codings.SVD()
        super().__init__(self, *args, **kwargs)

    def encode(self, grad, **kwargs):
        svd_code = self.svd.encode(grad)
        v = self.qsgd.encode(svd_code['v'])
        u = self.qsgd.encode(svd_code['u'])
        svd_code.update({'u': u, 'v': v})
        assert isinstance(svd_code, dict)
        return svd_code

    def decode(self, code, codes=[], **kwargs):
        u = self.qsgd.decode(code['u'], codes=[code['u'] for code in codes])
        v = self.qsgd.decode(code['v'], codes=[code['v'] for code in codes])
        code.update({'u': u, 'v': v})
        return self.svd.decode(code)

if __name__ == "__main__":
    n = 50
    x = torch.rand(n, n//10)
    
    for scheme in ['terngrad', 'qsgd']:
        print(scheme)
        kwargs = {'scheme':scheme}
        qsvd = QSVD(scheme)
        repeats = int(1000)
        codes = [qsvd.encode(x, **kwargs) for _ in range(repeats)]
        decode = partial(qsvd.decode, codes=copy.deepcopy(codes))
        approxs = [decode(code) for code in codes]

        data = map(lambda arg: {'y': arg[1], 'norm(y)**2': torch.norm(arg[1])**2},
                   zip(codes, approxs))
        sums = reduce(lambda x, y: {k: x[k] + y[k] for k in x}, data)
        avg = {k: v / len(codes) for k, v in sums.items()}
       
        rel_error = torch.norm(avg['y'] - x) / torch.norm(x)
        print(rel_error)
        assert rel_error < 0.25
