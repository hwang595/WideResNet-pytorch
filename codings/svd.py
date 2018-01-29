import torch
import numpy as np
import warnings
import sys
import time

from .coding import Coding

def _resize_to_2d(x):
    """
    x.shape > 2
    If x.shape = (a, b, *c), assumed that each one of (a, b) pairs has relevant information in c.
    """
    size = x.size()
    if all([s == 1 for s in size[2:]]):
        return x.view(size[0], size[1])
    # each of (a, b) has related features
    x = x.view(size[0], size[1], -1)
    # stack those related features into a tall matrix
    return x.view(size[0]*size[1], -1)


def _sample_svd(s, rank=0):
    if s[0] < 1e-6:
        return [0], torch.Tensor([1.0])
    probs = s / s[0] if rank == 0 else rank * s / s.sum()
    sampled_idx = []
    sample_probs = []
    for i, p in enumerate(probs):
        if np.random.rand() < p:
            sampled_idx += [i]
            sample_probs += [p]
    rank_hat = len(sampled_idx)
    if rank_hat == 0:  # or (rank != 0 and np.abs(rank_hat - rank) >= 3):
        return _sample_svd(s, rank=rank)
    return sampled_idx, torch.Tensor(sample_probs)


class SVD(Coding):
    def encode(self, grad, compress=True, svd_rank=0, random_sample=True):
        # move to CPU; torch's SVD is 5x faster on CPU
        if not compress:
            size = list(grad.size())
            return {'grad': grad, 'encode': False}#, {}

        orig_size = list(grad.size())
        ndims = len(grad.size())
        reshaped_flag = False
        if ndims > 2:
            grad = _resize_to_2d(grad)
            size = list(grad.size())
            ndims = len(size)
            reshaped_flag = True

        if ndims == 2:
            u, s, v = torch.svd(grad, some=True)
            if random_sample:
                i, probs = _sample_svd(s, rank=svd_rank)
                i = torch.LongTensor(i)
                #  if s.is_cuda:
                    #  i = i.cuda()
                    #  probs = probs.cuda()
                u = u[:, i]
                s = s[i] / probs
                v = v[:, i]
            elif svd_rank >= 0:
                u = u[:, :svd_rank]
                s = s[:svd_rank]
                v = v[:, :svd_rank]

            return {'u': u, 's': s, 'v': v, 'orig_size': orig_size,
                   'reshaped': reshaped_flag, 'encode': True,
                   'rank': svd_rank}
        return {'grad': grad, 'encode': False}


    def decode(self, encode_output, cuda=False):
        if isinstance(encode_output, tuple) and len(encode_output) == 1:
            encode_output = encode_output[0]
        encode = encode_output.get('encode', False)
        if not encode:
            grad = encode_output['grad']
            grad = torch.Tensor(grad)
            if cuda:
                grad = grad.cuda()
            return grad

        u, s, v = (encode_output[key] for key in ['u', 's', 'v'])
        if isinstance(u, (torch.Tensor, torch.cuda.FloatTensor)):
            u, s, v = (x.cpu().numpy() for x in [u, s, v])
        grad_approx = u @ np.diag(s) @ v.T
        grad_approx = torch.Tensor(grad_approx)
        if encode_output.get('reshaped', False):
            grad_approx = grad_approx.view(encode_output['orig_size'])
        if cuda:
            grad_approx = grad_approx.cuda()
        return grad_approx
