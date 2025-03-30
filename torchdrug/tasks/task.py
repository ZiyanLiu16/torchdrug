from collections.abc import Mapping, Sequence

import pdb

import torch
from torch import nn


class Task(nn.Module):

    _option_members = set()

    def _standarize_option(self, x, name):
        if x is None:
            x = {}
        elif isinstance(x, str):
            x = {x: 1}
        elif isinstance(x, Sequence):
            x = dict.fromkeys(x, 1)
        elif not isinstance(x, Mapping):
            raise ValueError("Invalid value `%s` for option member `%s`" % (x, name))
        return x

    def __setattr__(self, key, value):
        if key in self._option_members:
            value = self._standarize_option(value, key)
        super(Task, self).__setattr__(key, value)

    def preprocess(self, train_set, valid_set, test_set):
        pass

    def predict_and_target(self, batch, all_loss=None, metric=None):
        return self.predict(batch, all_loss, metric), self.target(batch)

    def predict(self, batch, all_loss=None, metric=None):
        raise NotImplementedError

    def target(self, batch):
        raise NotImplementedError

    def evaluate(self, pred, target):
        raise NotImplementedError

    def mps(self):
        device = torch.device("mps")
        self.recursive_to(obj=self, device=device)
        return self

    def recursive_to(self, obj, device, seen=None):
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return obj
        seen.add(obj_id)

        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: self.recursive_to(v, device, seen) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.recursive_to(v, device, seen) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self.recursive_to(v, device, seen) for v in obj)
        elif hasattr(obj, '__dict__'):
            for key, value in obj.__dict__.items():
                setattr(obj, key, self.recursive_to(value, device, seen))
            return obj
        elif hasattr(obj, 'to') and callable(getattr(obj, 'to')):
            try:
                return obj.to(device)
            except Exception:
                return obj
        return obj