# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import defaultdict
from collections import deque

import torch

# copy from mask rcnn benchmark

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger(object):
    """为了规范的记录和打印函数运行过程中的变量。

    Args:
        object ([type]): [description]
    """

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        # input a unpacked dictionary or key=value pair. e.g. logger.update(a=1, **a_dict)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(
                    name, meter.median, meter.global_avg)
            )
        return self.delimiter.join(loss_str)


if __name__ == "__main__":
    logger = MetricLogger()
    loss_dict = dict(zip('abcd', range(4)))
    print(loss_dict)
    logger.update(**loss_dict)
    for i in range(1000):
        if i % 4 == 0:
            a = i
            logger.update(a=i)
        if i % 4 == 1:
            b = i
            logger.update(b=i)
        if i % 4 == 2:
            c = i
            logger.update(c=i)
        else:
            d = i
            logger.update(d=i)
        if i%200 == 0:
            print(logger)