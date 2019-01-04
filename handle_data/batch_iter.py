# -*- coding: utf-8 -*-
import re
import numpy as np
import torch
from torch.autograd import Variable
import random

torch.manual_seed(233)
random.seed(233)


def create_batch_iter(data, batch_size, shuffle=True):
    data_size = len(data)
    if shuffle:
        np.random.shuffle(data)

    # 排序
    src_ids = sorted(range(data_size), key=lambda src_id: len(data[src_id][0]), reverse=True)
    data = [data[src_id] for src_id in src_ids]

    batched_data = []
    instances = []
    for instance in data:
        instances.append(instance)
        if len(instances) == batch_size:
            batched_data.append(instances)
            instances = []

    if len(instances) > 0:
        batched_data.append(instances)

    for batch in batched_data:
        yield batch


def  pair_data_variable(batch, vocab_srcs, vocab_tgts, config):
    batch_size = len(batch)

    src_lengths = [len(batch[i][0]) for i in range(batch_size)]
    # 因为之前排序了，是递减的顺序
    max_src_length = int(src_lengths[0])

    src_words = Variable(torch.LongTensor(max_src_length, batch_size).zero_(), requires_grad=False)
    tgt_words = Variable(torch.LongTensor(batch_size).zero_(), requires_grad=False)

    start = []
    end = []

    for idx, instance in enumerate(batch):
        sentence = vocab_srcs.word2id(instance[0])
        for idj, value in enumerate(sentence):
            src_words.data[idj][idx] = value
        tgt_words[idx] = vocab_tgts.word2id(instance[3])
        start.append(instance[1])
        end.append(instance[2])

    if config.use_cuda:
        src_words = src_words.cuda()
        tgt_words = tgt_words.cuda()

    return src_words, tgt_words, src_lengths,start,end
