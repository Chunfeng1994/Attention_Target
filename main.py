from handle_data import dataLoader,CreatVocab
import os
import torch
import numpy
import random
import argparse
from driver.Config import Configurable
from handle_data.CreatVocab import *
from handle_data.embed import *
from handle_data.train import train
from model.Vanilla import *
from model.Contextualized import *
from model.ContextualizedGates import *




if __name__ == '__main__':
    # random
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    random.seed(666)
    numpy.random.seed(666)

    # gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: ", torch.backends.cudnn.enabled)

    # parameters
    parse = argparse.ArgumentParser()
    parse.add_argument('--config_file', type=str, default='default.ini')
    parse.add_argument('--thread', type=int, default=1)
    parse.add_argument('--use_cuda', action='store_true', default=False)
    args, extra_args = parse.parse_known_args()

    config = Configurable(args.config_file, extra_args)
    torch.set_num_threads(1)
    config.use_cuda = False


    # load data

    train_data, sentence_length, src_dic, tgt_dic = dataLoader.read_sentence("./data/Z_data/all.conll.train", True)
    dev_data, sentence_length = dataLoader.read_sentence("./data/Z_data/all.conll.dev", False)

    test_data, sentence_length = dataLoader.read_sentence("./data/Z_data/all.conll.test", False)
    src_vocab, tgt_vocab = CreatVocab.create_vocabularies(train_data, 20000, src_dic, tgt_dic)


    embedding = None
    embedding = create_vocab_embs(config.embedding_file,src_vocab)

    # model

    model = Vanilla(config,src_vocab.size,PAD,tgt_vocab.size,embedding)

    if config.use_cuda:
        torch.backends.cudnn.enabled = True
        model = model.cuda()

    # train
    train(model, train_data, dev_data, test_data, src_vocab, tgt_vocab, config)

