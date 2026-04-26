# -*- coding: utf-8 -*-
# file: infer_example.py
# author: songyouwei <youwei0314@gmail.com>
# fixed: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.

import torch
import torch.nn.functional as F
import numpy as np

from data_utils import Tokenizer4Bert, pad_and_truncate
from models import LCF_BERT

from transformers import BertModel


class Inferer:
    """A simple inference example for LCF-BERT"""
    def __init__(self, opt):
        self.opt = opt
        self.tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model = opt.model_class(bert, opt).to(opt.device)
        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        self.model = self.model.to(opt.device)
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, text, aspect):
        aspect = aspect.lower().strip()
        text_left, _, text_right = [s.strip() for s in text.lower().partition(aspect)]

        aspect_indices = self.tokenizer.text_to_sequence(aspect)
        aspect_len = np.sum(aspect_indices != 0)

        text_bert_indices = self.tokenizer.text_to_sequence(
            "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
        text_len = np.sum(text_bert_indices != 0)
        concat_bert_indices = self.tokenizer.text_to_sequence(
            '[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
        concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
        concat_segments_indices = pad_and_truncate(concat_segments_indices, self.tokenizer.max_seq_len)

        aspect_bert_indices = self.tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

        data = {
            'concat_bert_indices': concat_bert_indices,
            'concat_segments_indices': concat_segments_indices,
            'text_bert_indices': text_bert_indices,
            'aspect_bert_indices': aspect_bert_indices,
            'aspect_indices': aspect_indices,
        }

        t_inputs = [torch.tensor([data[col]], device=self.opt.device) for col in self.opt.inputs_cols]
        t_outputs = self.model(t_inputs)
        t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()

        return t_probs


if __name__ == '__main__':
    model_classes = {
        'lcf_bert': LCF_BERT,
    }
    dataset_files = {
        'twitter': {
            'train': './datasets/acl-14-short-data/train.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            'train': './datasets/semeval14/Restaurants_Train.xml.seg',
            'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
        },
        'laptop': {
            'train': './datasets/semeval14/Laptops_Train.xml.seg',
            'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
        }
    }
    input_colses = {
        'lcf_bert': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices'],
    }

    class Option(object): pass
    opt = Option()
    opt.model_name = 'lcf_bert'
    opt.model_class = model_classes[opt.model_name]
    opt.dataset = 'restaurant'
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    # set your trained model here
    opt.state_dict_path = 'state_dict/lcf_bert_restaurant_val_acc_0.85'
    opt.max_seq_len = 85
    opt.bert_dim = 768
    opt.pretrained_bert_name = 'bert-base-uncased'
    opt.polarities_dim = 3
    opt.dropout = 0.1
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.local_context_focus = 'cdm'
    opt.SRD = 3

    inf = Inferer(opt)
    t_probs = inf.evaluate('the service is terrible', 'service')
    print(t_probs.argmax(axis=-1) - 1)
