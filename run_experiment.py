#!/usr/bin/env python

from __future__ import print_function
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *
import pandas as pd
import numpy as np
from scipy.stats import rankdata
from collections import OrderedDict, defaultdict
import re
import pickle
import sys
import json
import logging
import os
from l2rank.ndcg import dcg

class VocabCoding:
    
    def __init__(self, X):
        self.vocab = []
        for el in X:
            for sym in el:
                if sym not in self.vocab:
                    self.vocab.append(sym)

        self.to_index = {v:k for k,v in enumerate(self.vocab)}
        self.to_sym = {k:v for k,v in enumerate(self.vocab)}
        self.max_len = max(map(len, X))
        
    def transform(self, X):
        result = []
        for el in X:
            row = [self.to_index[x] for x in el]
            row += [-1] * (self.max_len - len(el))
            result.append(row)
        return np.array(result, dtype='int32')
        
    def to_one_hot(self, X):
        eye = np.eye(self.max_len + 1, dtype='bool')
        eye = np.delete(eye, -1, 1)
        result = eye[self.transform(X)]
        return result

def build_encoder(input_var, emb_type, emb_params,
                  enc_type, enc_params, repr_size):
    net = OrderedDict()
    
    if emb_type == 'learn':
        net['input'] = InputLayer((None, emb_params['max_len']),input_var, dtype='int32')
        net['emb']   = EmbeddingLayer(net['input'], emb_params['vocab_size'], emb_params['emb_size'])
    elif emb_type == 'one_hot':
        net['input'] = InputLayer((None, emb_params['max_len'], emb_params['vocab_size']), input_var, dtype='int32')
        net['emb']   = net['input']
    else:
        raise KeyError("Unknown emb_type %s" % emb_type)
    
    if enc_type == 'CNN':
        net['dimshuf'] = DimshuffleLayer(net['emb'], (0, 2, 1))
        net['conv_-1'] = net['dimshuf']
        for i in range(enc_params['n_layers']):
            net['conv_%d' % i] = Conv1DLayer(net['conv_%d' % (i - 1)],
                                             enc_params['n_filters'][i],
                                             enc_params['filter_size'][i])
        else:
            net['reshape'] = ReshapeLayer(net['conv_%d' % i], ([0], -1))
            net['dense_0'] = DenseLayer(net['reshape'], enc_params['hid_size'])
            net['enc'] = net['dense_0']
    elif enc_type == 'RNN':
        name_to_layer = {'RNN' : RecurrentLayer,
                         'LSTM': LSTMLayer,
                         'GRU' : GRULayer}
        net['input_mask'] = InputLayer((None, emb_params['max_len']), enc_params['mask'], dtype='int32')
        rnn_layer = name_to_layer[enc_params['rnn_type']]
        net['enc'] = rnn_layer(net['emb'], enc_params['hid_size'],
                               grad_clipping=enc_params['grad_clipping'],
                               only_return_final=True, mask_input=net['input_mask'])
    else:
        raise KeyError("Unknown enc_type %s" % enc_type)


    if repr_size:
        net['dense_1'] = DenseLayer(net['enc'], repr_size)
    else:
        net['dense_1'] = net['enc']

    net['output'] = net['dense_1']
    return net

def normalized_dot_product(protein, ligands, gram_matrix):
    # normalization, assuming protein.shape[0] == 1
    protein_norm  = protein / T.sqrt(T.sum(protein ** 2))
    ligands_norm = ligands / T.sqrt(T.sum(ligands ** 2, axis = 1, keepdims=True))
    # assume normalized
    if gram_matrix:
        result = T.dot(T.dot(protein_norm, gram_matrix), ligands_norm.transpose())[0]
    else:
        result = T.sum(protein_norm[0] * ligands_norm, axis=1)
    return result

def pairwise_hinge(pred, target, margin, kernel):
    n = pred.shape[0]
    delta_pred = pred.reshape((n, 1)) - pred.reshape((1, n))
    delta_target = target.reshape((n, 1)) - target.reshape((1, n))
    if kernel == 'sign':
        delta_target = T.sgn(delta_target)
    elif kernel == 'linear':
        pass
    else:
        raise KeyError("Unknown kernel %s" % kernel)
    losses = T.maximum(0, margin - delta_pred * delta_target) * T.invert(T.eye(n, dtype='bool'))
    norm_loss = T.sum(losses) / n / (n - 1)
    return norm_loss

def iterate_ligand_minibatches(df, prot_num, ligand_num, **kwargs):
    batchsize = kwargs.get("batchsize")
    include_single_ligands = kwargs.get("include_single_ligands")
    shuffle = kwargs.get("shuffle")

    num_ligands = [0] + [df.ix[df.index.levels[0][i]].shape[0] 
                         for i in range(df.index.levels[0].size - 1)]
    cum_sum_ligands = np.cumsum(num_ligands)
    
    n_prots = df.index.levels[0].size
    prot_indices = np.arange(n_prots)
    
    while True:
        
        if shuffle:
            np.random.shuffle(prot_indices)

        for prot_idx in prot_indices:
            batch_prot = df.index.levels[0][prot_idx]
            table = df.ix[batch_prot]
            n_ligands = table.shape[0]
            if not include_single_ligands and n_ligands == 1:
                continue
            ligand_indices = np.arange(n_ligands)

            if shuffle:
                np.random.shuffle(ligand_indices)
                
            if batchsize != None:
                for start_idx in range(0, n_ligands, batchsize):
                    excerpt = ligand_indices[start_idx:start_idx + batchsize]
                    yield (prot_num[prot_idx][None],
                           ligand_num[excerpt + cum_sum_ligands[prot_idx]],
                           table.ix[table.index[excerpt]]['Ki (nM)'].values)
            else:
                yield (prot_num[prot_idx][None],
                       ligand_num[ligand_indices + cum_sum_ligands[prot_idx]],
                       table.ix[table.index[ligand_indices]]['Ki (nM)'].values)

def masking_decorator(with_first, with_second):
    def decorator(foo):
        def wrapper(*args):
            new_args = list(args)
            if with_second:
                new_args.insert(2, np.where(args[1] != -1, 1,0).astype('int32'))
            if with_first:
                new_args.insert(1, np.where(args[0] != -1, 1,0).astype('int32'))
            return foo(*new_args)
        return wrapper
    return decorator

def compute_mse(pred, target):
    return np.mean((pred - target) ** 2)

def compute_hinge(delta_pred, delta_target, margin, kernel):
    n = delta_pred.shape[0]
    if kernel == 'sign':
        delta_target = np.sign(delta_target)
    elif kernel == 'linear':
        pass
    else:
        raise KeyError("Unknown kernel %s" % kernel)
    mask = np.invert(np.eye(n, dtype=bool))
    loss = np.sum(np.maximum(0, margin - delta_pred * delta_target) * mask)
    norm_loss = loss / n / (n - 1)
    return norm_loss

def compute_auc(delta_pred, delta_target):
    n = delta_pred.shape[0]
    n_right = ((delta_pred * delta_target) > 0.).sum()
    return  float(n_right) / n / (n - 1)

def compute_ndcg(pred, target, transformation, Ks):
    if transformation == 'percentile':
        relevances = (rankdata(target, method="min") - 1.) / target.size * 100
    else:
        raise KeyError("Unknown transformation %s" % transformation)

    relevances = relevances[pred.argsort()[::-1]]
    sorted_relevances = sorted(relevances, reverse=True)

    results = []
    for rank in Ks:
        best_dcg = dcg(sorted_relevances, rank)
        if best_dcg == 0:
            results.append(0.)
            continue

        results.append(dcg(relevances, rank) / best_dcg)
    return results

def compute_scores(pred, target, metrics):
    names = [metric["name"] for metric in metrics]
    if "hinge" in names or "auc" in metrics:
        delta_pred = pred[:, None] - pred[None, :]
        delta_target = target[:, None] - target[None, :]

    results = []
    for metric in metrics:
        if metric["name"] == "MSE":
            results.append(compute_mse(pred, target))
        elif metric["name"] == "hinge":
            results.append(compute_hinge(delta_pred, delta_target,
                                         metric['margin'], metric['kernel']))
        elif metric["name"] == "auc":
            results.append(compute_auc(delta_pred, delta_target))
        elif metric["name"] == "ndcg":
            results.extend(compute_ndcg(pred, target, 
                                        metric['transformation'], 
                                        metric['Ks']))
        else:
            raise KeyError("Unknown metric %s" % metric["name"])
    return np.array(results)

def main(argc, argv):
    path = os.path.dirname(argv[2])
    exp_num = os.path.basename(argv[2]).split('_')[0]
    logging.basicConfig(filename=path+'/'+exp_num+'.log',
                        format='%(asctime)s %(message)s',
                        level=logging.DEBUG)
    logger = logging.getLogger("main")
    assert argc == 3

    logger.info("Reading db...")
    with open(argv[1], 'r') as f:
        db = pickle.load(f)

    logger.info("Reading params...")
    with open(argv[2], 'r') as f:
        params = json.load(f)

####################################################################
    X = {}
    X['protein'] = np.array(db.index.levels[0])
    X['ligand']  = np.array(db.index.get_level_values(1))

    logger.info("Calculating features...")
    X_num = {}
    coding = {}
    names = ['protein', 'ligand']
    for name in names:
        coding[name] = VocabCoding(X[name])
        if   params[name]['emb_type'] == "learn":
            X_num[name] = coding[name].transform(X[name])
        elif params[name]['emb_type'] == "one_hot":
            X_num[name] = coding[name].to_one_hot(X[name])
            X_num[name].transpose((0, 2, 1))
        else:
            raise KeyError("Unknown emb_type %s" % emb_type)
    del X
####################################################################
    y = db['Ki (nM)'].values

    transformation = params["affinity"]["transformation"]
    if transformation == "log+min_max":
        y = -np.log(y)
        y = (y - y.min()) / (y.max() - y.min())
    else:
        raise KeyError("Unknown transformation %s" % transformation)

    db['Ki (nM)'] = y.astype('float32')
####################################################################
    logger.info("Building networks...")
    input_var = {}
    nets = {}
    for name in names:
        emb_type = params[name]['emb_type']
        if   emb_type == "learn":
            input_var[name] = T.imatrix(name=name)
        elif emb_type == "one_hot":
            input_var[name] = T.itensor3(name=name)
        else:
            raise KeyError("Unknown emb_type %s" % emb_type)

        emb_params = params[name]['emb_params']
        emb_params.update({'vocab_size': len(coding[name].vocab),
                           'max_len'   : coding[name].max_len})

        enc_type = params[name]['enc_type']
        enc_params = params[name]['enc_params']
        if enc_type == 'RNN':
            enc_params['mask'] = T.imatrix(name=name + '_mask')

        nets[name] = build_encoder(input_var[name],
                                   emb_type, emb_params,
                                   enc_type, enc_params,
                                   params[name]['repr_size'])
####################################################################
    logger.info("Preparing variables...")
    reprs = {}
    for name in names:
        reprs[name] = get_output(nets[name]['output'], deterministic=True)
    target_y = T.vector('Ki value', dtype='float32')
    l_rate_theano = T.scalar('learning rate')

    gram_matrix = None
    if params['dot_product']['learnable']:
        w_init = lasagne.init.GlorotUniform()
        n, m = (nets['protein']['output'].output_shape[1],
                nets['ligand' ]['output'].output_shape[1])
        gram_matrix = theano.shared(lasagne.utils.floatX(w_init((n, m))))
        if n == m:
            # force to be positive symmetric
            gram_matrix = T.dot(gram_matrix, gram_matrix.T)

    cosine_pred = normalized_dot_product(reprs['protein'], 
                                         reprs['ligand'],
                                         gram_matrix)
    normalized_cosine_pred = (cosine_pred + 1) / 2

    if (params['loss']['type'] == 'MSE'):
        loss = T.mean((normalized_cosine_pred - target_y) ** 2)
    elif (params['loss']['type'] == 'hinge'):
        loss = pairwise_hinge(normalized_cosine_pred, target_y,
                              params['loss']['margin'],
                              params['loss']['kernel'])
    else:
        raise KeyError("Unknown loss %s" % params['loss']['type'])

    weights = []
    for name in names:
        weights += get_all_params(nets[name]['output'], trainable=True)

    updates = lasagne.updates.adam(loss, weights, learning_rate=l_rate_theano)

####################################################################
    logger.info("Compiling functions...")

    t_input_args = []
    for name in names:
        t_input_args.append(input_var[name])
        if params[name]["enc_type"] == 'RNN':
            t_input_args.append(params[name]["enc_params"]['mask'])
    t_input_args += [target_y, l_rate_theano]
    train_fun = theano.function(t_input_args, [loss, normalized_cosine_pred], updates=updates)

    train_fun_wrap = masking_decorator(params['protein']["enc_type"] == 'RNN',
                                       params['ligand']["enc_type"] == 'RNN')(train_fun)
    test_fun = theano.function(t_input_args[:-1], [loss, normalized_cosine_pred])
    test_fun_wrap = masking_decorator(params['protein']["enc_type"] == 'RNN',
                                      params['ligand']["enc_type"] == 'RNN')(test_fun)


####################################################################
    logger.info("Train test dividing...")

    n_prots = db.index.levels[0].size
    n_ligands = [db.ix[db.index.levels[0][i]].shape[0] for i in range(db.index.levels[0].size)]
    cum_sum_ligands = np.cumsum([0] + n_ligands[:-1])

    np.random.seed(params['train_test_split']['seed'])
    data = defaultdict(list)
    ind = {}
    ind['train'], ind['test'] = np.split(np.random.permutation(n_prots),
                                         [int(n_prots * params['train_test_split']['train_frac'])])
    for name in ['train', 'test']:
        data[name] += [db.ix[db.index.levels[0][ind[name]]]]
        data[name] += [X_num['protein'][ind[name]]]
        inner_ind = sum([list(np.arange(n_ligands[i]) + cum_sum_ligands[i]) for i in ind[name] ], [])
        data[name] += [X_num['ligand'][inner_ind]]
####################################################################
    logger.info("Start learning...")
    n_epochs = params["learning_params"]["n_epochs"]
    n_batches = params["learning_params"]["n_batches"]
    l_rate = params["learning_params"]["l_rate"]
    metrics = params["metrics"]
    n_metrics = sum([1 if metric["name"] != "ndcg" else len(metric["Ks"]) for metric in metrics])

    train_loss_list = []
    test_loss_list = []

    iter_kwargs = {'shuffle'                : True,
                   'batchsize'              : params["learning_params"]["batch_size"],
                   'include_single_ligands' : params["loss"] == 'MSE'}

    train_iter = iterate_ligand_minibatches(*data["train"], **iter_kwargs)
    test_iter = iterate_ligand_minibatches(*data["test"], **iter_kwargs)


    scores = {}
    scores["train"] = np.zeros((n_epochs, n_metrics))
    scores["test"] = np.zeros((n_epochs, n_metrics))
    epoch_scores = np.zeros(n_metrics)

    for i in range(n_epochs):
        logger.info("Starting epoch %d" % i)
        train_loss = 0
        num_batches = 0
        epoch_scores.fill(0)
        for prot_batch, ligand_batch, target_batch in train_iter:
            train_loss_batch, pred_batch = train_fun_wrap(prot_batch,
                                                          ligand_batch, 
                                                          target_batch,
                                                          l_rate)
            epoch_scores += compute_scores(pred_batch, target_batch, metrics)
            train_loss += train_loss_batch
            num_batches += 1
            if num_batches == n_batches:
                break
        scores["train"][i] = epoch_scores / num_batches
        train_loss_list.append(train_loss / num_batches)
        logger.info("Train loss %f" % train_loss_list[-1])

        test_loss = 0
        num_batches = 0
        epoch_scores.fill(0)
        for prot_batch, ligand_batch, target_batch in test_iter:
            test_loss_batch, pred_batch = test_fun_wrap(prot_batch,
                                                        ligand_batch, 
                                                        target_batch)
            epoch_scores += compute_scores(pred_batch, target_batch, metrics)
            test_loss += test_loss_batch
            num_batches += 1
            if num_batches == n_batches:
                break
        scores["test"][i] = epoch_scores / num_batches
        test_loss_list.append(test_loss / num_batches)
        logger.info("Test loss %f" % test_loss_list[-1])

        if (i % 10 == 0):
            with open(path+'/'+exp_num+'_results.pkl', 'w') as f:
                pickle.dump(scores, f)
####################################################################
    logger.info("Dump results...")
    with open(path+'/'+exp_num+'_results.pkl', 'w') as f:
        pickle.dump(scores, f)


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)