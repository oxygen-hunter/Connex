# -*- coding:utf-8 -*-

import os
import logging
import typing as T
import json
from easydict import EasyDict as ed
from collections import defaultdict

from src import config as C
from src import tools as TL
from src.extractor import ExtractorRule
from src.matcher import MatcherMultichain


def _generate_token_info_from_key_info(key_info:T.Dict)->T.Dict:
    token_info = {}
    for k in ('chain', 'token_addr', 'token_name', 'token_symbol'): 
         token_info[k] = key_info.get(k, '')
    return token_info


def _generate_tx_hash_to_key_info_dict()->T.Dict:
    # help analyze
    chains_key_info = dict()
    tx_hash_to_key_info = dict() # map tx_hash to key_info

    with open(C.paths().key_info, 'r') as f:
        chains_key_info = json.load(f)
    
    for chain in chains_key_info:
        for key_info in chains_key_info[chain]:
            tx_hash = key_info['hash']
            tx_hash_to_key_info[tx_hash] = key_info

    return tx_hash_to_key_info


def generate_gt_from_tx():
    logging.info('Generating ground truth from tx')
    print('---generate gt from tx begin---')
    # generate gt only once
    if os.path.exists(C.paths().gt_from_tx):
        print('---gt from tx exist---')
        print('---generate gt from tx end---')
        return

    chains_to_pair = C.chains_to_pair
    gt: T.Dict[str, str] = dict()
    extractor = ExtractorRule()

    tx_dict:T.Dict[str, T.Dict] = dict()
    for chain in chains_to_pair:
        txs = TL.load_all_data_by_chain(chain)
        for tx in txs:
            tx_hash = tx['hash'].lower()
            tx_dict[tx_hash] = tx
            tx_dict[tx_hash]['matched'] = False
            tx_dict[tx_hash]['chain'] = chain

    # use tx.input data to find src tx hash
    for tx_hash in tx_dict:
        if tx_dict[tx_hash]['matched'] == True: continue
        try:
            src_tx_hash = extractor.decode_src_tx_hash_from_dst_tx(ed(tx_dict[tx_hash]), tx_dict[tx_hash]['chain'])
        except Exception as e:
            logging.error("extracting tx hash fail:" + tx_hash)
            logging.exception(e)
            continue
        if len(src_tx_hash) and src_tx_hash in tx_dict: # 查到的src tx hash必须出现在数据集中
            gt[src_tx_hash] = tx_hash
            tx_dict[tx_hash]['matched'] = True
            tx_dict[src_tx_hash]['matched'] = True # 少一次查gt
        
    print('load {tx} tx, generate {gt} gt'.format(
        tx=len(tx_dict),
        gt=len(gt)
    ))

    with open(C.paths().gt_from_tx, 'w') as f:
        json.dump(gt, f, indent=2)

    print('---generate gt from tx end---')



def generate_gt_from_api():
    logging.info('Generating ground truth from api')
    print('---generate gt from api begin---')
    # generate gt only once
    if os.path.exists(C.paths().gt_from_api):
        print('---gt from api exist---')
        print('---generate gt from api end---')
        return

    chains_to_pair = C.chains_to_pair
    gt: T.Dict[str, str] = dict()
    mm = MatcherMultichain()

    tx_dict:T.Dict[str, T.Dict] = dict()
    for chain in chains_to_pair:
        txs = TL.load_all_data_by_chain(chain)
        for tx in txs:
            tx_hash = tx['hash'].lower()
            tx_dict[tx_hash] = tx
            tx_dict[tx_hash]['matched'] = False
            tx_dict[tx_hash]['chain'] = chain

    # use mm api to find dst tx hash
    for tx_hash in tx_dict:
        if tx_dict[tx_hash]['matched'] == True: continue
        try:
            dst_tx_hash = mm.go(tx_hash)
            
        except Exception as e:
            logging.error("fetching tx hash fail:" + tx_hash)
            logging.exception(e)
            continue
        if dst_tx_hash in tx_dict: # dst tx必须出现在数据集中
            gt[tx_hash] = dst_tx_hash
            tx_dict[tx_hash]['matched'] = True
            tx_dict[dst_tx_hash]['matched'] = True # 少一次查gt
        
    print('load {tx} tx, generate {gt} gt'.format(
        tx=len(tx_dict),
        gt=len(gt)
    ))

    with open(C.paths().gt_from_api, 'w') as f:
        json.dump(gt, f, indent=2)

    print('---generate gt from api end---')


def generate_gt_classified_by_chain():
    print('---classify gt by chain begin---')
    gt = dict()
    with open(C.paths().gt, 'r') as f:
        gt = json.load(f)
    
    gt_classified_by_chain = dict()
    tx_from_chain:T.Dict[str, str] = dict() # map tx_hash to chain
    chains_to_pair = C.chains_to_pair

    for chain in chains_to_pair:
        txs = TL.load_all_data_by_chain(chain)
        for tx in txs:
            tx_hash = tx['hash']
            tx_from_chain[tx_hash] = chain
    
    for src_tx in gt:
        dst_tx = gt[src_tx]
        src_chain = tx_from_chain[src_tx]
        dst_chain = tx_from_chain[dst_tx]
        if src_chain not in gt_classified_by_chain:
            gt_classified_by_chain[src_chain] = {}
        if dst_chain not in gt_classified_by_chain[src_chain]:
            gt_classified_by_chain[src_chain][dst_chain] = []
        gt_classified_by_chain[src_chain][dst_chain].append([src_tx, dst_tx])
    
    with open(C.paths().gt_classified_by_chain, 'w') as f:
        json.dump(gt_classified_by_chain, f, indent=2)

    print('---classify gt by chain end---')


def generate_gt():
    print('---generate gt begin---')
    # generate gt from tx and api
    generate_gt_from_tx()
    generate_gt_from_api()

    # merge gt_from_tx and gt_from_api
    gt_from_tx = dict()
    gt_from_api = dict()
    with open(C.paths().gt_from_tx, 'r') as f:
        gt_from_tx = json.load(f)
    with open(C.paths().gt_from_api, 'r') as f:
        gt_from_api = json.load(f)
    
    gt = {**gt_from_tx, **gt_from_api}
    with open(C.paths().gt, 'w') as f:
        json.dump(gt, f, indent=2)

    print('{tx} gt from tx, {api} gt from api, total {total} gt'.format(
        tx=len(gt_from_tx),
        api=len(gt_from_api),
        total=len(gt)
    ))
    
    print('---generate gt end---')

    # classify gt by chain
    generate_gt_classified_by_chain()


def generate_paired_tokens_from_gt():
    logging.info('Generating paired tokens from gt')
    print('---generate paired tokens from gt begin---')

    gt = dict()
    with open(C.paths().gt, 'r') as f:
        gt = json.load(f)
    tx_hash_to_key_info = _generate_tx_hash_to_key_info_dict()

    paired_tokens = defaultdict(lambda : defaultdict(dict))
    for src_tx in gt:
        dst_tx = gt[src_tx]
        if src_tx not in tx_hash_to_key_info or dst_tx not in tx_hash_to_key_info:
            continue
        src_key_info = tx_hash_to_key_info[src_tx]
        dst_key_info = tx_hash_to_key_info[dst_tx]
        src_token_info = _generate_token_info_from_key_info(src_key_info)
        dst_token_info = _generate_token_info_from_key_info(dst_key_info)
        one_pair = {
            "token1": src_token_info,
            "token2": dst_token_info
        }
        paired_tokens.append(one_pair)
    
    with open(C.paths().paired_tokens_from_gt, 'w') as f:
        json.dump(paired_tokens, f, indent=2)
                
    print('---generate paired tokens from gt end---')