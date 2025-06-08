
import openpyxl
import logging
from collections import defaultdict as dd
from easydict import EasyDict as ed
import json
import os
import typing as T


from src import config as C
from src import tools as TL

def check_pairing_res(pairing_res_path:str='', gt_path:str='', output_paths:dict=None, chains_to_pair=[]):
    logging.info('Checking pairing results')
    print('---check begin---')
    # load me and gt
    pairing_res = dict()
    
    pairing_res_path = pairing_res_path or C.paths().pair_res
    gt_path = gt_path or C.paths().gt
    output_paths = output_paths or dict()
    for k in ('TP', 'FP', 'FN', 'statis', 'statis_excel'):
        if not output_paths.get(k, ''):
            output_paths[k] = C.paths().get(k)
    with open(pairing_res_path, 'r') as f:
        pairing_res = json.load(f)
    with open(gt_path, 'r') as f:
        GT = json.load(f)
    chains_to_pair = chains_to_pair or C.chains_to_pair
    data_all_trx = dd(set) # all occuring transactions from ourcllected data scope. Any trx out of this scope is not considered
    for chain in chains_to_pair:
        trxs = TL.load_all_data_by_chain(chain)
        for trx in trxs: 
            trx_hash = trx['hash'].lower()
            data_all_trx[chain].add(trx_hash)

    statis = {} # statistics
    TP = dict() # true positive
    FP = dict() # false positive, store me vs truth
    FN = dict() # false negative, store pairs in gt but not in me
    another_tp = dd(lambda: dd(lambda: 0))
    ours_all_in_one_lst = dd(list)
    for src_chain in pairing_res:
        for dst_chain in pairing_res[src_chain]:
            for one_pair in pairing_res[src_chain][dst_chain]:
                ours_all_in_one_lst[src_chain].append(one_pair)

    total_valid_gt_cnt = 0
    # compare me vs gt, get TP and FP
    for src_chain in pairing_res:
        statis[src_chain] = {}
        TP[src_chain] = {}
        FP[src_chain] = {}
        FN[src_chain] = {}
        if src_chain in GT:
            gt = GT[src_chain]['gt']
        else:
            gt = {}
        for dst_chain in pairing_res[src_chain]:
            statis[src_chain][dst_chain] = {'TP':0, 'FP':0, 'FN':0}
            TP[src_chain][dst_chain] = []
            FP[src_chain][dst_chain] = []
            FN[src_chain][dst_chain] = []
            for one_pair in pairing_res[src_chain][dst_chain]:
                ours_src_tx = one_pair[0].lower()
                ours_dst_tx = one_pair[1].lower()
                truth = gt.get(ours_src_tx, [])
                if not len(truth):
                    continue
                gt_dst_chain, gt_dst_tx = truth[0], truth[1].lower()
                if gt_dst_chain == dst_chain and gt_dst_tx == ours_dst_tx:
                    statis[src_chain][dst_chain]['TP'] += 1
                    TP[src_chain][dst_chain].append([ours_src_tx, ours_dst_tx])
                elif ours_dst_tx != '':
                    statis[src_chain][dst_chain]['FP'] += 1
                    vs = dict() # record gt for debug
                    vs['me'] = [ours_src_tx, ours_dst_tx, src_chain, dst_chain]
                    vs['gt'] = [ours_src_tx, gt_dst_tx, src_chain, gt_dst_chain]
                    FP[src_chain][dst_chain].append(vs)
        gt_not_in_data = 0
        for gt_src_tx, (dst_chain, gt_dst_tx) in gt.items():
            if dst_chain not in chains_to_pair: 
                continue
            gt_src_tx, gt_dst_tx = gt_src_tx.lower(), gt_dst_tx.lower()
            if not (gt_src_tx in data_all_trx[src_chain] and gt_dst_tx in data_all_trx[dst_chain]): 
                gt_not_in_data += 1
                continue
            total_valid_gt_cnt += 1
            if [gt_src_tx, gt_dst_tx] not in ours_all_in_one_lst[src_chain]:
                statis[src_chain][dst_chain]['FN'] += 1
                FN[src_chain][dst_chain].append([gt_src_tx, gt_dst_tx])
            else:
                another_tp[src_chain][dst_chain] += 1
        print(f"src chain: {src_chain}, gt not in data cnt: {gt_not_in_data}({round(TL.save_div(gt_not_in_data, len(gt)), 3)})")

    print(TL.defaultdict_to_dict(another_tp))
    # summary
    TP_sum = 0
    FP_sum = 0
    FN_sum = 0
    for src_chain in statis:
        for dst_chain in statis[src_chain]:
            TP_val = statis[src_chain][dst_chain]['TP']
            FP_val = statis[src_chain][dst_chain]['FP']
            FN_val = statis[src_chain][dst_chain]['FN']
            if not (TP_val+FP_val):
                statis[src_chain][dst_chain]['precision'] = precision =  0
            else:
                statis[src_chain][dst_chain]['precision'] = precision = TL.save_div(TP_val, (TP_val+FP_val))
            statis[src_chain][dst_chain]['recall'] = recall = TL.save_div(TP_val, (TP_val+FN_val))
            statis[src_chain][dst_chain]['F1'] = TL.save_div(2 * precision * recall , precision + recall)
            TP_sum += TP_val
            FP_sum += FP_val
            FN_sum += FN_val
    statis['summary'] = {}
    statis['summary']['TP'] = TP_sum
    statis['summary']['FP'] = FP_sum
    statis['summary']['FN'] = FN_sum
    statis['summary']['precision'] = precision = TL.save_div(TP_sum, (TP_sum+FP_sum))
    statis['summary']['recall'] = recall = TL.save_div(TP_sum, (TP_sum+FN_sum))

    statis['summary']['F1'] = TL.save_div(2 * precision * recall , precision + recall)

    print(f"total_valid_gt_cnt: {total_valid_gt_cnt}, TP+FN+FP:{TP_sum+FP_sum+FN_sum}")

    with open(output_paths['statis'], 'w') as f:
        json.dump(statis, f, indent=2)

    _write_statis_to_excel(statis, output_paths['statis_excel'])
    
    with open(output_paths['TP'], 'w') as f:
        json.dump(TP, f, indent=2)

    with open(output_paths['FP'], 'w') as f:
        json.dump(FP, f, indent=2)
    
    with open(output_paths['FN'], 'w') as f:
        json.dump(FN, f, indent=2)

    print('---check end---')
    return statis['summary']['precision'], statis['summary']['recall'], statis['summary']['F1']


def _write_statis_to_excel(statis:T.Dict, path:str):
    '''
    write statis to excel
    '''
    wb = openpyxl.Workbook()
    ws = wb.active

    title = [
        'src_chain -> dst_chain',
        'TP',
        'FP',
        'FN',
        'precision',
        'recall',
        'F1'
    ]
    ws.append(title)

    for src_chain in statis:
        if src_chain != 'summary':
            for dst_chain in statis[src_chain]:
                data = ed(statis[src_chain][dst_chain])
                data.precision = '%.2f%%' % (data.precision*100)
                data.recall = '%.2f%%' % (data.recall*100)
                if 'F1' in data:  data.F1 = '%.5f' % (data.F1)
                else: data.F1 = ''
                raw = ['%s -> %s' % (src_chain, dst_chain), data.TP, data.FP, data.FN, data.precision, data.recall, data.F1]
                ws.append(raw)
        else:
            data = ed(statis['summary'])
            data.precision = '%.2f%%' % (data.precision*100)
            data.recall = '%.2f%%' % (data.recall*100)
            if 'F1' in data:  data.F1 = '%.5f' % (data.F1)
            else: data.F1 = ''
            raw = [src_chain, data.TP, data.FP, data.FN, data.precision, data.recall, data.F1]
            ws.append(raw)

    wb.save(path)
