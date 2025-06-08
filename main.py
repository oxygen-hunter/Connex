# -*- coding:utf-8 -*-

import os
import typing as T
import json
import logging
from datetime import datetime
from easydict import EasyDict as ed
import argparse, yaml
import openpyxl
import atexit

from src import config as C
from src import tools as TL
from src.extractor import ExtractorLLM
from src.decoder import TxDecoder, LogDecoder
from collections import defaultdict as dd


def _decode_fields(tx_data:T.List[T.Dict], chain: str, 
                   log_data:T.Dict[str,T.Dict]={}):
    ret = []
    tx_decoder = TxDecoder()
    log_decoder = LogDecoder()
    no_log_cnt = 0
    for i, trx in enumerate(tx_data):
        trx = ed(trx)
        try:
            tx_obj = tx_decoder.go(trx, chain)
            one_decoded_info = [tx_obj]
            if len(log_data):
                decoded_logs = log_decoder.go(log_data.get(trx.hash, []), chain)
                one_decoded_info.extend(decoded_logs)
            else:
                no_log_cnt += 1
            ret.append(one_decoded_info)
        except Exception as e:
            logging.exception(e)
    a = '\n'
    if no_log_cnt: 
        logging.warning(f"no log record({no_log_cnt}) of {chain}")
    logging.info(f"fail trx record({len(tx_decoder.fail_record)}) of {chain}: {a.join('%s:%s'% (k, v) for k, v in tx_decoder.fail_record.items())}")
    logging.info(f"fail log record({len(log_decoder.fail_record)}) of {chain}: {a.join('%s:%s'% (k, v) for k, v in log_decoder.fail_record.items())}")
    return ret

def generate_decoded_kv(use_log:bool=False):
    """将raw data 解码成KV对的形式"""

    print('---decode begin---')
    logging.info('---decode begin---')
    chains_to_pair = C.chains_to_pair
    
    chain_txs: T.Dict[str, T.Dict] = dict() # load from file
    chain_logs:T.Dict[str, T.List] = {}     # load from file
    chain_decoded_data = {}     # write to file 
    decoded_count = 0
    tx_decoders, log_decoders = [], []
    p = C.paths().decoded_data

    if os.path.exists(p):
        f = open(p)
        chain_decoded_data = json.load(f)
        f.close()

    # tx_decoder, log_decoder = TxDecoder(), LogDecoder()
    for chain in chains_to_pair:
        if chain in chain_decoded_data:
            logging.info("chain %s decoded data exsits, length:%d. Skip" % (chain, len(chain_decoded_data[chain])))
            continue
        logging.info("begin decoding chain:" + str(chain))
        chain_txs[chain], chain_logs[chain] = TL.load_all_data_by_chain(chain, use_log)
        decoded_data = _decode_fields(chain_txs[chain], chain, chain_logs[chain])
        chain_decoded_data[chain] = decoded_data
        decoded_count += len(decoded_data)
        # for every successfully decoded chain, write the output immediatly 
        with open(p, 'w') as f:
            json.dump(chain_decoded_data, f, indent=1, cls=TL.BytesDecoder)
        TL.at_exit()
    print("---decode end, count: %d---" % decoded_count)
    logging.info("---decode end, count: %d---" % decoded_count)

    # Check unsuccessful trx / log 
    for chain in chains_to_pair: 
        fail_tx_cnt = 0
        fail_log_count = 0
        fail_log_of_trx_count = 0
        data_one_chain = chain_decoded_data[chain]
        for one_rec in data_one_chain: 
            if not len(one_rec): 
                fail_tx_cnt += 1
            no_fail_log = True 
            for _log in one_rec[1:]: 
                if not len(_log): 
                    no_fail_log = False 
                    fail_log_count += 1
            if not no_fail_log: 
                fail_log_of_trx_count += 1
        logging.info(f"chain:{chain}, fail_tx:{fail_tx_cnt}, fail_log:{fail_log_count}, fail_log_of_trx:{fail_log_of_trx_count}")


def generate_chains_key_info_rule(use_log:bool=False):
    """
        生成key info 
    """

    def extract_fields(data: T.List[T.Dict], chain: str, extractor=None) -> T.List[T.Dict]:
        logging.info("Extracting chain %s" % (chain))
        trx_key_info_list = []
        ext = extractor or ExtractorRule()
        for d in data:
            trx = ed(d[0])
            # TODO: some chain doesn't have txreceipt_status or has other names
            if trx.get('txreceipt_status', '0') == '0': continue # unsuccess transaction, 
            try:
                trx_key_info = ext.go(
                    trx, chain=chain, logs_dict=(d[1:] if len(d) > 1 else []), extract_rules=C.info_ext_config[C.current_bridge])
            except TypeError as e:
                if e.args[0] == "int() argument must be a string, a bytes-like object or a number, not 'NoneType'":
                    # noise
                    logging.error("Error dealing tx(noise):%s, mothed:%s,chain %s" % (trx.hash, trx.methodId,chain))
                continue
            except Exception as e:
                logging.error("Error dealing tx:%s, chain %s" % (trx.hash, chain))
                logging.exception(e)
                continue
            trx_key_info.update({'hash': trx.hash,'chain': chain,})
            trx_key_info = ed(trx_key_info)
            if not (len(trx_key_info.to) and len(trx_key_info.paired_chain) and len(trx_key_info.token_symbol) and trx_key_info.amount):
                # if not trx.input.startswith('0xb1a1a882'):
                logging.warning("some info extract fail! Trx hash: %s, info:%s" % (
                    str(trx.hash), str(trx_key_info)))
            trx_key_info_list.append(trx_key_info)
        return trx_key_info_list

    print('---extract begin---')
    chains_key_info = dict()
    extractor = ExtractorRule()
    dec_data = json.load(open(C.paths().decoded_data))
    for chain in dec_data:
        chains_key_info[chain] = extract_fields(dec_data[chain], chain, extractor)
    with open(C.paths().key_info_rule, 'w') as f:
        json.dump(chains_key_info, f, indent=2)
    print('---extract end---')


def generate_chains_key_info_dl(no_check_rpc:bool=False):
    """
        使用DL model生成key info
        See: https://huggingface.co/sentence-transformers
    """

    def extract_fields_dl(data: T.List[T.List[T.Dict]], chain: str, ext):
        key_info_list = []
        ok_cnt, fail_cnt, exp_cnt, not_in_cnt = 0, 0, 0, 0
        for i, one_trx in enumerate(data):
            if not len(one_trx[0]): continue
            try:
                info = ext.go(one_trx[0], one_trx[1:] if len(one_trx) > 1 else [], chain)
            except Exception as e:
                logging.exception(e)
                exp_cnt += 1
                continue
            if info == -1: 
                fail_cnt += 1
            elif info == -2:
                not_in_cnt +=1 
            elif len(info):
                key_info_list.append(info)
                ok_cnt += 1
        print(f"chain:{chain}, OK:{ok_cnt}, fail:{fail_cnt}, {exp_cnt}, {not_in_cnt}")
        return key_info_list

    print('---extract begin---')
    from sentence_transformers import SentenceTransformer
    from src.extractor_sim import ExtractorSim
    model = SentenceTransformer(C.paths().model.sentence_bert, device='cuda').cuda()
    extractor = ExtractorSim(model, '', not no_check_rpc)

    chains_key_info = {}
    dec_data = json.load(open(C.paths().decoded_data))
    extractor.calc_meta_structure_count(dec_data)

    extractor.go_for_voting_all_data(dec_data)
    extractor.verify_voting_result(dec_data)

    for chain in C.chains_to_pair:
        chains_key_info[chain] = extract_fields_dl(dec_data[chain], chain, extractor)
    with open(C.paths().key_info, 'w') as f:
        json.dump(chains_key_info, f, indent=2)
    # with open(C.paths().extracted_tokens, 'w') as f:
    #     for token in extractor.all_tokens:
    #         f.write(token + '\n')
    # with open(C.paths().extract_statis, 'w') as f:
    #     json.dump(extractor.extract_statis, f, indent=1)
    print('---extract end---')

def generate_chains_key_info_llm(LLM_api_conf_name:str='', no_check_rpc:bool=False):

    def _extract_fields_per_chain(data: T.List[T.List[T.Dict]], chain: str):
        key_info_list = []
        ok_cnt, fail_cnt, exp_cnt, not_in_cnt = 0, 0, 0, 0
        for i, one_trx in enumerate(data):
            if not len(one_trx[0]): continue
            try:
                info = ext.go(one_trx[0], one_trx[1:] if len(one_trx) > 1 else [], chain)
            except Exception as e:
                logging.exception(e)
                exp_cnt += 1
                continue
            if info == -1: 
                fail_cnt += 1
            elif info == -2:
                not_in_cnt +=1 
            elif len(info):
                key_info_list.append(info)
                ok_cnt += 1
        print(f"chain:{chain}, OK:{ok_cnt}, fail:{fail_cnt}, {exp_cnt}, {not_in_cnt}")
        return key_info_list

    print('---extract begin---')
    ext = ExtractorLLM(LLM_api_conf_name or 'xiaohumini-gpt-4o-mini', check_rpc=not (no_check_rpc))
    dec_data = json.load(open(C.paths().decoded_data))
    ext.calc_meta_structure_count(dec_data)
    # first round to traverse all data
    # for chain in C.chains_to_pair:
    #     ext.get_diverse_field(dec_data[chain], chain)

    # Voting
    start_time = datetime.now()
    ext.go_for_voting_all_data(dec_data)

    end_time = datetime.now()
    diff = end_time - start_time
    s = f"Total used ny LLM: total {diff.seconds} seconds = {diff.seconds // 3600}H {(diff.seconds % 3600)//60}min {diff.seconds % 60}s"
    print(s)
    logging.info(s)

    ext._save_LLM_record()
    ext.verify_voting_result(dec_data)

    chains_key_info = {}
    for chain in C.chains_to_pair:
        chains_key_info[chain] = _extract_fields_per_chain(dec_data[chain], chain)
    f = open(C.paths().key_info, 'w')
    json.dump(chains_key_info, f, indent=1)
    f.close()
    print('---extract end---')


def analyze():
    from src.analyzer import generate_FP_plus,generate_FN_plus, generate_TP_plus,analyze_FN_plus_token
    logging.info('Analyzing results')
    print('---analyze begin---')
    generate_FP_plus()
    generate_FN_plus()
    generate_TP_plus()

    analyze_FN_plus_token()
    
    print('---analyze end---')


def main(args):
    atexit.register(TL.at_exit)
    yaml_config = TL.set_args_from_yaml_config(args)

    if args.bridge:
        C.set_globals('current_bridge', args.bridge)
        C.reset_paths(args.LLM_sub_dir)
    
    if args.LLM_sub_dir:
        C.reset_paths(args.LLM_sub_dir)
        os.makedirs( os.path.join(C.paths().bridge_results, args.LLM_sub_dir), exist_ok=True)


    if args.chains_to_pair:
        C.set_globals('chains_to_pair', args.chains_to_pair.split(','))
    
    if args.force_proxy:
        C.set_globals('force_proxy', True)
    
    if args.time_window:
        C.set_globals('HYPERPARAMETER_TIME_WINDOW', int(args.time_window))
    
    if args.fee_rate:
        C.set_globals('HYPERPARAMETER_FEE_RATE', float(args.fee_rate))

    start_time = datetime.now()
    log_filename = os.path.join(C.paths().logs, datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '_' + args.phase +f'_{C.current_bridge}.log' )
    print("log filename: " + log_filename)
    logging.basicConfig(filename=log_filename,
                    format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)
    
    logging.info(f"current config: {str(yaml_config)}, {str(args)}")

    os.makedirs(C.paths().bridge_results, exist_ok=True)
    if args.phase in ('all', 'decode'):
        # 将raw data解码成 K-V 形式
        generate_decoded_kv(True)
        
    if args.phase == 'all' or args.phase == 'extract':
        if args.ext_type == 'dl':
            generate_chains_key_info_dl(args.no_check_rpc)
        elif args.ext_type == 'llm':
            generate_chains_key_info_llm(args.LLM_api_conf, args.no_check_rpc)
        else:
            raise NotImplementedError(f'{args.ext_type} not supported.')
    if args.phase == 'all' or args.phase == 'pair':
        from src.rules import generate_pairing_res
        generate_pairing_res() 
    if args.phase == 'all' or args.phase == 'check':
        from src.checker import check_pairing_res
        check_pairing_res()
    if args.phase == 'all' or args.phase == 'analyze':
        analyze()
    
    end_time = datetime.now()
    diff = end_time - start_time
    s = f"Total used: total {diff.seconds} seconds = {diff.seconds // 3600}H {(diff.seconds % 3600)//60}min {diff.seconds % 60}s"
    print(s)
    logging.info(s)

if __name__ == '__main__':
    if not os.path.exists(C.paths().logs):
        os.makedirs(C.paths().logs)
    argp = argparse.ArgumentParser()
    argp.add_argument('--phase', choices=['all', 'decode','extract', 'pair', 'match' ,'check', 'analyze'], required=False, default='all', help='which phase want to exec. Default: all')
    argp.add_argument('--config', required=False, default='', help='Specify a config file name, e.g Stargate.')
    argp.add_argument('--bridge', required=False, default='',help='Specify a bridge name, e.g Multichain-2023-7-7. NOTE: if you specify this variable, then the current_bridge in config file will be override.')
    argp.add_argument('--chains_to_pair', required=False, help='comma splitted string that specify chains to pair, e.g Ethereum,Optimism . NOTE: if you specify this variable, then the chains_to_pair in config file will be override.')
    argp.add_argument('--LLM_api_conf', required=False, default='',help='The api config filename of LLM, e.g OpenAI-2023-7-7(no file extension). NOTE: if you specify this variable, then the LLM_api_conf in config file will be override.')
    argp.add_argument('--LLM_sub_dir', required=False, default='',help='If you use different LLM other than default and want to save the result in another sub_dir, designate this arg. Note: then the LLM_sub_dir in config file will be override.')
    argp.add_argument('--time_window', required=False, default='',help='The time window for the checking algorithm. NOTE: if you specify this variable, then the HYPERPARAMETER_TIME_WINDOW in config file will be override.')
    argp.add_argument('--fee_rate', required=False, default='',help='The fee rate for the checking algorithm. NOTE: if you specify this variable, then the HYPERPARAMETER_FEE_RATE in config file will be override.')
    argp.add_argument('--no_check_rpc', action="store_true" ,default=False, required=False, help='if don\'t check rpc endpoints(which reduce a lot of time), set this to true; else ignore this flag')
    argp.add_argument('--force_proxy', action="store_true" ,default=False, required=False, help='if set to true, force the program to use proxy(readed from data/proxy.json). NOTE: if you specify this variable, then the chains_to_pair in config file will be override.')
    # argp.add_argument('--ext_rule', action="store_const", dest='ext_type', const='rule', help='use rule to extract info')
    argp.add_argument('--ext_dl', action="store_const", dest='ext_type', const='dl', help='use Deep Learning to extract info')
    # argp.add_argument('--ext_comp', action="store_const", dest='ext_type', const='comp', help='use Deep Learning to extract info')
    argp.add_argument('--ext_llm', action="store_const", dest='ext_type', const='llm', help='use LLM to extract info')
    args = argp.parse_args()
    # try:
    #     main(args)
    # except Exception as e:
    #     logging.exception(e)
    main(args)