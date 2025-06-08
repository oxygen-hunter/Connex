# -*- coding:utf-8 -*-

import typing as T 
import json,os 
from src import config as C, tools as TL, doc_info as DOC
from multiprocessing import Pool
from collections import defaultdict
import itertools, copy
import dill
from easydict import EasyDict as ed

def get_rules() -> T.List[T.Callable[[T.Any, T.Any], bool]]:
    def rule1_to_addr(tx1, tx2, **kwargs):
        if not (isinstance(tx1.to, str) and isinstance(tx2.to, str)):
            return False
        return tx1.to.lower() == tx2.to.lower()

    def rule2_timestamp(tx1, tx2, **kwargs):
        # return True # 暂时不管timestamp，看数据是怎样的
        timewindow = kwargs.get('timewindow', None) or C.HYPERPARAMETER_TIME_WINDOW
        if not (tx2.timestamp) or not (tx1.timestamp): 
            return False
        if isinstance(tx2.timestamp, str) and isinstance(tx1.timestamp, str) and not (len(tx2.timestamp) and len(tx1.timestamp) ): 
            return False
        time_diff = TL.save_value_int(tx2.timestamp) - TL.save_value_int(tx1.timestamp)
        return time_diff >= 0 and time_diff < timewindow # tx2晚于tx1，且时间差有限

    def rule3_token_name(tx1, tx2, **kwargs):
        if TL.check_whether_token_name_and_symbol_match(tx1.token_name, tx1.token_symbol, tx2.token_name, tx2.token_symbol):
            return True 
        symbol1, symbol2 = list(map(DOC.get_token_symbol_type, (tx1.token_symbol, tx2.token_symbol)))
        if (symbol1 , symbol2) in (("USD", "eth"), ("eth", "USD")):
            # NOTE: tolerate USD and eth:
            return True 
        return False

    def rule4_fee(tx1, tx2, **kwargs):
        # consider both rate and amount, because
        # 1. bridge has min fee (depends on token's price / from chain / to chain),
        #    e.g. min fee for TetherUSD bridged from Ethereum to BSC is 1.9$
        # 2. bridge has fee rate (depends on token / from chain / to chain), 
        #    e.g. fee rate for wrappedETH bridged from Ethereum to Optimism is 0.001%
        if not (isinstance(tx1.amount, (int,float)) 
                and isinstance(tx2.amount, (int,float))):
            return False
        symbol1, symbol2 = list(map(DOC.get_token_symbol_type, (tx1.token_symbol, tx2.token_symbol)))
        amount1, amount2 = tx1.amount, tx2.amount
        treshold_fee_rate = kwargs.get('fee_rate', None) or C.HYPERPARAMETER_FEE_RATE
        if (symbol1 , symbol2) in (("USD", "eth"), ("eth", "USD")):
            # NOTE: tolerate USD and eth:
            # 1 ETH = $3200 ~ $4000
            if symbol1 == 'eth': 
                amount1 = tx1.amount * 3600
            elif symbol2 == 'eth': 
                amount2 = tx2.amount * 3600
            treshold_fee_rate *= 1.5
        if amount1 == 0: return False
        fee_amount = abs(amount1 - amount2)
        fee_rate = fee_amount / amount1
        # if fee_amount < 0: return False
        return fee_rate < treshold_fee_rate
    
    def rule5_role(tx1, tx2,**kwargs):
        return tx1.role == 'src' and tx2.role == 'dst'
    
    def rule6_paired_chain(tx1, tx2, **kwargs):
        if not (tx1.paired_chain) or not (tx2.paired_chain): return False
        return tx1.paired_chain == tx2.chain and tx2.paired_chain == tx1.chain

    return [rule1_to_addr, rule2_timestamp, rule3_token_name, rule4_fee, rule5_role, rule6_paired_chain]


def deal_additional_pairs(pairs:T.List[T.Tuple[str, str]], txhash2ts):
    """
        for 1-to-many pairs, choose a most close(according to timestamp) destination transaction
    """

    def _calc_timesteamp_diff(x, y):
        x = int(x)
        y = int(y)
        if x < 0 or y < 0: 
            return (1<<256)
        return (y - x) if y-x >= 0 else (1<<256)

    def get_ahead_star(edges:T.List[T.Tuple[str, str]]):
        ahead_star = defaultdict(list)
        for src, dst, in edges:
            ahead_star[src].append(dst)
        return ahead_star

    ahead_star = get_ahead_star(pairs)
    cur_pair_res = []
    for src_hash in ahead_star:
        # 选择时间戳里最接近的一个
        if len(ahead_star[src_hash]) == 1: 
            # 只有一个候选
            cur_pair_res.append((src_hash, ahead_star[src_hash][0]))
        else:
            a = sorted(ahead_star[src_hash], 
                    key=lambda x: _calc_timesteamp_diff(txhash2ts[src_hash],txhash2ts[x]) )
            if txhash2ts[a[0]] < txhash2ts[src_hash]:
                # 最近的一个也是早于src，说明都没匹配上
                assert all (txhash2ts[_x] < txhash2ts[src_hash] for _x in ahead_star[src_hash] )
                # fail_pair_res[src_chain][dst_chain].append(
                #     (src_hash, txhash2ts[src_hash], txhash2ts[a[0]])
                # )
            else:
                cur_pair_res.append((src_hash, a[0]))
    return cur_pair_res



rule_statis = {}
def pairing(tx1, tx2, rules, **kwargs) -> bool:
    for i, rule in enumerate(rules):
        if not rule(tx1, tx2, **kwargs):
            if i not in rule_statis: rule_statis[i] = 0
            # rule_statis[i] += 1
            return False, i
    return True, -1

def pairing_many_with_all(src_trx_list, dst_trx_list:list, rules, **kwargs):
    ret = []
    # os.system("taskset -p 0x7fffffffffff %d" % os.getpid())
    rules = dill.loads(rules)
    for src_trx, dst_trx in itertools.product(src_trx_list, dst_trx_list):
        check_res = (pairing(src_trx, dst_trx, rules, **kwargs))
        if check_res[0]: 
            ret.append((src_trx.hash, dst_trx.hash))

    return ret

def pairing_txs(txs1: T.List[T.Dict], txs2: T.List[T.Dict], **kwargs) -> T.List[T.Tuple[str, str]]:
    '''
        @return list of hash of paired txs
        (tx1 hash, tx2 hash) means there is a cross-chain trascation from tx1 to tx2
    '''
    paired_txs = []
    rules = dill.dumps(get_rules())

    batch_size = 100
    all_res = []
    with Pool(processes=min(80, len(txs1) // batch_size)) as pool:
        current_list = []
        for i, tx1 in enumerate(txs1):
            if tx1.timestamp == '':
                continue
            current_list.append(tx1)
            if (i and i % batch_size == 0) or (i == len(txs1) - 1): 
                all_res.append(pool.apply_async(pairing_many_with_all, (current_list, txs2, (rules)), kwds=kwargs))
                current_list = []
            # check_res_list = pairing_many_with_all(tx1, txs2, rules, **kwargs)
        pool.close()
        pool.join()
    for r in all_res:
        paired_txs.extend(r.get())
    
    txhash2ts = { x.hash:TL.save_value_int(x.timestamp) for x in itertools.chain(txs1, txs2)}
    paired_txs = deal_additional_pairs(paired_txs, txhash2ts)
    return paired_txs

def generate_pairing_res(pair_args={}, key_info_path:str='', output_path='', chains_to_pair=[]):
    print('---pairing begin---')
    chains_to_pair = chains_to_pair or C.chains_to_pair
    chains_key_info = ed()
    with open(key_info_path or C.paths().key_info, 'r') as f:
        chains_key_info = ed(json.load(f))
    pairing_res = {}
    for src_chain in chains_to_pair:
        for dst_chain in chains_to_pair:
            if src_chain == dst_chain:
                continue
            if src_chain not in pairing_res:
                pairing_res[src_chain] = {}
            if dst_chain not in pairing_res[src_chain]:
                pairing_res[src_chain][dst_chain] = []
            # res = pool.apply_async(pairing_txs, (chains_key_info[src_chain], chains_key_info[dst_chain]), kwds=pair_args)
            # debug_res.append((src_chain, dst_chain, res))
            res = pairing_txs(
                chains_key_info[src_chain], chains_key_info[dst_chain],**pair_args)

            pairing_res[src_chain][dst_chain] = res
            print(f'src chain: {src_chain}, dst chain: {dst_chain}, pair {len(res)} pairs')
    with open(output_path or C.paths().pair_res, 'w') as f:
        json.dump(pairing_res, f, indent=2)
    print('---pairing end---')