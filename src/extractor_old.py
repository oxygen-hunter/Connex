# -*- coding:utf-8 -*-

from web3 import Web3
import random, json, os, copy, logging, string, itertools
from easydict import EasyDict as ed
from collections import defaultdict as dd
import typing as T
from src import tools as TL, config as C, secret as S, doc_info as DOC, tracer as TCE
from src.common import my_timer
import openai
import httpx


class Extractor(TL.RPCHandler):

    chain_id_to_name = {
        "Multichain": {
            43114: 'Avalanche', 56: 'BSC', 42220: 'Celo', 25: 'Cronos', 1: 'Ethereum',
            250: 'Fantom', 122: 'Fuse', 32659: 'Fusion', 4689: 'IOTEX', 1666600000: 'Harmony',
            128: 'Huobi', 321: 'Kucoin', 1285: 'Moonriver', 1284: 'Moonbeam', 66: 'OKExChain',
            137: 'Polygon', 336: 'Shiden', 40: 'Telos', 84: 'xDai', 42161: 'Arbitrum',
            288: 'Boba', 10: 'Optimism'},
        "Stargate":{
            101:'Ethereum', 102:"BSC", 106:"Avalanche", 109:"Polygon", 110:"Arbitrum", 112:"Fantom", 111:"Optimism"
        },
        "Portal":{}
    }
    def __init__(self, check_rpc:bool=True) -> None:
        """
            Extractor basic
        """
        self.extract_statis = {}
        self.flatten_keys_diff_count = dd( lambda : {"src": dd(lambda: 0), "dst": dd(lambda: 0)} )
        self.flatten_keys_total_count = {}
        super().__init__(check_switch=check_rpc)

    def get_chain_name_from_id(self, chainid:int, bridge_name:str):
        for brg in self.chain_id_to_name.keys():
            if bridge_name.startswith(brg): 
                return self.chain_id_to_name[brg].get(chainid, '')
        return ''

    def extract_by_rule(self, tx:T.Dict[str, T.Any], logs:T.List[T.Dict[str, T.Any]], rules:T.Dict[str, T.List[str]]):
        return TL.extract_values_by_given_fields(tx, logs, rules)

    def deal_token_and_amount(self, token_addr:str, amount:int, tx_obj:dict, chain:str, w3):
        """token name, token symbol, decimal, share w3 endpoint"""
        token_name = ''
        token_symbol = ''
        decimals = 0
        if TL.save_value_int(token_addr) == 0: 
            token_name, token_symbol = 'eth', 'eth'
            decimals = 18
        elif isinstance(token_addr, str) and len(token_addr):
            try:
                token_name, token_symbol, decimals = TL.get_token_name_symbol_decimals(
                    addr=token_addr, chain=chain, w3=w3)
            except Exception as e:
                token_name, token_symbol = 'eth', 'eth'
                decimals = 18
                logging.error("Error handling trx:" + tx_obj['hash'])
                logging.exception(e)
        # compute amount = amount / 10^decimal
        amount = int(amount) / 10**decimals

        return token_name, token_symbol, amount


    def _flatten_all_keys(self, d:dict, cur_key:str=''):
        tmp_ret = []
        ret = []
        for subkey in d.keys():
            value = d[subkey]
            if isinstance(value, dict):
                tmp_ret.extend(self._flatten_all_keys(value, subkey))
            else:
                tmp_ret.append(subkey)
        if len(cur_key) and len(tmp_ret):
            for r in tmp_ret:
                ret.append(cur_key + '.' + r)
            return ret
        else:
            return tmp_ret

    def get_diverse_field(self, all_data:T.List[T.List[dict]], chain:str):
        p = C.paths().diversed_fields
        flag_rerun = True
        if os.path.exists(p):
            _t = json.load(open(p))
            self.flatten_keys_diff_count = _t['unique_counts']
            self.flatten_keys_total_count = _t['all_counts']
            if chain in self.flatten_keys_diff_count: 
                flag_rerun = False

        if flag_rerun:
            keys_value_pair = {"src": dd(set), "dst": dd(set)}
            keys_count = {"src": dd(lambda: 0), "dst":dd(lambda: 0)}
            for one_data in all_data:
                if not len(one_data[0]): continue 
                func_name = one_data[0]['pureFunctionName'] or one_data[0]['methodId']
                role = DOC.function_name_to_role[C.current_bridge].get(func_name, None)
                if not role: continue
                all_keys = self._flatten_all_keys(one_data[0],'transaction')
                if len(one_data) > 1: 
                    for log_data in one_data[1:]:
                        if not len(log_data): continue
                        all_keys.extend(self._flatten_all_keys(log_data, 'log[%s]' % log_data['event']))
                a = self.extract_by_rule(one_data[0], 
                                one_data[1:] if len(one_data) > 1 else [], 
                                {key:[key] for key in all_keys})
                for k,v in a.items(): 
                    keys_count[role][k] += 1
                    if isinstance(v, (list, dict, ed)): 
                        v = json.dumps(v, sort_keys=True)
                    keys_value_pair[role][k].add(v)
            
            tmp_dict = {"src": dd(lambda: 0), "dst":dd(lambda: 0)}
            for role in keys_value_pair:
                for k in keys_value_pair[role]: 
                    tmp_dict[role][k] = len(keys_value_pair[role][k])
            self.flatten_keys_diff_count[chain] = tmp_dict
            self.flatten_keys_total_count[chain] = keys_count
            _t = {"unique_counts":self.flatten_keys_diff_count,"all_counts":self.flatten_keys_total_count}
            f = open(p, 'w')
            json.dump(_t, f, indent=1)
            f.close()


class ExtractorRule(Extractor):

    def __init__(self, use_log = False) -> None:
        super().__init__()
        self.use_log = use_log

    def go(self, tx_obj:ed, chain: str, 
           logs_dict:T.List[T.Dict[str, T.Any]]=None, extract_rules:dict={}) -> T.Tuple[str, str, str, str, float, str, str, str]:
        """
            @return: Tuple of (to addr, token addr, token name, token symbol, amount, timestamp, role, paired chain)
        """
        w3 = self.choose_one_rpc(chain)
        tx_obj = ed(tx_obj)
        # 抽取信息
        extracted_infos = self.extract_by_rule(tx_obj, logs_dict, extract_rules)

        to, token_addr, amount, timeStamp, role, = (
            extracted_infos.get('to', None),
            extracted_infos.get('token_addr', ''),
            extracted_infos.get('amount', None),
            extracted_infos.get('timeStamp', None),
            extracted_infos.get('role', None),
        )
        if C.current_bridge.startswith('Optimism'):
            paired_chain = extracted_infos.get('paired_chain', None)
        else:
            paired_chain_id = extracted_infos.get('paired_chain_id', None)
            paired_chain = self.get_chain_name_from_id(paired_chain_id, C.current_bridge)
        amount_origin = amount
        token_name, token_symbol, amount = self.deal_token_and_amount(token_addr, amount, tx_obj, chain, w3)
        return {'to':to, 'token_addr':token_addr, 'token_name':token_name,
                'token_symbol':token_symbol, 'amount':amount, 'timestamp':(timeStamp), 
                'role':role, 'paired_chain':paired_chain, 'amount_origin':amount_origin,
        }
        # return to, token_addr, token_name, token_symbol, amount, timeStamp, role, paired_chain

    def decode_src_tx_hash_from_dst_tx(self, tx_obj, chain) -> str:
        '''
        @return src tx hash decoded from dst tx, or ''
        '''
        w3 = self.choose_one_rpc(chain)
        abi = TL.get_abi(tx_obj.to, chain) # TODO: abi not found
        ct = w3.eth.contract(Web3.to_checksum_address(tx_obj.to), abi=abi)
        try:
            a = ct.decode_function_input(tx_obj.input)
            args = ed(a[1])
        except ValueError as e:
            logging.exception(e)
            return ''
        # txs is bytes encoded by web3
        # for example: b"3\x9c\xd7K;\xa9f\xc3'R`\x89\x99\xa5g%qa\xc6\x15,\xe4Z\x8e\x03\xdb\x1e\x11\xc3}\xebY"
        src_tx_hash = args.get('txs', '')
        src_tx_hash = Web3.to_hex(src_tx_hash).lower() if len(src_tx_hash) else ''
        return src_tx_hash


class ExtractorDL(Extractor):
    
    QUERY_INSTR = 'Represent this sentence for searching relevant passages:'

    def __init__(self, encoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.pre_encoded_queries = []
        self.preset_queries = ('to', 'amount', 'token', 'pair chain')
        self.all_tokens = set()
    
    def _get_pre_encoded(self)->T.List[T.Any]:
        """
            encode the query keys, return their embeddings
        """
        if len(self.pre_encoded_queries): 
            return self.pre_encoded_queries
        self.pre_encoded_queries = self._encode_keys(self.preset_queries)
        return self.pre_encoded_queries
    
    def _encode_keys(self, keys:T.List[str], add_prefix:bool=True)->T.List[T.Any]:
        ret = []
        for k in keys:
            ret.append(self.encoder.encode((self.QUERY_INSTR if add_prefix else '') + k, show_progress_bar=False)) 
        return ret

    @staticmethod
    def _nest_dict(d:dict, cur_key='')->T.List[str]:
        tmp_ret = []
        ret = []
        for subkey in d.keys():
            value = d[subkey]
            if isinstance(value, dict):
                tmp_ret.extend(ExtractorDL._nest_dict(value, subkey))
            else:
                value_type = TL.guess_para_type(value)
                tmp_ret.append(subkey + ' ' + value_type)
        if len(cur_key) and len(tmp_ret):
            for r in tmp_ret:
                ret.append(cur_key + ' ' + r)
            return ret
        else:
            return tmp_ret

    @staticmethod
    def _cut_varnames(name:str):
        """
            Given a long variable name, cut it into (short) sentence
        """
        my_dict = ['token', 'ERC20','Token']
        for idx, word in enumerate(my_dict):
            while word in name:
                name = name.replace(word, '$'+str(idx) + ' ', 1)
        new_name = ''
        idx = 0
        while idx < len(name):
            char = name[idx]
            if char == '$': 
                idx += 1
                num = ''
                while name[idx].isdigit():
                    num += name[idx]
                    idx += 1
                num = int(num)
                new_name += ' %s ' % (my_dict[num])
                continue
            elif char in string.ascii_uppercase:
                new_name += ' '
                new_name += char
            else:
                new_name += char
            idx += 1
        return new_name

    @staticmethod
    def extract_from_compound_keys(comp_keys:str, trx:T.Dict[str, T.Any], 
                                   logs:T.List[T.Dict[str, T.Any]], 
                                   reshuffle:bool=True):
        ck = list(a for a in comp_keys.split(' ') if len(a))
        if reshuffle:
            ck = [ck[-3], ck[-1]] + ck[0:-7]
        candidates = [] # log name may be repeated, so we use a canidate-list to contain all possible object
        if ck[0] == 'transaction':
            candidates.append(trx['args'])
            remain_keys = ck[2:]
        else:
            obj = None
            log_name = ck[1]
            for log in logs :
                if not len(log): continue
                if log['event'] == log_name :
                    if ck[2] == 'emitAddress':
                        candidates.append( [log['address']] + log['args'])
                    else:
                        candidates.append(log['args'])
            remain_keys = ck[2:]
        assert len(candidates)
        for obj in candidates:
            try:
                for rk in remain_keys:
                    obj = obj[rk]
                return obj
            except KeyError: 
                continue
        return obj
    
    @staticmethod
    def _prepare_all_keys(trx:dict, logs:T.List[dict]):
        """
            Prepare a set of keys from transaction's function name, function arguments,
            and log event's names + arguments
        """
        # deal transactions 
        if len(trx):
            trx_keys = ExtractorDL._nest_dict(trx['args'], 'transaction ' + trx['pureFunctionName'])
        else:
            trx_keys = []
        # deal logs
        log_keys = []
        for log_entry in logs:
            if not len(log_entry): continue
            one_log_keys = ExtractorDL._nest_dict(log_entry['args'])
            one_log_keys = map(lambda x: "log %s %s" % (log_entry['event'] , x), one_log_keys)
            log_keys.extend(one_log_keys)
            # Additional: the emit address of a log entry is considered
            # log_keys.append("log %s emitAddress %s address" % (log_entry['event'], ' '.join(log_entry['args'].keys())))

        all_keys_old:T.List[str] = trx_keys + log_keys
        all_keys_new:T.List[str] = []
        # extend the semantics from template 
        # the keys follow the forms of: 
        # <source>(transaction/log), <name>, <arguments component1>[,<arguments component2>, ...]

        for key in all_keys_old:
            tmp = key.split(' ')
            all_keys_new.append(
                "{arguments} with type {_type} from {source} named {name} ".format(
                    arguments=' '.join(tmp[2:-1]),
                    _type=tmp[-1],
                    name=tmp[1],
                    source=tmp[0]
                )
            )
        return all_keys_new

    def _add_extracted_top5(self, key:str, arrays:T.Iterable[str]):
        if key not in self.extract_statis:
            self.extract_statis[key] = {}
        top = ';'.join(arrays)
        if top not in self.extract_statis[key]:
            self.extract_statis[key][top] = 0
        self.extract_statis[key][top] += 1

    def _encode_and_extract_from_given_keys_queries(
            self, trx:dict, logs:T.List[dict],
            candidate_keys:T.List[str], 
            query_keys:T.List[str], query_values:T.List[T.Any] = ())->dict:
        """
            encode all keys from {name-value} set, and extract the target keys(Queries) from 
            the most close embeddings
        """
        all_keys_old = candidate_keys
        if not len(all_keys_old): 
            return {}
        all_keys = list(map(lambda x: self._cut_varnames(x), all_keys_old))
        for k in all_keys:
            self.all_tokens.add(' '.join(self.encoder.tokenizer.tokenize(k)))
        all_keys_embedding = self.encoder.encode(all_keys, show_progress_bar=False)
        # this stores tuple (extracted value, scores, the top5 result)
        chosen_keys:T.List[T.Tuple[T.Any, float, T.Dict[str, T.Any]]] = []
        
        if len(query_keys) and not len(query_values):
            query_values = self._encode_keys(query_keys)
        for i, p in enumerate(query_values):
            all_keys_sim_scores = {all_keys[i] : float(util.cos_sim(p, x).squeeze()) for (i, x) in enumerate(all_keys_embedding)}
            sorted_keys = sorted(all_keys, key=lambda x: all_keys_sim_scores[x], reverse=True)
            top5_keys = sorted_keys[:min(5, len(sorted_keys))]
            self._add_extracted_top5(query_keys[i], top5_keys)
            top5_record:T.Dict[str, T.Any] = {} # from key to value
            for _key_in_top5 in top5_keys:
                index = all_keys.index(_key_in_top5)
                assert index != -1, "Error in getting keys when extracting:" + _key_in_top5
                _key_old = all_keys_old[index]
                _value_in_top5 = self.extract_from_compound_keys(_key_old, trx, logs)
                top5_record[_key_in_top5] = _value_in_top5
            most_close = sorted_keys[0]
            chosen_keys.append((top5_record[most_close], all_keys_sim_scores[most_close], top5_record))
        extracted_values = chosen_keys
        fields1 = dict(zip(query_keys, extracted_values))
        return fields1

    def _encode_and_extract(self, trx:dict, logs:T.List[dict])->dict:
        return self._encode_and_extract_from_given_keys_queries(
                trx, logs, 
                self._prepare_all_keys(trx, logs), self.preset_queries ,self._get_pre_encoded())

    def go(self, trx:dict, logs:T.List[dict], chain:str, extract_rules:dict={}):
        w3 = self.choose_one_rpc(chain)
        
        fields1 = self._encode_and_extract(trx, logs)
        if not len(fields1):
            return {}
        fields1 = list(map(lambda x: x[0],  fields1))
        # some rule-based fields
        # Role, timeStamp, paired chain(Optimism)
        rule_based_keys = ['role', 'timeStamp']
        if C.current_bridge.startswith('Optimism'):
            rule_based_keys += ['paired_chain']
        rule_based_rules = {}
        for k in rule_based_keys:
            rule_based_rules[k] = extract_rules[k]
        fields2 = self.extract_by_rule(trx, logs, rule_based_rules)

        if C.current_bridge.startswith('Multichain'):
            pair_chain_id = fields1['pair chain']
            pair_chain = self.get_chain_name_from_id(pair_chain_id, C.current_bridge)
        else:
            pair_chain = fields2['paired_chain']
        if not fields1['amount']:
            fields1['amount'] = 0
        try:
            token_name, token_symbol, amount = self.deal_token_and_amount(fields1['token'], int(fields1['amount']), trx, chain, w3)
        except ValueError:
            token_name, token_symbol, amount = self.deal_token_and_amount(fields1['token'], int(fields1['amount'], base=16), trx, chain, w3)

        return {'to':fields1['to'], 'token_addr':fields1['token'], 'token_name':token_name,
                 'token_symbol':token_symbol, 'amount':amount, 'timestamp':(fields2['timeStamp']), 
                 'role':fields2['role'], 
                 'paired_chain':pair_chain
            }


class ExtractorCompound(ExtractorDL):

    # SRC_CHAIN = 'the unique identifier of source chain which can be an ID or name'
    # DST_CHAIN = 'the unique identifier of destination chain which can be an ID or name'
    # AMOUNT = 'the amount of swapped or exchanged token'
    # TO_ADDR = 'the receiver or target address'
    # TOKEN_ADDR = 'the unique identifier of token which can be an address or name'
    SRC_CHAIN = 'source chain'
    DST_CHAIN = 'the unique identifier of destination chain which can be an ID or name'
    AMOUNT = 'the amount of swapped or exchanged token'
    TO_ADDR = 'the receiver or target address'
    TOKEN_ADDR = 'the unique identifier of token which can be an address or name'
    def __init__(self, encoder) -> None:
        super().__init__(encoder)
        self.preset_queries = (
            self.SRC_CHAIN,
            self.DST_CHAIN,
            self.AMOUNT,
        )

    def go(self, trx:dict, logs:T.List[dict], chain:str, extract_rules:dict={}):
        
        # 1. paired chain and token amount
        fields1 = self._encode_and_extract(trx, logs)
        if not len(fields1): 
            logging.warning("Fields1 extract fail. hash: %s" % (trx['hash']))
            return {}
        src_chain_value, src_chain_score, _ = fields1[self.SRC_CHAIN]
        dst_chain_value, dst_chain_score, _ = fields1[self.DST_CHAIN]
        if src_chain_score > dst_chain_score :
            # NOTE: src trx would contain a "dest" argument, and vice versa
            role = 'dst'
            paired_chain = src_chain_value
        else:
            role = 'src'
            paired_chain = dst_chain_value

        # 2. timestamp
        rule_based_keys = ['timeStamp']
        # Optimism is special: 
        # it only supports 2 chain, so there would not be such args
        if C.current_bridge.startswith('Optimism'):
            rule_based_keys += ['paired_chain']
        rule_based_rules = {}
        for k in rule_based_keys:
            rule_based_rules[k] = extract_rules[k]
        fields2 = self.extract_by_rule(trx, logs, rule_based_rules)
        timestmap = fields2['timeStamp']
        if C.current_bridge.startswith('Optimism'):
            paired_chain = fields2['paired_chain']
        if isinstance(paired_chain, int):
            paired_chain = self.get_chain_name_from_id(paired_chain, C.current_bridge)

        # 3. src/dst address, token type(address)
        # first, find all address type from the attr values
        all_keys_old = self._prepare_all_keys(trx, logs)
        candidate_addr_keys_old = []
        for key in all_keys_old:
            value = self.extract_from_compound_keys(key, trx, logs)
            if TL.str_is_address_type(value):
                candidate_addr_keys_old.append(key)
        # candidate_addr_keys = list(map(lambda x: self._cut_varnames(x), candidate_addr_keys_old))
        queries = (self.TO_ADDR, self.TOKEN_ADDR)
        candidate_addr_results = self._encode_and_extract_from_given_keys_queries(trx, logs, candidate_addr_keys_old, queries)

        to_addr = candidate_addr_results[self.TO_ADDR][0]
        token_addr = candidate_addr_results[self.TOKEN_ADDR][0]

        w3 = self.choose_one_rpc(chain)

        try:
            token_name, token_symbol, amount = self.deal_token_and_amount(token_addr, int(fields1[self.AMOUNT][0]), trx, chain, w3)
        except ValueError:
            token_name, token_symbol, amount = self.deal_token_and_amount(token_addr, int(fields1[self.AMOUNT][0], base=16), trx, chain, w3)

        return {
            'to':to_addr, 'token_addr':token_addr, 'token_name':token_name,
                 'token_symbol':token_symbol, 'amount':amount, 'timestamp':(timestmap), 
                 'role':role, 
                 'paired_chain':paired_chain,
                 'amount_origin':fields1[self.AMOUNT][0], 
                 'top5': {
                     'to': candidate_addr_results[self.TO_ADDR][2],
                     'token_addr':candidate_addr_results[self.TOKEN_ADDR][2],
                     'amount':fields1[self.AMOUNT][2]
                 }
        }
