# -*- coding:utf-8 -*-

import os
import requests, time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json, random, yaml
import typing as T 
import openai
from easydict import EasyDict as ed
from functools import partial
from web3 import Web3
import eth_utils, hexbytes
import logging
import re, string, binascii, hashlib
from collections import deque, defaultdict as dd 

import tiktoken
from src import config as C, doc_info as DOC


# get api/key from config file
API_PARAMS_FOR_ABI = "?module=contract&action=getabi&address={addr}&apikey={apikey}"
API_PARAMS_FOR_ERC20_TRANSFER = "?module=account&action=tokentx&address={addr}&apikey={apikey}&page={page}&offset={offset}&startblock={startblock}&endblock={endblock}&sort=asc"
API_PARAMS_FOR_INTERNAL_TX = "?module=account&action=txlistinternal&address={addr}&apikey={apikey}&page={page}&offset={offset}&startblock={startblock}&endblock={endblock}&sort=asc"
API_PARAMS_FOR_NORMAL_TX = "?module=account&action=txlist&address={addr}&apikey={apikey}&page={page}&offset={offset}&startblock={startblock}&endblock={endblock}&sort=asc"
API_URLS = json.load(open(C.paths().etherscan_like_api_urls, 'r')) if os.path.exists(C.paths().etherscan_like_api_urls) else {}
API_URLS_FOR_ABI = {x:API_URLS[x]+API_PARAMS_FOR_ABI for x in API_URLS}
API_URLS_FOR_ERC20_TRANSFER = {x:API_URLS[x]+API_PARAMS_FOR_ERC20_TRANSFER for x in API_URLS}
API_URLS_FOR_INTERNAL_TX  = {x:API_URLS[x]+API_PARAMS_FOR_INTERNAL_TX for x in API_URLS}
API_URLS_FOR_NORMAL_TX  = {x:API_URLS[x]+API_PARAMS_FOR_NORMAL_TX for x in API_URLS}

API_KEYS = json.load(open(C.paths().etherscan_like_api_keys, 'r')) if os.path.exists(C.paths().proxy) else {}

API_PARAMS_GET_BLOCKNO_BY_TIMESTAMP = "?module=block&action=getblocknobytime&timestamp={timestamp}&closest=after&apikey={apikey}"
API_URLS_FOR_START_BLOCKNO = {x:API_URLS[x]+API_PARAMS_GET_BLOCKNO_BY_TIMESTAMP for x in API_URLS}

# GLOBAL_PROXY = json.load(open(C.paths().proxy, 'r')) if os.path.exists(C.paths().proxy) else {}
GLOBAL_PROXY = None 

DESTRUCTORS = {} # for some globals to save to file
DESTRUCTORS_CONTROL_TABLE = set() # only those desctructors recorded by this table will be executed

def load_global_as_need(name:str, callback:T.Callable[[], T.Any], destructor:T.Callable[[],None]=None):
    """
        Some global vars need a long time to load (from file), 
        yet was not always used. 
        Load them when needed
    """
    def dump_global_to_file(global_name:str, filename:str):
        f = open(filename, 'w') 
        json.dump(globals()[global_name], f, indent=1)
        f.close()

    r = globals().get(name, None)
    if r is None: 
        r = callback()
        globals()[name] = r 
        if destructor is not None:
            if isinstance(destructor, str):
                destructor = partial(dump_global_to_file, name, destructor)
            if name not in DESTRUCTORS:
                DESTRUCTORS[name] = destructor
    return r


def num_tokens_from_string(string: str, encoding_name: str = 'o200k_base') -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def load_global_from_file(name:str, filename:str, _default=None, destructor=None):
    """
        A wrapper for calling `load_global_as_need`
    """
    if not os.path.exists(filename):
        return _default
    ret = load_global_as_need(name, lambda: json.load(open(filename)), destructor)
    return ret 


def at_exit():
    for name, func in DESTRUCTORS.items():
        if name in DESTRUCTORS_CONTROL_TABLE:
            func()
    DESTRUCTORS_CONTROL_TABLE.clear()

def load_all_data_by_chain(chain: str, 
                with_log:bool=False) -> T.Union[T.List[T.Dict], T.Dict[str, T.List[T.Dict]]]:
    """
        - @param `chain`: 
        - @param `with_log`: if true, return transactions with the corresponding log
        - @return if `with_log` is false, return List of transactions of one chain, list element is a dict.
                else, return list of transactions, together with a mapping: the trx hash => the corresponding logs(a list)
    """
    dir_path = os.path.join(C.paths().bridge_tx_data, chain)
    walk = os.walk(dir_path)
    all_data_list = []
    loaded_hash = set()
    if with_log:
        f = os.path.join(C.paths().bridge_log_data, chain + '.json')
        logs_obj = json.load(open(f))

    for path, dir_list, file_list in walk:
        for file in file_list:
            file = os.path.join(path, file)
            with open(file, 'r') as f:
                data = json.load(f)
            for d in data: 
                if d['hash'] not in loaded_hash:
                    all_data_list.append(d)
                    loaded_hash.add(d['hash'])
    
    # if len(all_data_list) > 7000: 
    #     all_data_list = all_data_list[5000:7000]
    if with_log:
        logs_dict:T.Dict[str, T.List[T.Dict]] = {}
        logs_dict_hash = {}
        for log in logs_obj:
            th = log['transactionHash']
            if th not in logs_dict: 
                logs_dict[th] = []
                logs_dict_hash[th] = set()
            s = json.dumps(log, sort_keys=True)
            if s not in logs_dict_hash[th]:
                logs_dict[th].append(log)
                logs_dict_hash[th].add(s)
        return all_data_list, logs_dict
    else:
        return all_data_list


def get_url(target_url:str, method:str='get', data=None,headers=None, proxy:dict=None, _json=None):
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=3)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    if C.force_proxy and proxy is None: 
        proxy = GLOBAL_PROXY
    retry_times = 3
    while retry_times > 0:
        try:
            if method == 'get':
                resp = session.get(target_url, proxies=proxy,headers=headers)
            elif method == 'post':
                resp = session.post(target_url, data=data, json=_json, proxies=proxy,headers=headers)
            else:
                resp = ''
            break
        except Exception as e:
            logging.exception(e)
            retry_times -= 1
            if retry_times <= 0:
                raise e
    return resp.text

def _download_abi(addr:str, chain:str)->T.Tuple[int, str]:
    resp = get_url(API_URLS_FOR_ABI[chain].format(
        addr = addr,
        apikey = random.choice(API_KEYS[chain])
    ), proxy=GLOBAL_PROXY)
    abi = json.loads(resp)
    abi = ed(abi)
    if abi.status == '0':
        return 0, ""
    if abi.status != '1':
        return -1, ""
    return 1, str(abi.result)

def download_abi(addr:str, chain:str)->str:
    status, abi = _download_abi(addr, chain)
    try_times = 3 # 失败重试3次
    while status == -1 and try_times > 0:
        time.sleep(5-try_times)
        status, abi = _download_abi(addr, chain)
        try_times -= 1
    return status, abi

def _encode_abi_to_signature(abi:str) -> dict:
    """
        given abi, return all signatures it contains
    """
    def _encode_one_type_abi(abi_input, only_4_bytes=False):
        sig_to_text = {}
        for abi_entry in abi_input:
            # a = "{name}({args})".format(name=abi_entry['name'], args=','.join(t['type'] for t in abi_entry['inputs']))
            # sig = (Web3.keccak(text=a)).hex()
            a = eth_utils.abi._abi_to_signature(abi_entry)
            sig = (Web3.keccak(text=a)).hex()
            if only_4_bytes:
                sig = sig[:10] # 4bytes * 2 + 2(=len("0x"))
            b = "{name}({args})".format(
                name=abi_entry['name'], 
                args=','.join(("%s %s%s" % (t['type'], 'indexed ' if t.get('indexed', False) else '',t['name'])) for t in abi_entry['inputs']))
            sig_to_text[sig]  = {'text': b, 'full': json.dumps(abi_entry)}
        return sig_to_text

    # web3.Web3.keccak(text="")
    abi_obj:T.List[T.Dict[str, T.Any]] = json.loads(abi)
    func_sigs = _encode_one_type_abi((_obj for _obj in abi_obj if _obj['type'] == 'function'), True)
    event_sigs= _encode_one_type_abi(_obj for _obj in abi_obj if _obj['type'] == 'event')
    # print(abi_obj)
    return func_sigs, event_sigs


def update_signature_db_with_argname(sigs:dict, db_path:str, db_name:str):
    SIG_DB = load_global_from_file(db_name, db_path, {}, db_path)
    flag_sig_db_changed_all = False
    for sig, d in sigs.items():
        text_sig_with_name = d['text']
        full_abi_of_sig = d['full']
        flag_sig_db_change_local = True
        tmp_abi = parse_funcname_to_abi(text_sig_with_name)
        text_sig_without_name = "{name}({args})".format(
            name=tmp_abi['name'], 
            args=(','.join(t['type'] for t in tmp_abi['inputs']) if len(tmp_abi['inputs']) else ''))
        if sig in SIG_DB:
            if len(SIG_DB[sig]) > 1:
                chosen_index = -1 
                for i, t in enumerate(SIG_DB[sig]):
                    if t['text_signature'] == text_sig_without_name: 
                        chosen_index = i 
                        break
                if chosen_index == -1:
                    SIG_DB[sig].append({
                        'text_signature_name':text_sig_with_name, 
                        "full_abi":full_abi_of_sig, 
                        'text_signature':text_sig_without_name,
                        "hex_signature":sig
                    })
                    # assert 0, "index is -1, multi signature found:" + sig # (TODO)
                else:
                    SIG_DB[sig][chosen_index]['text_signature_name'] = text_sig_with_name
                    SIG_DB[sig][chosen_index]['full_abi'] = full_abi_of_sig
            elif text_sig_with_name != SIG_DB[sig][0]['text_signature']:
                # Only when the text_sig != text_sig_with_arg_name, write the result 
                SIG_DB[sig][0]['text_signature_name'] = text_sig_with_name
                SIG_DB[sig][0]['full_abi'] = full_abi_of_sig
            else:
                flag_sig_db_change_local = False
        else:
            SIG_DB[sig] = [{"text_signature_name":text_sig_with_name, 'text_signature':text_sig_without_name, "full_abi":full_abi_of_sig, }]
        flag_sig_db_changed_all = flag_sig_db_changed_all or flag_sig_db_change_local
    if flag_sig_db_changed_all:
        DESTRUCTORS_CONTROL_TABLE.add(db_name)

def get_abi(addr:str, chain:str, fallback_sig:str='', sig_type:str='')->str:
    """
        @param `fallback_sig`: if the whole ABI of `addr` was not found, 
            then return the constructed abi of signature(if can be found in database)
        @param `sig_type`: if `fallback_sig` is not empty, this must be "function" or "event"
        @return abi of addr on chain
    """ 
    ABI_DATABASE = load_global_from_file('ABI_DATABASE', C.paths().abis, {}, C.paths().abis)
    addr = addr.lower()
    if chain not in ABI_DATABASE:
        ABI_DATABASE[chain] = {}
    if addr in ABI_DATABASE[chain] and len(ABI_DATABASE[chain][addr]):
        return ABI_DATABASE[chain][addr]
    else:
        if (addr not in ABI_DATABASE[chain]) and (addr != ''):
            _download_status, abi = download_abi(addr, chain)
            if _download_status == 0: 
                # the addr does not contain ABI
                ABI_DATABASE[chain][addr] = ''
                DESTRUCTORS_CONTROL_TABLE.add('ABI_DATABASE')
        else:
            abi = ''
        if abi and len(abi):
            ABI_DATABASE[chain][addr] = abi
            func_sigs, event_sigs = _encode_abi_to_signature(abi)
            update_signature_db_with_argname(func_sigs, C.paths().signatures, 'SIG_DB')
            update_signature_db_with_argname(event_sigs, C.paths().event_signatures, 'EVENT_SIG_DB')
            DESTRUCTORS_CONTROL_TABLE.add('ABI_DATABASE')
            return abi
        elif len(fallback_sig):
            if sig_type == 'function': 
                SIG_DB = load_global_from_file('SIG_DB', C.paths().signatures,{}, C.paths().signatures)
            elif sig_type == 'event': 
                SIG_DB = load_global_from_file('EVENT_SIG_DB', C.paths().event_signatures,{}, C.paths().event_signatures)
            else:
                assert 0, "Error sig_type:" + str(sig_type) 
            if fallback_sig in SIG_DB:
                records = SIG_DB[fallback_sig]
                if 'full_abi' in records[-1]:
                    return records[-1]['full_abi']
                if 'text_signature_name' in records[-1]: # choose the last one, since we add the ABI-sourced sigs to the end of list 
                    abi = parse_funcname_to_abi(records[-1]['text_signature_name'], True, sig_type)
                else:
                    abi = parse_funcname_to_abi(records[-1]['text_signature'], False, sig_type)
                return json.dumps([abi])
        else:
            return ''


def check_rpc_endpoint_validity(w3):
    try:
        n = w3.eth.block_number
        assert isinstance(n, int)
        return True
    except Exception as e:
        logging.info('RPC %s check failed: %s' % (w3.provider.endpoint_uri, str(e)))
        return False

def _get_token_name_symbol_decimals_from_local_cache(addr:str, chain:str)->T.Tuple[str, str, int]:
    # chain, address => { 'name'/'symbol' => ${value} }
    TOKEN_DATABASE:T.Dict[str, 
        T.Dict[str,T.Dict[str, str]]] = load_global_from_file(
            'TOKEN_DATABASE', 
            C.paths().tokens, {}, C.paths().tokens)
    if chain in TOKEN_DATABASE:
        if addr in TOKEN_DATABASE[chain]:
            token_info = TOKEN_DATABASE[chain][addr]
            return token_info['name'], token_info['symbol'], token_info['decimals']
    return "", "", 0

def _get_token_name_symbol_decimals_from_web(addr:str, chain:str, w3:Web3)->T.Tuple[str, str, int]:
    """
    get abi from local handmade token abi, other than from get_abi(addr, chain)
    because we only need name/symbol/decimals
    what's more, some token is proxy contract whose abi contains nothing about name/symbol/decimals
    """
    HANDMADE_TOKEN_ABI = load_global_from_file('HANDMADE_TOKEN_ABI', C.paths().handmade_token_abi, '') 
    abi = HANDMADE_TOKEN_ABI 
    try:
        contract = w3.eth.contract(Web3.to_checksum_address(addr), abi=abi)
        name = contract.functions.name().call()
        symbol = contract.functions.symbol().call()
        decimals = contract.functions.decimals().call()
    except Exception as e:
        logging.error('get token from web fail, may meet unknown token, token: %s, chain: %s' % (addr, chain))
        logging.exception(e)
        name = ''
        symbol = ''
        decimals = -1
    finally:
        return name, symbol, decimals

def get_token_name_symbol_decimals(addr:str, chain:str, w3:Web3=None)->T.Tuple[str, str, int]:
    '''
        @return token name, token symbol and decimals
    '''
    TOKEN_DATABASE = load_global_from_file(
        'TOKEN_DATABASE', 
        C.paths().tokens, {}, 
        C.paths().tokens
    )
    if not addr or addr.lower() == 'eth': 
        return 'eth', 'eth', 18
    addr = normalize_address_type(addr.lower())
    if save_value_int(addr) == 0: 
        return 'eth', 'eth', 18
    name, symbol, decimals = _get_token_name_symbol_decimals_from_local_cache(addr, chain)
    if decimals == -1 and not C.force_reload: 
        # 如果是-1 说明之前已经查过这个Token，它不是Token, 不用浪费时间查了
        return "", "", 0
    if not len(name) and (decimals == 0 or (decimals == -1 and C.force_reload)):
        # decimal 是0, 不在DB里. 查询; 或者是-1，代表之前查过，但是强制重新加载
        name, symbol, decimals = _get_token_name_symbol_decimals_from_web(addr, chain, w3)
        token_info = {
            'name': name ,
            'symbol': symbol,
            'decimals': decimals
        }
        if not chain in TOKEN_DATABASE:
            TOKEN_DATABASE[chain] = {}
        if decimals not in (0, ):
            TOKEN_DATABASE[chain][addr] = token_info
        DESTRUCTORS_CONTROL_TABLE.add('TOKEN_DATABASE')
    return name, symbol, decimals

def strip_underline_of_key(d:dict):
    """
        给定一个dict, 将其所有key中的下划线去掉
        e.g. {"_amount":1}  ==> {"amount":1}
    """
    ret = {}
    for key in d: 
        if not isinstance(key, str): continue
        new_key = key.strip("_")
        ret[new_key] = d[key]
    return ret

def str_is_address_type(string:str):
    """
        Given a string, return whether it's address type (Ethereum-like style)
    """
    return (isinstance(string, str) and 
            re.match(r'''^(0[xX])?[0-9a-fA-F]{40}$''', string) is not None)

def guess_para_type(para):
    """
        Given a parameter, guess its type .
        Can be one of: 
        address ; integer ; string
    """
    if isinstance(para, str): 
        if str_is_address_type(para): 
            return 'address'
        else:
            return 'string'
    elif 'bool' in str(type(para)):
        return 'boolean'
    elif isinstance(para, int):
        return 'integer'
    else:
        assert 0, type(para)

PLACEHOLDER = 0
def remove_non_ascii_values(data:dict):
    global PLACEHOLDER
    ret = {}
    for key in data:
        if isinstance(data[key], dict):
            ret[key] = remove_non_ascii_values(data[key])
        elif not isinstance(data[key], str):
            ret[key] = data[key]
        elif all((c in string.printable) for c in data[key]):
            ret[key] = data[key]
        else:
            ret[key] = byte_to_hex(data[key], True)
            if len(ret[key]) > 100: 
                ret[key] = 'placeholder' + str(PLACEHOLDER)
                PLACEHOLDER += 1
    return ret

def byte_to_hex(s:str, add_prefix=False):
    if isinstance(s, str):
        ret = binascii.b2a_hex(s.encode('latin1')).decode('latin1')
    elif isinstance(s, bytes):
        ret = binascii.b2a_hex(s).decode('latin1')
    else:
        assert 0, type(s)
    if add_prefix:
        return '0x' + ret
    return ret

def get_erc20_transfer_events(chain:str, addr:str, start_block:int, 
                              end_block:int, contract_addr:str=None):
    """
        Given chain name and address, return its erc20 transfer address.
        if `contract_addr` is specified, then the token contract address is also used in filter.
        `current_block` specifies the current block number of instance and is used when filtering the range of start/end block
    """
    page = 0
    ret = []
    if start_block > end_block: 
        start_block, end_block = end_block, start_block
    while True:
        page += 1
        url = API_URLS_FOR_ERC20_TRANSFER[chain].format(
            addr = addr,
            apikey = random.choice(API_KEYS[chain]),
            page=page, 
            offset=100,
            startblock=start_block,
            endblock=end_block
        )
        if contract_addr is not None:
            url += '&contractaddress=' + str(contract_addr)
        resp = get_url(url)
        try:
            resp = json.loads(resp)
        except json.JSONDecodeError:
            page -= 1
            time.sleep(1)
            continue
        if resp.get('status') == '0' and resp.get('message') == 'No transactions found':
            break
        if resp.get('status') != '1' or resp.get('message') != 'OK' : 
            logging.warning("unexpected status in `get_erc20_transfer_events`:" + str(resp))
            break
        if not len(resp['result']): 
            break
        ret.extend(resp['result'])
        time.sleep(0.5)
    return ret


def get_transactions(chain:str, address:str, start_block:int, end_block:int, tx_type:str = 'internal'):
    """
        return the internal transactions of given `address` within the range of `start_block` and `end_block`
    """
    page = 0
    ret = []
    if tx_type == 'internal': 
        api_url = API_URLS_FOR_INTERNAL_TX[chain]
    else:
        api_url = API_URLS_FOR_NORMAL_TX[chain]
    while True:
        page += 1
        url = api_url.format(
            addr=address, startblock=start_block, endblock=end_block, 
            apikey=random.choice(API_KEYS[chain]), page=page, offset=100)
        resp = get_url(url)
        try:
            obj = json.loads(resp)
            obj = ed(obj)
        except json.JSONDecodeError:
            page -= 1
            time.sleep(1)
            continue
        if obj.get('status') == '0':
            break
        if obj.status != "1" or obj.message != 'OK':
            logging.warning("unexpected status in `get_internal_txns`:" + str(resp))
            break
        if not len(obj.result):
            break
        ret.extend(obj.result)
        time.sleep(0.5)
    return ret


def get_block_num_by_timestamp(timestamp:int, chain:str):
    """
        return the closest block number with given `timestamp`
    """
    url = API_URLS_FOR_START_BLOCKNO[chain].format(timestamp=timestamp, apikey=random.choice(API_KEYS[chain]))
    resp = get_url(url)
    obj = json.loads(resp)
    obj = ed(obj)
    if obj.status != "1":
        return -1
    if not len(obj.result):
        return 0
    return int(obj.result)


def parse_funcname_to_abi(funcname:str, 
            with_argument_name:bool=True, 
            given_type='function',
            is_tuple:bool=False):
    """
        Given function name, parse it to abi mode.
        - @param `is_tuple`: if true, then the `funcname` should be like "(uint,address,uint)", 
                that is, no prefix function name given 
    """
    left_brackets = []
    tuple_comp = None
    for i, c in enumerate(funcname):
        if c == '(':
            left_brackets.append(i)
        elif c == ')':
            last_left_index = left_brackets.pop()
            if len(left_brackets) == 1:
                content = funcname[last_left_index:i+1]
                tuple_comp = parse_funcname_to_abi(content, with_argument_name, given_type, True)
                funcname = funcname[:last_left_index] + "tuple" +funcname[i+1:]
                break

    ret = {}
    if is_tuple:
        if with_argument_name:
            pat = r'''\((?P<args>(\S+ (indexed )?\S*,?)+)?\)''' 
        else:
            pat = r'''\((?P<args>(\S+,?)+)?\)'''
    elif with_argument_name:
        pat = r'''(?P<name>\S+)\((?P<args>(\S+ (indexed )?\S*,?)+)?\)'''
    else:
        pat = r'''(?P<name>\S+)\((?P<args>(\S+,?)+)?\)'''
    # funcname: swap(uint16,uint256,uint256,address,uint256,uint256,(uint256,uint256,bytes),bytes,bytes)
    obj = re.search(pat, funcname)
    if obj is None:
        return ret
    if not is_tuple:
        name = obj.group('name')
        ret['name'] = name
    args = obj.group('args')
    args = args.split(',') if args is not None else []
    input_args = []
    for i, arg in enumerate(args):
        arg = arg.strip(' ')
        if 'indexed ' in arg: 
            flag_indexed = True
            arg = arg.replace('indexed ', '')
        else:
            flag_indexed = False
        flag_arg_has_name = len(arg.split(' ')) > 1
        current_arg_type = arg.split(' ')[0]
        if current_arg_type == 'tuple' and tuple_comp is not None:
            t = tuple_comp
        else:
            t = {'internalType': current_arg_type, 'name': arg.split(' ')[1] if (with_argument_name and flag_arg_has_name) else "arg" + str(i), 'type': current_arg_type}
        if 'event' == given_type:
            t['indexed'] = flag_indexed
        input_args.append(t)
    
    if not is_tuple:
        ret['inputs'] = input_args
        ret['outputs'] = []
        ret['type'] = given_type
        if given_type == 'event':
            ret['anonymous'] = False
        ret['stateMutability'] = 'payable'
        return ret 
    else:
        ret['type'] = 'tuple'
        ret['name'] = ''
        ret["internalType"] = "struct Types.X"
        ret["components"] = input_args
        return ret


def get_rpc_endpoints(chain:str='', choose_one:bool=True, return_w3_obj:bool=False):
    g = globals()
    rpc_endpoints = g.get('rpc_endpoints', None)
    if rpc_endpoints is None:
        rpc_endpoints = json.load(open(C.paths().rpc_endpoints)) if os.path.exists(C.paths().rpc_endpoints) else {}
        g['rpc_endpoints'] = rpc_endpoints
    if not len(chain): 
        return rpc_endpoints
    else:
        if choose_one:
            ret = random.choice(rpc_endpoints[chain])
            if return_w3_obj: return ret 
            return ret.provider.endpoint_uri
        else:
            if return_w3_obj: return rpc_endpoints[chain]
            else:
                return list(map(lambda x: x.provider.endpoint_uri, rpc_endpoints[chain]))


def _remove_wrong_answers(trx:dict, logs:T.List[dict]):
    """
        Some keys and values are too long and seems impossible to be the correct answer.
        E.g., every event log has fields `transactionHash`,`blockHash`,`blockNumber`, 
        they are identical to each other when logs came from same transaction.
    """
    if 'input' in trx:
        del trx['input'] # The 'input' field was decoded into "args"
    if 'chainId' in trx:
        del trx['chainId']
    trx['args'] = remove_non_ascii_values(trx['args'])
    for log in logs:
        if not len(log): continue
        try:
            del log['transactionHash'], log['blockHash'], log['blockNumber']
        except : 
            pass 
        log['args'] = remove_non_ascii_values(log['args'])
    return trx, logs


def _handle_instance_to_ask_LLM(trx, logs, wrap_with_json:bool=False)->str:
    """
        replace long strings and convert them to prompt
    """
    trx, logs = _remove_wrong_answers(trx, logs)
    a = json.dumps(trx, cls=BytesDecoder)
    if wrap_with_json:
        a = '\n```json\n%s```\n' % a
    trx_str = "transaction: %s\n" % (a)
    logs_str = ''
    for i, log in enumerate(logs):
        if not len(log): continue
        a = json.dumps(log, cls=BytesDecoder)
        if wrap_with_json:
            a = '\n```json\n%s```\n' % a
        logs_str += "log[%s]:%s\n" % (log['event'], a)
    return trx_str + logs_str


def extract_values_by_given_fields(tx:T.Dict[str, T.Any], log:T.List[T.Dict[str, T.Any]], 
                        rules:T.Dict[str, T.List[str]], strict_mode:bool=False):

    def BFS(initial_element, extend_callback, confirm_callback):
        queue = deque([initial_element])
        
        while len(queue): 
            head = queue.popleft()
            if confirm_callback(head): 
                return head
            for ch in extend_callback(head):
                queue.append(ch)
        return None

    def _get_dict_value_ignore_case(d:dict, target_k:str):
        for k in d: 
            lk = k.lower()
            if lk == target_k.lower() : return d[k]

    def _extract(path:str):
        ps = path.split('.')
        if ps[0] == 'transaction': obj = tx 
        elif ps[0].startswith('log'): 
            idx:str = (ps[0].replace('log', ''))
            if idx.isdigit():
                obj = log[int(idx)]
            else:
                # 可以按index选择log，也可以按照event名字选择log
                # 如果有2个以上的方括号，就代表有多个同名log
                cnt_of_bracket = idx.count('[')
                index_of_log_wanted = -1
                if cnt_of_bracket == 2:
                    _split = (idx.find(']['))
                    part1, part2 = idx[:_split], idx[_split+2:]
                    idx = part1.replace('[', '').replace(']', '')
                    index_of_log_wanted = int(part2.replace('[', '').replace(']', ''))
                elif cnt_of_bracket == 1:
                    idx = idx.replace('[', '').replace(']', '')
                    index_of_log_wanted = 1
                else:
                    assert 0, f"Extarcting values from {idx} Error"
                obj = None
                i = 0
                for _log in log:
                    if 'event' not in _log : continue
                    if _log['event'] == idx:
                        i +=1
                        if i == index_of_log_wanted:
                            obj = _log 
                            break
        else:
            # 如果规则既没有指定transaction也没有指定log，就直接返回预设值
            return ps[0] if not strict_mode else None
        if obj is None:
            return None
        cur = ed(obj)
        for p in ps[1:]:
            nextcur = cur.get(p, None) or _get_dict_value_ignore_case(cur, p)
            if nextcur is None: 
                # 如果路径唯一，则遍历找到它
                traversed = BFS(cur, lambda x: x.values() if isinstance(x, dict) else [], lambda x: isinstance(x, dict) and p in x)
                cur = traversed[p] if traversed else None
            else:
                cur = nextcur
        return cur

    ret = ed({})
    r = ed(rules)
    for key in r: 
        if isinstance(r[key], list):
            for value in r[key]:
                try:
                    extracted_fields = _extract(value)
                except:
                    ret[key] = extracted_fields = None
                    break
                if extracted_fields is not None: 
                    ret[key] = extracted_fields 
                    break
        elif isinstance(r[key], dict):
            # 如果是一个dict，则该dict的每一个键是函数名，每一个值是对应的规则
            fn = tx['pureFunctionName']
            cur_rules = r[key][fn]
            for cr in cur_rules:
                try:
                    extracted_fields = _extract(cr)
                except:
                    ret[key] = extracted_fields = None
                    break
                if extracted_fields is not None:
                    ret[key] = extracted_fields
                    break
        else:
            assert 0, "Error when extracting fields by rule"
    return ret

def pure_json_object(content:str, return_str:bool=False, fast_judge:bool=True):
    """
        Sometimes the LLM return text that cannot be converted to json directly.
        Remove all unused sentences other than json object.
    """
    if fast_judge: 
        content = content.replace('```json', '').replace('```', '')
    try: 
        ret = json.loads(content)
        if return_str: 
            return content
        return ret
    except json.JSONDecodeError:
        ret = ''
        left_bracket, brck_found = 0, False
        for c in content:
            if not brck_found and c == '{':
                brck_found = True
            if not brck_found : continue
            if c == '{':
                left_bracket += 1 
            elif c == '}': 
                left_bracket -= 1
            ret += c
            if left_bracket == 0:
                if return_str: return ret 
                else: return json.loads(ret)

def md5_encryption(data):
    md5 = hashlib.md5()  # 创建一个md5对象
    md5.update(data.encode('utf-8'))  # 使用utf-8编码数据
    return md5.hexdigest()  # 返回加密后的十六进制字符串

def save_value_int(v):
    if isinstance(v, str):
        if not len(v): return 0
        if v.startswith('0x'): return int(v, base=16)
        try:
            return int(v)
        except:
            try: return int(v, base=16)
            except: return 0
    if not v: return 0
    return v

def save_div(a, b):
    if b == 0:
        return float("nan")
    return a / b


def hex_pad(i:int, no_prefix:bool=False):
    if not isinstance(i, int): 
        raise TypeError(type(i))
    ret = hex(i)
    if no_prefix:
        ret = ret[2:]
    if len(ret) % 2 != 0:
        if ret.startswith('0x'): 
            ret = '0x0' + ret[2:]
        else:
            ret = '0' + ret
    return ret


def normalize_address_type(addr:str):
    """
        normalize a string type to EVM-like address type. 
        i.e., a string that starts with '0x' and followed by 20 bytes (40 length in hex-format)
    """
    ret = addr
    if not isinstance(addr, str) or not len(addr): 
        return '0x' + ('0' * 40)
    if len(addr) < 42:
        ret = hex_pad(int(addr, base=16), True)
        ret = '0x' + ret.rjust(40, '0')
    return ret

def mid_for(L:list):
    """
        traverse a list from the medium index to its two sides
    """
    if len(L) <= 3: 
        for t in L: yield t
        return 
    s = len(L) // 2
    p ,q = s-1 , s
    while p >=0 and q < len(L):
        yield L[p]
        p -= 1
        yield L[q]
        q += 1
    while p >=0 : 
        yield L[p]
        p -= 1
    while q < len(L):
        yield L[q]
        q += 1 

def edit_distance(s1, s2):
    """
    Calculates the Levenshtein (edit) distance between two strings.
    Args:
        s1: The first string.
        s2: The second string.
    Returns:
        The edit distance between s1 and s2.
    """
    n = len(s1)
    m = len(s2)
    # Create a matrix to store distances between prefixes of the strings
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    # Initialize first row and column
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    # Populate the matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],      # Deletion
                                   dp[i][j - 1],      # Insertion
                                   dp[i - 1][j - 1])   # Substitution

    return dp[n][m]


def check_whether_token_name_and_symbol_match(src_name, src_symbol, dst_name, dst_symbol, strict_mode=False):
    if dst_name.lower() == src_name.lower() or dst_symbol.lower() == src_symbol.lower(): 
        return True
    if strict_mode: 
        return False 
    if src_symbol in DOC.STABLE_COIN_SYMBOLS and dst_symbol in DOC.STABLE_COIN_SYMBOLS: 
        return True
    if src_symbol in DOC.ETH_TOKEN_SYMBOLS and dst_symbol in DOC.ETH_TOKEN_SYMBOLS: 
        return True
    dis = edit_distance(src_symbol, dst_symbol)
    if dis <= 1 or save_div(dis , len(src_symbol)) <= 0.1 or save_div(dis, len(dst_symbol)) <= 0.1: 
        return True
    return False


def defaultdict_to_dict(d):
    """
    将 defaultdict 转换为普通的 dict。
    
    :param d: 要转换的 defaultdict
    :return: 转换后的普通 dict
    """
    if isinstance(d, dd):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


def copy_dict_to_nested_defaultdict(d1: dict, d2: dd) -> None:
    """
    将字典 d1 中的所有内容复制到可能多层嵌套的 defaultdict d2 中，并且不影响 d2 的 defaultdict 功能。

    Args:
        d1: 要复制的普通字典。
        d2: 要复制到的 defaultdict。
    """
    for key, value in d1.items():
        if isinstance(value, dict):
            # 如果值是字典，则递归调用自身来处理嵌套
            copy_dict_to_nested_defaultdict(value, d2[key])
        else:
            # 如果值不是字典，则直接赋值
            d2[key] = value


def set_args_from_yaml_config(args):
    if args.config: 
        with open(os.path.join(C.paths().running_config, args.config + '.yaml')) as f:
            yaml_config = yaml.safe_load(f)
            for key, value in yaml_config['override_config'].items():
                C.set_globals(key, value)
            C.reset_paths()
            for key, value in yaml_config['unoverride_args'].items():
                if not hasattr(args, key) or not getattr(args, key):
                    setattr(args, key, value)
            for key, value in yaml_config['override_args'].items():
                setattr(args, key, value)
    else:
        yaml_config = {}

    return yaml_config

def group_strings_by_hierarchy(array: T.List[str]) -> str:
    """
    对于输入的字符串数组array: List[str]，使用小数点"."分隔每个字符串，然后按照层次结构将它们组合起来。
    组合的例子：
        (1) ['log[Approval].owner', 'log[Approval].spender'] => 'log[Approval](owner, spender)'
        (2) ['log[CreatedOrder].order.takeAmount', 'log[CreatedOrder].order.giveChainId'] => 'log[CreatedOrder](order(takeAmount,giveChainId))'
        (3)额外要求：如果某个字符串在分隔后，某个部分是"args"，则这个部分应该被忽略，但字符串的其他部分仍然按照上述规则处理。
            例如 ['log[Approval].args.owner', 'log[Approval].args.spender'] => 'log[Approval](owner, spender)'
    """
    def build_tree(parts):
        if not parts:
            return ""
        root = parts[0]
        if len(parts) == 1:
            return root
        sub_parts = parts[1:]

        if not sub_parts:
            return root

        if sub_parts[0] == "args":
            sub_parts = sub_parts[1:]
        if not sub_parts:
            return root
        return root, build_tree(sub_parts)

    def merge_trees(trees: T.List[tuple]) -> str:

        if not trees:
           return ""

        if len(trees) == 1 :
           if isinstance(trees[0],str):
                return trees[0]
           else :
               return  f"{trees[0][0]}({merge_trees([trees[0][1]])})"

        merged_result = {}

        for tree in trees:
            if isinstance(tree,str):
               if tree not in merged_result:
                  merged_result[tree] = None
               continue
            
            root, subtree = tree
           
            if root not in merged_result:
                merged_result[root] = []
            if subtree is not None:
              merged_result[root].append(subtree)


        result_str = ""
        for root, subtrees in merged_result.items():
            if subtrees is None :
                 result_str += f"{root}," 
            elif not subtrees:
               result_str += f"{root},"
            else:
               result_str += f"{root}({merge_trees(subtrees)}),"
        
        return result_str.rstrip(",")

    trees = []
    for s in array:
        parts = s.split(".")
        trees.append(build_tree(parts))
    
    return merge_trees(trees)

def find_n_keys_with_max_values(d: T.Dict[T.Any, int], n: int) -> T.List[T.Any]:
    """
    查找字典中具有最大值的 n 个键。
    Args:
        d: 字典，其中键可以是任何类型，值是整数。
        n: 要查找的具有最大值的键的数量。
    Returns:
        具有最大值的 n 个键的列表。
    """
    if n <= 0:
      return []
    # 将字典的项（键值对）转换为列表，并按值降序排序
    sorted_items = sorted(d.items(), key=lambda item: item[1], reverse=True)
    # 从排序后的列表中提取前 n 个键
    top_n_keys = [item[0] for item in sorted_items[:min(n, len(sorted_items))]]    
    return top_n_keys


def ask_LLM(prompt, provider:str, N:int=1):
    """
    Sends a text prompt to the LLM and returns the generated response.
    :param prompt: String containing the prompt text for the LLM.
    :return: Generated text response from the LLM.
    """

    def read_from_yaml(file_path):
        with open(file_path, 'r') as file:
            return ed(yaml.safe_load(file))

    def read_from_yaml_wrap(config_name:str):
        config_cache = globals().get('LLM_config_cache', None)
        if config_cache is not None and config_name in config_cache:
            return config_cache[config_name]
        else:
            config = read_from_yaml(os.path.join(C.paths().LLM_config_dir, config_name+'.yaml'))
            if config_cache is None:
                config_cache = {}
            config_cache[config_name] = config
            globals()['LLM_config_cache'] = config_cache
            return config

    def init_client(config):
        # config = read_from_yaml_wrap(provider)  # Read the configuration from YAML
        api_config = config.get('api', {})
        client = openai.OpenAI(
            # http_client=httpx.Client(proxies=proxy['http']),
            base_url=api_config.get('endpoint', ''),
            api_key=api_config.get('key', '')
        )
        return client


    def get_client(config, client_name:str=''):
        client_name = client_name or provider
        clients = globals().get('LLM_clients', None)
        if clients is not None:
            client = clients.get(client_name, None)
            if client is not None:
                return client 
        else:
            clients = {}
        clients[client_name] = init_client(config)
        globals()['LLM_clients'] = clients
        return clients[client_name]

    def _wrap_ask(client, config):
        return client.chat.completions.create(
            model= config.model.id,
            temperature=config.model.temperature,
            max_tokens=config.model.max_tokens, 
            messages=prompt,
            n=N,
        )

    config = read_from_yaml_wrap(provider) 
    client = get_client(config, config.api.get('name', ''))
    # OpenAI API call to interact with the LLM
    try:
        response = _wrap_ask(client, config)
        if not len(response.choices[0].message.content): 
            assert 0, "Empty response from LLM"
    except :
        error = True
        for fallback_config in config.get('fallback_configs', []) + [config]:
            time.sleep(1)
            try:
                client = get_client(fallback_config, fallback_config.api.get('name', ''))
                response = _wrap_ask(client, fallback_config)
                if not len(response.choices[0].message.content): 
                    assert 0, "Empty response from LLM"
                error = False
                break
            except Exception as e:
                logging.error(str(e))
                continue
        if error:
            print('fallback_configs:')
            print(config.get('fallback_configs', []))
            raise Exception("All fallback models failed")

    return response

class BytesDecoder(json.JSONEncoder):
    def default(self, data):
        if isinstance(data, hexbytes.main.HexBytes):
            return data.hex()
        if isinstance(data, bytes):
            return data.decode('latin1')
        if isinstance(data, tuple):
            return str(data)
        return json.JSONEncoder.default(self, data)

class RPCHandler:
    def __init__(self, chains_to_check=None, check_switch:bool=True) -> None:
        g = globals()
        rpc_endpoints = g.get('rpc_endpoints', None)
        if rpc_endpoints:
            self.rpcs = rpc_endpoints
            return
        rpc_endpoints = json.load(open(C.paths().rpc_endpoints)) if os.path.exists(C.paths().rpc_endpoints) else {}
        self.rpcs = {}
        if not chains_to_check: 
            chains_to_check = C.chains_to_pair
        for chain in chains_to_check:
            self.rpcs[chain] = []
            for url in rpc_endpoints[chain]:
                w3 = Web3(Web3.HTTPProvider(url))
                if check_switch and not check_rpc_endpoint_validity(w3): continue
                self.rpcs[chain].append(w3)
        g['rpc_endpoints'] = self.rpcs

    def choose_one_rpc(self, chain: str) -> Web3:
        """
            choose (randomly) from the prepared rec endpoints in order not to 
            initiate an instance every time when called
        """
        return random.choice(self.rpcs[chain])