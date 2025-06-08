# -*- coding:utf-8 -*-

import typing as T
import random
import json
import os
import re
import logging, copy
import binascii
from easydict import EasyDict as ed
from collections import defaultdict as dd
# Web3 
import web3
from web3 import Web3
from hexbytes import HexBytes
from eth_utils import event_abi_to_log_topic
from web3._utils.events import get_event_data
# tools and config
from src import config as C, tools as TL, tracer as TCE

class Decoder(TL.RPCHandler):
    """
        Decode transaction and/or logs args to K-V pair
    """
    def __init__(self) -> None:
        self.init_addr_dict()
        super().__init__()

    def choose_one_rpc(self, chain: str) -> Web3:
        """
            choose (randomly) from the prepared rec endpoints in order not to 
            initiate an instance every time when called
        """
        return random.choice(self.rpcs[chain])

    def init_addr_dict(self):
        d = {
            "Ethereum":{
            },
            'BSC':{}, 
            'Polygon':{},
            'Fantom':{}, 
            'Base':{},
            "Arbitrum":{}, 
            "Optimism":{
            }
        }
        d['BSC'].update(d['Ethereum'])
        d['Fantom'].update(d['Ethereum'])
        d['Polygon'].update(d['Ethereum'])
        d['Optimism'].update(d['Ethereum'])
        d['Base'].update(d['Ethereum'])
        d['Arbitrum'].update(d['Ethereum'])

        self.d = d 

    def get_actual_logic_address(self, chain:str, addr:str):
        """
            Some transaction will conduct a `delegate call` from proxy contract to logic contract.
            However, the `address` field in a log entry was set to **proxy contract**, 
            which makes it challenge to decode log.
            This function will return the actual logic contract address(if it were).
            TODO: find the logic contract instead of a dict
        """
        d = self.d
        if chain in d and addr.lower() in d[chain]: 
            return d[chain][addr.lower()]
        return addr


def normalize_transaction(trx:dict):
    ret = copy.deepcopy(trx)
    if 'functionName' not in ret:
        ret['functionName'] = ''
    if ret['input'].startswith('0x00000'):
        ret['input'] = '0x' + ret['input'][10:]
    if 'methodId' not in ret:
        ret['methodId'] = ret['input'][:10]
    try:
        del ret['v']
        del ret['yParity']
    except:
        pass
    return ret

class TxDecoder(Decoder):

    def __init__(self) -> None:
        super().__init__()
        self.skip_method = set()
        self.cache = dd(list)
        self.fail_record = dd(lambda: 0)

    def go(self, tx_obj:dict, chain:str, given_abi:T.Union[str, None]=None):
        pass # will be release upon paper accepted


def _deal_nested_log_item(item:dict):
    ret = {}
    ret['indexed'] = item['indexed']
    value = item['value']
    ret['name'] = item['name']
    if isinstance(value, dict):
        ret['type'] = 'tuple'
        ret['components'] = []
        string = item['type'].strip('()')
        type_comps = string.split(',')
        i = 0
        for k, v in value.items():
            if isinstance(v, dict): 
                # TODO: ?
                a = _deal_nested_log_item(v)
                ret["components"].append({"type":"tuple","components":a})
            else:
                ret["components"].append({'type':type_comps[i], "name":k})
            i += 1
    else:
        ret['type'] = item['type']
        ret['internalType'] = item['type']
    
    return ret


class LogDecoder(Decoder):

    def __init__(self) -> None:
        super().__init__()
        self.skip_event = set()
        self.fail_record = dd(lambda: 0)

    def decode_log(self,result: T.Dict[str, T.Any], abi:T.List[T.Dict], contract_instance):
        pass # will be release upon paper accepted

    def go(self, log_entries:T.List[T.Dict[str, T.Any]], chain:str):
        pass # will be release upon paper accepted


