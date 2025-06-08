# -*- coding:utf-8 -*-
import os 
from pathlib import Path

if __name__=='__main__':
    import sys
    p = Path(__file__).absolute().parent.parent
    os.chdir(p)
    sys.path.append(str(p))

from src import config as C, tools as TL, secret as S
from src.vm import VM
from collections import defaultdict as dd

import json, typing as T
import logging
import time
import atexit

class Tracer:
    def __init__(self, write_out_path:str='') -> None:
        self.func_open = open
        self.write_out_path = write_out_path or (C.paths().trace_cache)
        self.counter = 0 
        self.changed = False
        p = (C.paths().trace_cache)
        if os.path.exists(p):
            f = open(p)
            self.cache = json.load(f)
            f.close()
        else:
            self.cache = {}

    def _save_trace_cache(self):
        if not self.changed: return 
        self.changed = False
        with self.func_open(self.write_out_path, 'w') as f:
            json.dump(self.cache, f, indent=1)

    # def __del__(self):
    #     self._save_trace_cache()

    def send_request_common(self, url, method, params):
        self.changed = True
        H = {"Content-Type": "application/json", "Accept": "application/json"}
        j = {
            "jsonrpc": "2.0","id": 0,
            "method": method,
            "params": params
        }
        resp = TL.get_url(url, 'post', headers=H, _json=j, )
        return resp

    def send_debugtrace_request(self, chain:str, tx_hash, method:str='debug_traceTransaction'):

        traceConfig = {'tracerConfig':{'enableMemory':True, 'disableStack':False, 'disableStorage':False, 'enableReturnData':True}}
        params = [tx_hash, traceConfig]
        resp = self.send_request_common(S.RPC_TRACE_API[chain], method, params)
        time.sleep(1)
        return resp
    
    def send_tenderly_trace_request(self, chain, tx_hash, ):
        if chain in self.cache and  tx_hash in self.cache[chain] and self.cache[chain][tx_hash]:
            return self.cache[chain][tx_hash]
        d = {"jsonrpc": "2.0", "id": 0,
            "method": "tenderly_traceTransaction",
            "params": [tx_hash]
        }
        resp = TL.get_url(S.RPC_TRACE_API_TENDERLY[chain], 'post', _json=d, )
        if chain not in self.cache: self.cache[chain] = {}
        self.cache[chain][tx_hash] = resp
        self.counter += 1
        self.changed = True
        if self.counter % 30 == 0: 
            self._save_trace_cache()
        time.sleep(1)
        return resp

    def get_tx_info(self, tx_hash, chain):
        resp = self.send_request_common(TL.get_rpc_endpoints(chain, True, False), 'eth_getTransactionByHash', [tx_hash,])
        obj = json.loads(resp)
        r = obj.get('result', None)
        retry_times = 0
        while r is None and retry_times < 5:
            # retry once 
            time.sleep(1 + 4 * retry_times)
            resp = self.send_request_common(TL.get_rpc_endpoints(chain, True, False), 'eth_getTransactionByHash', [tx_hash,])
            try:
                obj = json.loads(resp)
                r = obj.get('result', None)
            except json.decoder.JSONDecodeError as e:
                logging.exception(e)
                logging.warning(f"retry {retry_times}")
            finally:
                retry_times += 1
        if r is None:
            logging.error("Chain %s getting fail! %s" % (chain, tx_hash))
        to_addr, block_num, input_data, _gas, _input_value, caller = r['to'], r['blockNumber'], r['input'], r['gas'], r['value'], r['from']
        # resp2 = self.send_request_common(TL.get_rpc_endpoints(chain, True, False), 'eth_getCode', [to_addr, block_num])
        # obj2 = json.loads(resp2)
        # code = obj2['result']
        return to_addr, block_num, input_data, _gas, _input_value, caller

    def convert_tenderly_to_common(self, resp:str):
        if isinstance(resp, dict) and all( a in resp for a in ('log', 'call', 'asset')):
            return resp
        obj = json.loads(resp)
        res = obj['result']
        ret = dd(list)
        for log_obj in res.get('logs', ()):
            ret['log'].append({'address': log_obj['raw']['address'], 'topics':log_obj['raw']['topics'], 'data':log_obj['raw']['data']})
        for trace_obj in res.get('trace', ()):
            ret['call'].append(trace_obj)
            if trace_obj['type'] == 'CALL' and TL.save_value_int(trace_obj.get('value', '0')) != 0:
                ret['asset'].append({'type':'cash', 'to':trace_obj['to'], 'from':trace_obj['from'] , 'value':TL.save_value_int(trace_obj.get('value'))})
        for asset in res.get('assetChanges', ()):
            if 'to' not in asset or 'rawAmount' not in asset: continue
            if asset['assetInfo'].get('type') == 'Native': 
                k = 'symbol'
                _t = 'cash'
            else:
                k = 'contractAddress'
                _t = 'token'
            ret['asset'].append({'type':_t, 'to': asset['to'] , 'value':TL.save_value_int(asset['rawAmount']),
                        'addr':asset['assetInfo'][k]})
        return ret


    def get_trace_common(self, chain:str, tx_hash, trace_type:str='call') -> T.List[T.Tuple[str, str]]:
        
        if chain in S.RPC_TRACE_API_TENDERLY:
            resp = self.send_tenderly_trace_request(chain, tx_hash)
            if 'Please try again in 30 seconds' in resp or 'rate limit exceeded' in resp or not (resp): 
                del self.cache[chain][tx_hash]
                time.sleep(35)
                resp = self.send_tenderly_trace_request(chain, tx_hash)
            flows = self.convert_tenderly_to_common(resp)
            if trace_type == 'call': 
                return list( (a['to'],a['input']) for a in flows[trace_type] if a['type'] == 'DELEGATECALL')
            return flows[trace_type]

        if chain in self.cache and  tx_hash in self.cache[chain]:
            caches = self.cache[chain][tx_hash]
            if trace_type == 'call': 
                return list( (a['to'],a['input']) for a in caches[trace_type] if a['type'] == 'DELEGATECALL')
            return caches[trace_type]
        resp = self.send_debugtrace_request(chain, tx_hash, 'debug_traceTransaction')
        ret = []
        try:
            obj = json.loads(resp)
            to_addr, block_num, input_data, _gas, _input_value, caller = self.get_tx_info(tx_hash, chain)
            vm = VM(obj['result']['structLogs'], chain=chain, input_data=input_data, to_addr=to_addr, 
                    block_num=block_num, gas=_gas, input_value=_input_value, caller=caller)
            if trace_type == 'call':
                trace_calls = vm.call_traces
                for trace_call in trace_calls:
                    if trace_call['type'] != 'DELEGATECALL': continue
                    ret.append((trace_call['to'], trace_call['input']))
            elif trace_type == 'log':
                trace_logs = vm.log_traces
                ret = trace_logs
            elif trace_type == 'asset':
                ret = vm.asset_flows
            if chain not in self.cache: self.cache[chain] = {}
            self.cache[chain][tx_hash] = {'call':vm.call_traces, 'log':vm.log_traces, 'asset': vm.asset_flows}
        except Exception as e:
            logging.exception(e)

        return ret

    def get_logs_trace(self, chain:str, tx_hash):
        resp = self.send_debugtrace_request(chain, tx_hash, 'debug_traceTransaction')
        obj = json.loads(resp)
        ret = []
        try:
            result = obj['result']
            trace_logs = result['logs']
            return trace_logs
        except Exception as e:
            logging.exception(e)


def get_tracer(*args, **kwargs):
    g = globals()
    t = g.get('TRACER', None)
    if t is not None:
        return t 
    t = Tracer(*args, **kwargs)
    g['TRACER'] = t 
    atexit.register(t._save_trace_cache)
    return t 


if __name__=='__main__':
    _tc = Tracer()
    _a = _tc.send_debugtrace_request('Arbitrum', '0x6a2f0684de89eb2ea551fd5c832f5791c4d45eea986ecb024522ea11899db96f')
    print(_a)