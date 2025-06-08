# -*- coding:utf-8 -*-

import os, json, logging
from pathlib import Path
import typing as T
from easydict import EasyDict as ed
if __name__=='__main__':
    import sys
    p = Path(__file__).absolute().parent.parent
    os.chdir(p)
    sys.path.append(str(p))

from src import tools as TL


GET_TX_INFO_URL = {
    'Ethereum':'https://rpc.ankr.com/eth/c8bf28d6fd4bd064dd5c0168b7280ed0a285b5bc0a0e24ecc7ea2becc02ca15f', 
    'BSC':'https://rpc.ankr.com/bsc/c8bf28d6fd4bd064dd5c0168b7280ed0a285b5bc0a0e24ecc7ea2becc02ca15f', 
    'Arbitrum':'https://rpc.ankr.com/arbitrum/c8bf28d6fd4bd064dd5c0168b7280ed0a285b5bc0a0e24ecc7ea2becc02ca15f', 
    'Base':'https://rpc.ankr.com/base/c8bf28d6fd4bd064dd5c0168b7280ed0a285b5bc0a0e24ecc7ea2becc02ca15f', 
    'Optimism':'https://rpc.ankr.com/optimism/c8bf28d6fd4bd064dd5c0168b7280ed0a285b5bc0a0e24ecc7ea2becc02ca15f', 
    'Polygon':'https://rpc.ankr.com/polygon/c8bf28d6fd4bd064dd5c0168b7280ed0a285b5bc0a0e24ecc7ea2becc02ca15f', 
}

def str_split_step(s:str, step:int):
    return list( s[i:i+step] for i in range(0, len(s), step))

def get_code_by_addr(addr:str, block_num:str, chain:str)->str:
    H = {"Content-Type": "application/json", "Accept": "application/json"}
    addr = TL.normalize_address_type(addr)
    params = [addr, block_num]
    j = {
        "jsonrpc": "2.0","id": 0,
        "method": 'eth_getCode',
        "params": params
    }
    resp = TL.get_url(GET_TX_INFO_URL[chain], 'post', headers=H, _json=j)
    obj = json.loads(resp)
    return obj['result']


def to_hex_string(s:str):
    if s.startswith('0x') or s.startswith('0X'): 
        return s
    return hex(int(s))

class Memory:
    def __init__(self):
        self.value_map:T.Dict[int,int] = {}
        self.debug_records = []
        self.WORD_LEN = 32

    def save_value_int(self, v):
        return TL.save_value_int(v)

    def pretty_print(self, pos:int, length:int):
        for i in range(pos, pos + length):
            print('%02x' % self.value_map.get(i, 0), end=('' if i % 32 != 31 else ' '))
        print()
    
    def print_debug_rec(self):
        for rec in self.debug_records:
            if rec.op in ('MSTORE', 'MSTORE8'):
                print(rec.op, "pos: %s, value:%s" % (rec.stack[-1], rec.stack[-2] if len(rec.stack) > 2 else '') )
            elif rec.op in ('CALLDATACOPY', 'RETURNDATACOPY'):
                print(rec.op, 'dest:%s, offset:%s, length:%s' % (rec.stack[-1], rec.stack[-2], rec.stack[-3] if len(rec.stack) > 3 else '') )
            elif rec.op in ('CALL', 'CALLCODE'):
                print(rec.op, "retOffset: %s, retLength:%s" % (rec.stack[-6], rec.stack[-7], ) )
            elif rec.op in ('DELEGATECALL', 'STATICCALL'):
                print(rec.op, "retOffset: %s, retLength:%s" % (rec.stack[-5], rec.stack[-6], ) )
            else:
                print(rec.op, rec.stack[-1], rec.stack[-2],)

    def store_one_uint_word(self, pos, value, length=32):
        cur_value = value
        end_pos = pos + length - 1
        for i in range(length):
            v = cur_value & (0xff)
            self.value_map[end_pos - i] = v
            cur_value = cur_value >> 8
    
    def store(self, pos, value, length=32):
        length = self.save_value_int(length)
        pos = self.save_value_int(pos)
        value = self.save_value_int(value)
        if length <= self.WORD_LEN:
            self.store_one_uint_word(pos, value, length)
            return
        cur_ptr = 0 
        while cur_ptr < length:
            a = (1 << (self.WORD_LEN * 8)) - 1
            shift_len = ((length - cur_ptr) * 8) - (self.WORD_LEN * 8)
            if shift_len < 0: 
                shift_len += self.WORD_LEN * 8
                a = (1 << shift_len) - 1
                v = value & a
            else:
                a = a << shift_len 
                v = (value & a ) >> shift_len
            self.store_one_uint_word(pos + cur_ptr, v, min(self.WORD_LEN, length - cur_ptr) )
            cur_ptr += self.WORD_LEN
    
    def load_one_uint_word(self, pos, length=32):
        length = self.save_value_int(length)
        pos = self.save_value_int(pos)
        ret = 0
        for i in range(length):
            # v = self.value_map.get(pos + i, 0)
            # if i: v = (v << (i * 8))
            # ret += v
            ret = ret << 8
            ret += self.value_map.get(pos + i, 0)
        return ret 

    def load(self, pos, length=32):
        # ! EVM 是大端序： https://calnix.gitbook.io/eth-dev/evm-storage-opcodes/evm
        length = self.save_value_int(length)
        pos = self.save_value_int(pos)
        if length <= self.WORD_LEN: 
            return self.load_one_uint_word(pos, length)
        cur_length = length
        cur_pos = pos
        ret = 0
        while cur_length > 0:
            bit = min(cur_length, self.WORD_LEN)
            v = self.load_one_uint_word(cur_pos, bit) 
            ret = (ret << (bit * 8)) + v
            cur_length -= self.WORD_LEN
            cur_pos += self.WORD_LEN
        return ret
    
    def copy_in(self, pos, length, value:int):
        pos = self.save_value_int(pos)
        length = self.save_value_int(length)
        return self.store(pos, value, length)

    def add_debug_record(self, trace):
        self.debug_records.append(trace)

class VM:
    def __init__(self, traces=None, filename=None, **kwargs) -> None:
        self.traces = []
        self.call_traces = []
        self.log_traces = []
        self.asset_flows = []
        self.mem = Memory()
        if traces: 
            self.init_from_list(traces)
        elif filename:
            self.init_from_file(filename)
        input_data = kwargs.get('input_data', '')
        self.input_data = input_data[2:] if input_data.startswith('0x') else input_data
        self.to_addr = kwargs.get('to_addr', '')
        self.block_num = kwargs.get('block_num', '')
        self.chain = kwargs.get('chain', '')
        self.gas = TL.save_value_int(kwargs.get('gas', 0))
        self.input_value = TL.save_value_int(kwargs.get('input_value', 0))
        self.caller = kwargs.get('caller', 0)
        try:
            self.analyze()
        except Exception as e:
            logging.exception(e)

    def init_from_list(self, content:T.List[T.Dict[str, T.Any]]):
        self.traces = content
    
    def init_from_file(self, filename:str):
        with open(filename) as f:
            self.traces = json.load(f)

    def analyze_one_contract_trace(self, traces:T.List[dict], contract_addr:str, input_data:str, mem:Memory=None, gas:int=0, input_value:int=0, caller:str=''):
        """
            analyze trace within one contract, that is, if there's a call/delegatecall,
            the function will be recursed
        """
        if input_data.startswith('0x'):
            input_data = input_data[2:]
        trace_index = 0
        return_data = ''
        if mem is None:
            mem = Memory()
        while trace_index < len(traces):
            trace = ed(traces[trace_index])
            trace.stack = list(map(to_hex_string, trace.stack))
            if trace.op == 'DELEGATECALL':
                _gas, _addr, _argsOffset, _argsLength, _retOffset, _retLength = trace.stack[-6:][::-1]
                call_input = TL.hex_pad(mem.load(_argsOffset, _argsLength))
                self.call_traces.append({
                    'type':'DELEGATECALL', 'from': contract_addr,'to': _addr , 
                    'input': (call_input), 'value':0, 'gas':_gas
                })
                # ! DelegateCall differs from CallCode in the sense that it executes the given address
                # ! code with the caller as context and the caller is set to the caller of the caller
                # See: https://github.com/ethereum/go-ethereum/blob/master/core/vm/evm.go#L177
                comp = self.analyze_one_contract_trace(traces[trace_index+1:], contract_addr, call_input, Memory(), _gas, caller)
                if comp:
                    used_pc, ret_value = comp
                    if not len(ret_value):
                        ret_value = '0x0'
                    if mem.save_value_int(_retLength):
                        mem.store(_retOffset, ret_value, _retLength)
                    mem.add_debug_record(trace)
                    return_data = ret_value
                    trace_index += used_pc
                else:
                    logging.warning('No data return from this call')
            elif trace.op == 'CALL':
                _gas, _addr, _value, _argsOffset, _argsLength, _retOffset, _retLength = trace.stack[-7:][::-1]
                call_input = TL.hex_pad(mem.load(_argsOffset, _argsLength))
                self.call_traces.append({
                    'type':'CALL', 'from': contract_addr, 'to': _addr , 
                    'input': (call_input), 'value':_value, 'gas':_gas
                })
                self.asset_flows.append({
                    'type':'cash', 'from': contract_addr, 'to': _addr , 'value':_value,
                })
                used_pc, ret_value = self.analyze_one_contract_trace(traces[trace_index+1:], _addr, call_input, Memory(), _gas, _value, contract_addr)
                if not len(ret_value):
                    ret_value = '0x0'
                if mem.save_value_int(_retLength) :
                    mem.store(_retOffset, ret_value, _retLength)
                mem.add_debug_record(trace)
                return_data = ret_value
                trace_index += used_pc
            elif trace.op == 'STATICCALL':
                _gas, _addr, _argsOffset, _argsLength, _retOffset, _retLength = trace.stack[-6:][::-1]
                call_input = TL.hex_pad(mem.load(_argsOffset, _argsLength))
                self.call_traces.append({
                    'type':'STATICCALL', 'from': contract_addr,'to': _addr , 
                    'input': (call_input), 'value':0, 'gas':_gas
                })
                
                tmp_res = self.analyze_one_contract_trace(traces[trace_index+1:], _addr, call_input, Memory(), _gas, 0, contract_addr)
                if tmp_res is not None:
                    used_pc, ret_value = tmp_res
                else:
                    used_pc, ret_value = len(traces) - trace_index, None
                if not len(ret_value):
                    ret_value = '0x0'
                if mem.save_value_int(_retLength):
                    mem.store(_retOffset, ret_value, _retLength)
                mem.add_debug_record(trace)
                return_data = ret_value
                trace_index += used_pc
            elif trace.op == 'CALLCODE':
                raise NotImplementedError('CALLCODE')
            elif trace.op == 'MSTORE':
                pos, value = trace.stack[-1], trace.stack[-2]
                mem.store(pos, value)
                mem.add_debug_record(trace)
            elif trace.op == 'MSTORE8':
                pos, value = trace.stack[-1], trace.stack[-2]
                mem.store(pos, value, 1)
                mem.add_debug_record(trace)
            elif trace.op == 'MLOAD':
                pos = trace.stack[-1]
                next_trace = ed(traces[trace_index + 1])
                a = to_hex_string(next_trace.stack[-1])
                if a != hex(mem.load(pos)): 
                    # logging.warning("memory dismatch: %s v.s. %s" % (a, hex(mem.load(pos)) ) )
                    # write back
                    mem.store(pos, a)
                    mem.add_debug_record(trace)
            elif trace.op == 'MCOPY':
                raise NotImplementedError('MCOPY')
            elif trace.op == 'CALLDATACOPY':
                _destoffset, _offset, _length = trace.stack[-1], trace.stack[-2], trace.stack[-3]
                _destoffset, _offset, _length = tuple(map(mem.save_value_int, (_destoffset, _offset, _length) ))
                a = input_data[_offset * 2 : (_offset + _length) * 2]
                if _length != 0 and len(a): 
                    mem.copy_in(_destoffset, _length, int(a, base=16))
                    mem.add_debug_record(trace)
            
            elif trace.op == 'CODECOPY':
                _destoffset, _offset, _length = trace.stack[-1], trace.stack[-2], trace.stack[-3]
                _destoffset, _offset, _length = tuple(map(mem.save_value_int, (_destoffset, _offset, _length)))
                if _length != 0:
                    code = get_code_by_addr(contract_addr, self.block_num, self.chain)
                    if code.startswith('0x'): 
                        code = code[2:]
                    a = code[_offset * 2 : (_offset + _length) * 2]
                    if len(a):
                        mem.copy_in(_destoffset, _length, int(a, base=16))
                        mem.add_debug_record(trace)
                    else:
                        logging.warning("Code offset({offset},{length}) exceed code length({code_len})".format(
                            offset=_offset, length = _length, code_len = len(code)
                        ))
            elif trace.op == 'EXTCODECOPY':
                raise NotImplementedError('EXTCODECOPY')
            elif trace.op == 'RETURN':
                _offset, _length = trace.stack[-1], trace.stack[-2]
                # _offset, _length = tuple(map(mem.save_value_int, (_offset, _length)))
                ret = mem.load(_offset, _length)
                return trace_index+1, TL.hex_pad(ret, True)[2:]
            elif trace.op == 'RETURNDATACOPY':
                _destoffset, _offset, _length = trace.stack[-1], trace.stack[-2], trace.stack[-3]
                _destoffset, _offset, _length = tuple(map(mem.save_value_int, (_destoffset, _offset, _length)))
                a = return_data[_offset * 2 : (_offset + _length) * 2] if (len(return_data) and return_data != '0x0') else ''
                if _length != 0 and len(a):
                    mem.copy_in(_destoffset, _length, mem.save_value_int(a))
                mem.add_debug_record(trace)
            elif trace.op.startswith('LOG'):
                _offset, _length = trace.stack[-1], trace.stack[-2]
                topic_len = int(trace.op[3:])
                topics = []
                for i in range(topic_len): 
                    topics.append(trace.stack[-3 - i])
                data = mem.load(_offset, _length)
                self.log_traces.append({
                    'address': contract_addr,
                    'topics':topics, 
                    'data':hex(data)
                })
                if len(topics) and topics[0] == '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef':
                    self.asset_flows.append({
                        'type':'token', 'from': topics[1], 'to': topics[2] , 'value':TL.save_value_int(data),
                        'addr':contract_addr
                    })
            elif trace.op in ('STOP', 'REVERT', 'SELFDESTRUCT'):
                return trace_index + 1, ''

            trace_index += 1
        # logging.warning('No data return from this call')
        return len(traces), ''

    def analyze(self):
        if not self.traces: return 
        self.call_traces.append({
                'type':'CALL', 'from': self.caller, 'to': self.to_addr , 
                'input': (self.input_data), 'value':self.input_value, 'gas':self.gas
            })
        if self.input_value not in (0, 0.0):
            self.asset_flows.append({
                'type':'cash', 'from': self.caller, 'to': self.to_addr , 'value':self.input_value,
            })
        self.analyze_one_contract_trace(self.traces, self.to_addr, self.input_data, Memory(), self.gas, self.input_value, self.caller)


def testvm():
    with open('tmp_Arbitrum.json') as f:
        obj = json.load(f)
    input_data = '0x1fd8010c00000000000000000000000000000000000000000000000000000000000000400000000000000000000000000000000000000000000000000000000000000200b39f7d6a6e532b0a76ff04f8680a325792e997b67dd16353c922128b3689c7f10000000000000000000000000000000000000000000000000000000000000140000000000000000000000000000000000000000000000000000000000000018000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000623302b90575186c55d874cfa510cab5da563a2a0000000000000000000000000000000000000000000000000009f295cd5f000000000000000000000000000000000000000000000000000000000000000000890000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000066163726f73730000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000f6a756d7065722e65786368616e67650000000000000000000000000000000000000000000000000000000000000000000000000000000000000cf8907de9a7440000000000000000000000000000000000000000000000000000000065e0a7a80000000000000000000000000000000000000000000000000000000000000080ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff0000000000000000000000000000000000000000000000000000000000000000d00dfeeddeadbeef8932eb23bad9bddb5cf81426f78279a53c6c3b71'
    vm = VM(obj['result']['structLogs'], chain='Arbitrum',to_addr='0x1231DEB6f5749EF6cE6943a275A1D3E7486F4EaE',input_data=input_data, block_num='0xb11f079')
    print(vm.call_traces)


def test_token_transfer():
    # https://etherscan.io/tx/0xc55524b6a315e8a14b3b3b89d624ab1f13cf65f71f5c3d1d091cde50789a4d25
    with open('test_storage.json') as f: 
        obj = json.load(f)
    input_data = '0x9fbf10fc000000000000000000000000000000000000000000000000000000000000006e00000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000001000000000000000000000000e9569fc84f72134a62c0ff426e2e932a7e4c013a00000000000000000000000000000000000000000000000000000006fc23ac0000000000000000000000000000000000000000000000000000000006f332da80000000000000000000000000000000000000000000000000000000000000012000000000000000000000000000000000000000000000000000000000000001c00000000000000000000000000000000000000000000000000000000000000200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000060000000000000000000000000000000000000000000000000000000000000001400000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000014e9569fc84f72134a62c0ff426e2e932a7e4c013a0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
    vm = VM(obj['result']['structLogs'], chain='Ethereum',to_addr='0x8731d54E9D02c286767d56ac03e8037C07e01e98',
            input_data=input_data, block_num='0x1270521')
    print(vm.log_traces)

if __name__=='__main__':
    test_token_transfer()