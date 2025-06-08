# -*- coding:utf-8 -*-

"""
    Extractor aims to extract some infomation from the massive records.
    V1.0: mannual extract for each Bridge
"""

import multiprocessing.pool
from web3 import Web3
import random, json, os, copy, logging, string, itertools, multiprocessing
from easydict import EasyDict as ed
from collections import defaultdict as dd
import typing as T
from src import tools as TL, config as C, secret as S, doc_info as DOC, tracer as TCE
from src.common import my_timer, AncestorTree
from src.extractor_old import Extractor, ExtractorDL, ExtractorRule, ExtractorCompound
import openai

def _debug_should_stop(candidate_ans_str):
    return '"amount": "log[FulfilledOrder].args.takeAmount", "chain": "log[FulfilledOrder].args.order.giveChainId", "timestamp": "transaction.timeStamp", "to": "log[FulfilledOrder].args.receiverDst", "token": "log[FulfilledOrder].args.takeTokenAddress"' in candidate_ans_str

class ExtractorLLM(Extractor):
    
    # ! Open AI Python SDK: https://cookbook.openai.com/examples/assistants_api_overview_python
    # ! and https://platform.openai.com/docs/api-reference/chat/create
    # ! API refernce: https://platform.openai.com/docs/api-reference/introduction

    QUERY_TIMES_FOR_VOTING = 5

    def __init__(self, model_type:str='gpt-4o-mini', check_rpc:bool=True):
        super().__init__(check_rpc)
        f = open(C.paths().LLM_prompt_score)
        self.PROMPT = json.load(f)
        self.PRESET_KEYS = [C.PRESET_KEYS_TIMESTAMP, C.PRESET_KEYS_TOKEN, C.PRESET_KEYS_AMOUNT, C.PRESET_KEYS_TO, C.PRESET_KEYS_CHAIN,]
        f.close()
        self._load_LLM_record()
        self.LLM_records_changed = False
        self.client = None
        self.lock = multiprocessing.Lock()
        self.GT = get_transaction_hash_in_gt_scope(load_GT())
        self.first_query_times = dd(lambda : dd(lambda: self.QUERY_TIMES_FOR_VOTING))
        self.voting_keys = dd(lambda: dd(lambda: dd(lambda: dd(lambda: dd(lambda: 0))))) # chain => (func_name => (preset_keys => (chosen_keys by LLM => count)) ) 
        self.first_chosen_fields = dd(lambda: dd(lambda: dd(lambda: 0))) # chain => (func_name =>  ) 
        self.second_chosen_fields = dd(lambda: dd(lambda: dd(lambda: 0))) # chain => (func_name =>  ) 
        self.final_chosen_keys = dd(lambda: dd(lambda: dd(lambda: dd(lambda: 0)))) # chain => (role => meta_structure => {} ) 
        self.llm_chosen_fields_status = dd(lambda: dd(lambda: dd(lambda: dd(lambda : dict())))) # chain => (func_name => (preset_keys => (status) ) )
        self.unfound_answer_trxs = dd(lambda: dd(lambda: dd(list)))
        self.model_type = model_type

    def _add_LLM_record(self, response:openai.types.chat.chat_completion.ChatCompletion, input_id=None, **kwargs):
        """
            add one record to file
        """
        self.LLM_record_indexes[input_id] = len(self.LLM_records.chats)
        a = {
            'input_id':input_id,
            "output":response.choices[0].message.content,
            "in_len":response.usage.prompt_tokens,
            "out_len":response.usage.completion_tokens or TL.num_tokens_from_string(response.choices[0].message.content)
        }
        if len(kwargs): 
            a.update(kwargs)
        self.LLM_records.chats.append(a)
        self.LLM_records.statis.inputs += response.usage.prompt_tokens
        self.LLM_records.statis.outputs += response.usage.completion_tokens
    
    def __del__(self):
        # ! this is only for debugging when developping code. remove destructor when code is ready
        self._save_LLM_record()
        self._save_final_keys()

    def _meta_structure_should_drop(self, meta:str):
        return 'tokenid' in meta.lower()
    
    def _get_dependable_meta_structure(self, chain:str, trx:dict, logs,)->str:
        meta_structure = _get_meta_structure(trx, logs, True, self.meta_structures[chain]['ignore'])
        if meta_structure not in self.meta_structures[chain]['keys']: 
            # 如果不在之前选定的记录中，就选择它的祖先。
            # TODO: 后续变成LLM选择
            meta_structure = self.meta_structures[chain]['tree'].get_ancestor(meta_structure)
        if self._meta_structure_should_drop(meta_structure): 
            return ""
        return meta_structure

    def go(self, trx:dict, logs:T.List[dict], chain:str):
        """
            call this functin after `get_diverse_field`, `go_for_voting` and `verify_voting_result`
        """
        def _go(chosen_keys):
            chosen_keys_wrap = { a:(b if isinstance(b,list) else [b] ) for a, b in chosen_keys.items()} 
            extract_res = self.extract_by_rule(trx, logs, chosen_keys_wrap)
            to, token_addr, amount, timeStamp, paired_chain = (
                extract_res.get(C.PRESET_KEYS_TO, None),
                extract_res.get(C.PRESET_KEYS_TOKEN, ''),
                extract_res.get(C.PRESET_KEYS_AMOUNT, None),
                extract_res.get(C.PRESET_KEYS_TIMESTAMP, None),
                extract_res.get(C.PRESET_KEYS_CHAIN, None),
            )

            paired_chain = DOC.chain_id_to_name[C.current_bridge].get(paired_chain, '')

            to = _normalize_address_type(to)
            token_addr = _normalize_address_type(token_addr)

            token_name, token_symbol, amount = self.deal_token_and_amount(token_addr, int(amount), trx, chain, self.choose_one_rpc(chain))

            return {
                'to':to, 'token_addr':token_addr, 'token_name':token_name,
                'token_symbol':token_symbol, 'amount':amount, 'timestamp':(timeStamp), 
                'role':role, 'paired_chain':paired_chain, 'chain':chain,
                'hash':trx['hash']
            }

        def _is_chosen_keys_valid(chosen_keys:dict):
            for k in C.PRESET_KEYS: 
                if k not in chosen_keys: return False 
                if not len(chosen_keys[k]): return False
            return True

        func_name = trx['pureFunctionName'] or trx['methodId']
        role = DOC.function_name_to_role[C.current_bridge].get(func_name, None)
        meta_structure = self._get_dependable_meta_structure(chain, trx, logs)
        # if meta_structure in self.final_chosen_keys[chain][role]:
        chosen_keys = self.final_chosen_keys[chain][role][meta_structure]
        if _is_chosen_keys_valid(chosen_keys): 
            return  _go(chosen_keys) 
        all_descendants = self.meta_structures[chain]['tree'].get_all_descendants(meta_structure)
        for one_descendant in all_descendants: 
            if one_descendant not in self.final_chosen_keys[chain][role]: continue
            chosen_keys = self.final_chosen_keys[chain][role][one_descendant]
            if not _is_chosen_keys_valid(chosen_keys): continue
            try: return _go(chosen_keys)
            except: continue

        # 从邻近的meta_structure里选一个
        for other_chain in self.final_chosen_keys:
            if meta_structure in self.final_chosen_keys[other_chain][role]:
                other_keys = self.final_chosen_keys[other_chain][role][meta_structure]
                if not other_keys: continue
                try: 
                    ret = _go(other_keys)
                    return ret
                except: 
                    continue
        for other_meta_structure in self.final_chosen_keys[chain][role]:
            if other_meta_structure in all_descendants: continue # 上面已经检查过这个集合
            other_keys = self.final_chosen_keys[chain][role][other_meta_structure]
            if not other_keys: continue
            try: 
                ret = _go(other_keys)
                return ret
            except: 
                continue
        return -1 
        # else:
        #     # logging.warning(f"meta strcuture not found answer: {meta_structure}")
        #     return -2
        

    def _save_LLM_record(self):
        self.LLM_records_changed = False
        if not hasattr(self, 'LLM_records'): return 
        with open(C.paths().LLM_log, 'w') as f: json.dump(self.LLM_records, f, indent=1)
    
    def _load_LLM_record(self):
        p = C.paths().LLM_log
        if os.path.exists(p):
            with open(p) as f: 
                self.LLM_records = ed(json.load(f))
            self.LLM_record_indexes = {}
            for i, chat in enumerate(self.LLM_records.chats):
                self.LLM_record_indexes[chat.input_id] = i
        else:
            self.LLM_records = ed({
                'chats':[],
                'statis':{'inputs':0, 'outputs':0}
            })
            self.LLM_record_indexes:T.Dict[str, int] = {} # from "input_id"(str) to "its index within array(i.e. self.LLM_records.chats)"

    def _load_final_keys(self):
        if os.path.exists(C.paths().chosen_keys):
            f = open(C.paths().chosen_keys)
            _o = json.load(f)
            f.close()
            TL.copy_dict_to_nested_defaultdict(_o, self.final_chosen_keys)
            return True
        return False
    
    def _save_final_keys(self):
        with open(C.paths().chosen_keys, 'w') as f:
            json.dump(self.final_chosen_keys, f, indent=1)

    def _decode_LLM_responce(self, resp:str):
        try:
            obj = json.loads(resp)
            return obj
        except json.JSONDecodeError:
            return {}
    
    def reorganize_bridge_data_by_role_and_hash(self, all_bridge_data):
        ret = {}
        default_role = None
        for chain in all_bridge_data:
            if chain not in ret: 
                ret[chain] = {}
            for one_data in all_bridge_data[chain]:
                trx = one_data[0]
                if not len(trx): continue
                func_name = trx['pureFunctionName'] or trx['methodId']
                role = DOC.function_name_to_role[C.current_bridge].get(func_name, default_role)
                if role is None:
                    continue
                meta_structure:str = self._get_dependable_meta_structure(chain, trx, one_data[1:])
                if not meta_structure: 
                    continue
                if role not in ret[chain]:  ret[chain][role] = {}
                if meta_structure not in ret[chain][role]:
                    ret[chain][role][meta_structure] = {}
                ret[chain][role][meta_structure][trx['hash']] = (one_data)
        return ret


    def _filter_bridge_data_by_role_and_ts(self, all_bridge_data, chain, role, timestamp, treshold:int=3600):
        return filter_bridge_data_by_role_and_ts(all_bridge_data, chain, role, timestamp, treshold)


    def begin_voting(self, chain:str, role:str, meta_strucutre:str=''):
        # voting algorithm
        if meta_strucutre in self.voting_keys[chain][role] and len(self.voting_keys[chain][role][meta_strucutre]):
            return 
        if self.LLM_records_changed:
            self._save_LLM_record()
        iid = '%s.%s' % (chain, role)
        for chat in self.LLM_records.chats:
            tmp = chat['input_id'].split('.')
            if chat['input_id'].startswith(iid) and TL.md5_encryption(meta_strucutre) == chat['meta_structure']:
                try:
                    output = TL.pure_json_object(chat['output'], True)
                except:
                    output = None
                if output is None: continue
                obj = self._decode_LLM_responce(output)
                if not len(obj): continue # this response failed
                trx_hash = tmp[2]
                try:
                    for preset_key in C.PRESET_KEYS:
                        for ck, score, _ in obj[preset_key][:3]:
                            self.voting_keys[chain][role][meta_strucutre][preset_key][ck] += score
                except Exception as e:
                    logging.exception(e)

    def _handle_instance_to_check(self, trx, logs)->str:
        return TL._handle_instance_to_ask_LLM(trx, logs)

    def _pack_first_query_prompt(self, trx:dict, logs:T.List[dict], chain:str, role:str)->T.List[dict]:
        """
            The first query would add current instance to prompt (role: user)
        """
        instance_str = self._handle_instance_to_check(trx, logs)
        prompt = copy.deepcopy(self.PROMPT['first'])
        _query_sentences = ';'.join("(%d)%s" % (i, v) for i, v in enumerate(C.PRESET_KEYS_TO_PROMPT.values()))
        _query_sentences = _query_sentences.format(role='source' if role == 'dst' else 'destination')
        prompt[3]['content'] = prompt[3]['content'] % (_query_sentences, instance_str,)
        return prompt


    def _wrap_ask_LLM(self, input_id, trx, logs, chain, role, meta_structure:str):
        input_id1 = input_id
        if input_id1 not in self.LLM_record_indexes:
            prompt = self._pack_first_query_prompt(trx, logs, chain, role)
            if sum(TL.num_tokens_from_string(a['content']) for a in prompt) > C.MAX_CONTEXT_SIZE:
                # Too much token of this 
                logging.error("trx contains too much token: %s" % (trx['hash']))
                return
            # self.ask_LLM_pool.apply_async(self.ask_LLM_base, (prompt, input_id1, False, True))
            first_llm_res = self.ask_LLM_base(prompt, input_id1, False, True, meta_structure=TL.md5_encryption(meta_structure))
            self.LLM_records_changed = True
        self.first_query_times[chain][meta_structure] -= 1

    def calc_meta_structure_count(self, all_data:T.Dict[str, T.List[dict]]):
        """
            Traverse all data, to calculate the count of each meta structure.
            Less frequent meta-structure will be dropped
        """
        cache_path = os.path.join(C.paths().meta_structure)
        self.meta_structures = _init_meta_structure_trees_from_file(cache_path)
        if self.meta_structures: 
            return 

        GT = self.GT

        for chain in C.chains_to_pair: 
            chain_data = all_data[chain]
            gt_of_chain = GT[chain]
            metas_cnt: T.Dict[str, int] = dd(lambda:0)
            metas_cnt_orig: T.Dict[str, int] = dd(lambda:0)
            metas_str2sets: T.Dict[str, set] = {}
            flatten_keys_cnt: T.Dict[str, int] = dd(lambda :0)
            for tx_instance in chain_data:
                if not len(tx_instance[0]): continue
                tx_hash = tx_instance[0]['hash'].lower()
                # if tx_hash not in gt_of_chain: continue
                # 这个循环用于统计单个log的出现次数
                meta_struct:list = _get_meta_structure(tx_instance[0], tx_instance[1:])
                for ms in meta_struct: flatten_keys_cnt[ms] += 1
            metas_relation_tree = AncestorTree()
            flatten_keys_ignore = list(m for m in flatten_keys_cnt if flatten_keys_cnt[m] < self.QUERY_TIMES_FOR_VOTING )
            for tx_instance in chain_data:
                if not len(tx_instance[0]): continue
                tx_hash = tx_instance[0]['hash'].lower()
                # if tx_hash not in gt_of_chain: continue
                # 过一遍数据，统计每个meta_structure的个数。 这里分原始的和经过分组的
                meta_struct:list = _get_meta_structure(tx_instance[0], tx_instance[1:], False, flatten_keys_ignore)
                meta_struct_str = ';'.join(m for m in meta_struct)
                metas_cnt_orig[meta_struct_str] += 1 
                # metas_cnt[meta_struct_str] += 1 # 如果用了ancestor，则这里不能加
                meta_struct_set = set(meta_struct)
                if meta_struct_str in metas_str2sets: continue
                metas_str2sets[meta_struct_str] = meta_struct_set
                metas_relation_tree._make_set(meta_struct_str, metas_cnt_orig[meta_struct_str])
                for _pre_str, _pre_sets in metas_str2sets.items(): 
                    if meta_struct_str == _pre_str: continue
                    if meta_struct_set == _pre_sets: continue 
                    if meta_struct_set.issubset(_pre_sets): 
                        # A 是 B 的子集，证明A是B的祖先
                        metas_relation_tree.add_relation(meta_struct_str, _pre_str, metas_cnt_orig[meta_struct_str], metas_cnt_orig[_pre_str])
                    elif _pre_sets.issubset(meta_struct_set):
                        metas_relation_tree.add_relation(_pre_str, meta_struct_str, metas_cnt_orig[_pre_str],metas_cnt_orig[meta_struct_str])
            # metas_relation_tree.export_to_graphviz(f'tmp/{C.current_bridge}_{chain}')
            for tx_instance in chain_data:
                if not len(tx_instance[0]): continue
                tx_hash = tx_instance[0]['hash'].lower()
                # if tx_hash not in gt_of_chain: continue
                # 如果使用了ancestor，则这个循环用于统计ancestor的出现次数(后代算在ancestor头上)
                meta_struct:list = _get_meta_structure(tx_instance[0], tx_instance[1:], False, flatten_keys_ignore)
                meta_struct_str = ';'.join(m for m in meta_struct)
                metas_cnt[meta_struct_str] += 1
                # all_descendants = metas_relation_tree.get_all_descendants(meta_struct_str)
                # max_desc = TL.find_n_keys_with_max_values( {k: metas_cnt_orig[k] for k in all_descendants}, 10)
                # for ancestor in all_descendants:
                #     metas_cnt[ancestor] += 1
            metas_clipped = list(itertools.filterfalse(lambda x: metas_cnt[x] <= self.QUERY_TIMES_FOR_VOTING, metas_cnt.keys()))
            trx_coverage_of_metas_clipped = sum(metas_cnt_orig[x] for x in metas_clipped)
            logging.info(f"Meta structure: chain {chain}: total {len(metas_cnt_orig)}, left {len(metas_clipped)}. Coverage: {trx_coverage_of_metas_clipped}/{len(chain_data)}={trx_coverage_of_metas_clipped/len(chain_data)}%")
            obj = {
                "keys": {x : metas_cnt[x] for x in metas_clipped},
                "tree": metas_relation_tree,
                'ignore': flatten_keys_ignore
            }
            self.meta_structures[chain] = obj 
            logging.info(f'Chain:{chain}, metas_clip:')
            for m in metas_clipped: logging.info(f"{metas_cnt_orig[m]}:{m}")

        with open(cache_path, 'w') as f:
            obj = {chain: {'keys':item['keys'], 'tree': item['tree'].to_json(), 'ignore': item['ignore'] } for chain, item in self.meta_structures.items()}
            json.dump(obj, f, indent=2)

    def go_for_voting(self, trx:dict, logs:T.List[dict], chain:str):
        """
            The first cycle, in order to get the chosen key(voted by multi queries )
        """
        if not len(trx) and not len(logs): 
            return -1, 'data_incomplete'
        # if trx['hash'].lower() not in self.GT[chain]: 
        #     # This instance should not be handled
        #     return -4, 'data_out_of_scope' 
        func_name = trx['pureFunctionName'] or trx['methodId']
        if C.current_bridge.startswith('Across'): 
            assert 0
            default_role = 'src'
        else:
            default_role = None
        role = DOC.function_name_to_role[C.current_bridge].get(func_name, default_role)
        ancestor = self._get_dependable_meta_structure(chain, trx, logs)
        if ancestor not in self.meta_structures[chain]['keys']: 
            # This is not a common case
            # Skip it 
            return -2, 'not_in_meta_structure'
        if role is None:
            return -3, 'role_not_found'
        with self.lock:
            left_times = self.first_query_times[chain][ancestor]
            input_id = "%s.%s.%s" % (chain, role, trx['hash'])
            
            if ancestor in self.voting_keys[chain][role] and len(self.voting_keys[chain][role][ancestor]) == len(self.PRESET_KEYS):
                # this has been voted
                return 1, ''

            if left_times > 0 and input_id not in self.LLM_record_indexes:
                self._wrap_ask_LLM(input_id, trx, logs, chain, role, ancestor)
            elif left_times <= 0:
                self.begin_voting(chain, role, ancestor)
            else:
                # This trx had been queried
                self.first_query_times[chain][ancestor] -= 1
            return 1, ''


    def go_for_voting_all_data(self, all_data):
        
        def _process_task(chain_data, chain_name:str):
            voting_statis = dd(lambda: 0)
            valid_trx = 0
            for one_trx in chain_data:
                if not len(one_trx[0]): continue
                a, _info  = self.go_for_voting(one_trx[0], one_trx[1:] if len(one_trx) > 1 else [], chain_name)
                if a > 0:
                    valid_trx += 1
                else:
                    voting_statis[_info] += 1

            print(f"{chain}, Valid:{valid_trx}")
            if len(voting_statis) > 0: 
                print(f"Voting Statis: {TL.defaultdict_to_dict(voting_statis)}")

        p = (C.paths().voting_results)
        if not os.path.exists(p):
            has_voting_res = False 
        else:
            has_voting_res = True
            with open(p) as f: 
                voting_res = json.load(f)
            TL.copy_dict_to_nested_defaultdict(voting_res, self.voting_keys)
        for chain in C.chains_to_pair:
            if has_voting_res and chain in self.voting_keys: continue
            _process_task(all_data[chain], chain)
        
        with open(p, 'w') as f: 
            json.dump(self.voting_keys, f, indent=1)

    def _pack_assistant_string_to_prompt(self, previous_resp:str, key_to_reask:str)->T.List[T.Dict[str,str]]:
        """
            When asking the second round, pack the previous answer as "assistant" role, and 
            tell LLM that it was wrong on the key `key_to_reask`
        """
        ret = []
        ret.append({'role':'assistant', 'content':previous_resp})
        ret.append({'role':'user', 'content':self.PROMPT['second'][0]['content'] % key_to_reask })
        return ret 


    def ask_LLM_base(self, message:T.List[T.Dict[str, str]], 
                input_id=None, force_reload:bool=False, add_to_record:bool=False, 
                N:int = 1, **kwargs):
        response = TL.ask_LLM(message, self.model_type)
        # count input tokens length if is not set
        response.usage.prompt_tokens = response.usage.prompt_tokens or sum(TL.num_tokens_from_string(a['content']) for a in message)
        # count output tokens length if is not set
        response.usage.completion_tokens = response.usage.completion_tokens or (TL.num_tokens_from_string(response.choices[0].message.content))
        # add to record if needed
        if add_to_record:
            self._add_LLM_record(response, input_id, **kwargs)
        return response


    def _get_transaction_hash_set_from_LLM_records(self, new_bridge_data={}):
        ret = {}
        if not len(self.LLM_records['chats']): 
            assert 0
        for chat in self.LLM_records['chats']:
            input_id = chat['input_id']
            t = input_id.split('.')
            chain, role, tx_hash = t[0:3]
            meta_md5 = chat['meta_structure']
            if chain not in ret: ret[chain] = {}
            if role not in ret[chain]: ret[chain][role] = {}
            if meta_md5 not in ret[chain][role]: ret[chain][role][meta_md5] = []
            if tx_hash not in ret[chain][role][meta_md5]:
                ret[chain][role][meta_md5].append(tx_hash)
        return ret

    def verify_voting_result(self, all_bridge_data:T.Dict[str, T.List[T.List[T.Dict]]]):

        if self._load_final_keys(): 
            pass

        new_bridge_data = self.reorganize_bridge_data_by_role_and_hash(all_bridge_data)
        asked_tx_hash_set = self._get_transaction_hash_set_from_LLM_records(new_bridge_data)
        GT = load_GT()

        def choose_trx_set_to_check(target_length:int, pre_tx_set:list, must_in_gt:bool=False):
            target_length = int(target_length)
            if len(choices_of_tx_hash) < target_length: 
                logging.error(f"Need to choose {target_length} from length of {len(choices_of_tx_hash)}, Returning Old records..")
                target_length = len(choices_of_tx_hash)
                # return pre_tx_set
            if must_in_gt:
                pre_chosen_tx_in_GT = list(a for a in pre_tx_set if a in GT[chain]['gt'])
            else:
                pre_chosen_tx_in_GT = pre_tx_set
            if len(pre_chosen_tx_in_GT) >= target_length : 
                return pre_chosen_tx_in_GT
            ret = copy.deepcopy(pre_chosen_tx_in_GT)
            while len(ret) < target_length: 
                new_set = random.choices(choices_of_tx_hash, k=int(target_length)-len(ret))
                if must_in_gt:
                    new_set = list(a for a in new_set if a in GT[chain]['gt'])
                ret.extend(new_set)
                ret = list(set(ret))
            return ret

        def _get_fields_sort_by_conf_scores(d:dict):
            ret = sorted(d.keys(), key = lambda x: d[x], reverse=True)
            return list( r for r in ret if (r and (r.startswith('log') or r.startswith('transaction')) ) )

        def _get_all_candidate_ans(fields_chosen_by_llm, chain, role, meta_structure):
            all_chosen_fields:T.List[str] = []
            scores_of_chosen_fields:T.Dict[str, float] = {}
            for _fld in itertools.product(*fields_chosen_by_llm):
                if len(set(_fld)) < len(_fld): continue # ! remove conflicts
                chs = dict(zip(C.PRESET_KEYS, _fld))
                chosen_fields_str = json.dumps(chs, sort_keys=True)
                all_chosen_fields.append(chosen_fields_str)
                _score = sum(self.voting_keys[chain][role][meta_structure][_k][chs[_k]] for _k in C.PRESET_KEYS )
                scores_of_chosen_fields[chosen_fields_str] = _score
            all_chosen_fields = sorted(all_chosen_fields, key = lambda x: scores_of_chosen_fields[x], reverse=True)
            return all_chosen_fields, scores_of_chosen_fields

        def _stratify_candidates(candds:list):
            layers = [5, 10, 30, 50, 100]
            ret = []
            idx = 0
            for ly in layers: 
                new_idx = int(len(candds) * ly / 100)
                ret.append(candds[idx : new_idx])
                idx = new_idx
            return ret

        def cache_checking_results(write_to_file:bool=False, obj=None):
            p = os.path.join(C.paths().candidate_chekcking_res)
            if write_to_file:
                with open(p, 'w') as f: 
                    json.dump(obj, f, indent=1)
                return 
            ret = dd(lambda: dd(lambda: dd(dict)))
            if os.path.exists(p):
                f = open(p)
                o = json.load(f)
                f.close() 
                for k1, v1 in o.items(): 
                    for k2, v2 in v1.items() : 
                        for k3, v3 in v2.items() :
                            ret[k1][k2][k3] = v3
            return ret

        def _wrap_check_one_candidate_for_tx_set(candidate_ans_str:str, target_length_to_check:int, check_pass_treshold:float, possible_answer_for_dst:str=''):
            """
                检查一个候选答案(`candidate_ans_str`)，一共检查`target_length_to_check`条记录。
                如果找到的比例大于`check_pass_treshold`, 则返回True, 否则返回False.
                `possible_answer_for_dst` 可用于加速在src上的匹配
            """
            out_of_scope_txs = []
            tx_set = []
            for checked_tx_hash in candidate_tx_checking_res[chain][role].get(candidate_ans_str, []):
                if checked_tx_hash not in tx_set: tx_set.append(checked_tx_hash)
            # checked_candidates_on_runtime[candidate_ans_str].update(tx_set)
            tx_set = choose_trx_set_to_check(target_length_to_check, tx_set, False)
            for tx_hash in tx_set:
                in_set = tx_hash in checked_candidates_on_runtime[candidate_ans_str]
                checked_candidates_on_runtime[candidate_ans_str].add(tx_hash)
                # TODO: Remove this
                # if '"to": "transaction.to"' in candidate_ans_str:
                #     candidate_ans_unfound[candidate_ans_str] += 1
                #     candidate_tx_checking_res[chain][role][candidate_ans_str][tx_hash] = 'unfound'
                #     continue
                if (tx_hash) in candidate_tx_checking_res[chain][role][candidate_ans_str]: 
                    # This has been checked before
                    if candidate_tx_checking_res[chain][role][candidate_ans_str][tx_hash] == 'out_of_scope':
                        out_of_scope_txs.append(tx_hash)
                    elif candidate_tx_checking_res[chain][role][candidate_ans_str][tx_hash] == 'found':
                        candidate_ans_found[candidate_ans_str] += (1 if not in_set else 0)
                    elif candidate_tx_checking_res[chain][role][candidate_ans_str][tx_hash] == 'unfound':
                        candidate_ans_unfound[candidate_ans_str] += (1 if not in_set else 0)
                    else: assert 0
                    continue
                tx_instance = new_bridge_data[chain][role][meta_structure][tx_hash]
                trx_in_our_gt_and_scope = role == 'dst' or (tx_hash in GT[chain]['gt'] and GT[chain]['gt'][tx_hash][0] in C.chains_to_pair)
                # The first answer provided by LLM 
                with my_timer() as _tm:
                    check_res = verify_one_result(
                        all_bridge_data, chain, tx_instance, 
                        json.loads(candidate_ans_str), role == 'src', 
                        self.choose_one_rpc(chain), possible_answer_for_dst)
                    cdd_used_time_for_each_trx.append(round(_tm.get_interval(), 3))
                all_found = all(v == 'found' for k, v in check_res.items())
                out_of_scope = any(v == 'out_of_scope' for k, v in check_res.items())
                if out_of_scope or not trx_in_our_gt_and_scope:
                    out_of_scope_txs.append(tx_hash)
                    candidate_tx_checking_res[chain][role][candidate_ans_str][tx_hash] = 'out_of_scope'
                    continue
                if not all_found:
                    candidate_ans_unfound[candidate_ans_str] += 1
                    candidate_tx_checking_res[chain][role][candidate_ans_str][tx_hash] = 'unfound'
                else:
                    candidate_ans_found[candidate_ans_str] += 1
                    candidate_tx_checking_res[chain][role][candidate_ans_str][tx_hash] = 'found'
            if len(out_of_scope_txs) >= len(tx_set) * 0.95:
                # * it is not out-of-scope, but rather wrong in `chain` field
                for tx in out_of_scope_txs:
                    candidate_tx_checking_res[chain][role][candidate_ans_str][tx] = 'unfound'
                candidate_ans_unfound[candidate_ans_str] += len(out_of_scope_txs)
                return False
            assert (candidate_ans_found[candidate_ans_str] + candidate_ans_unfound[candidate_ans_str] 
                    == len(tx_set) - len(out_of_scope_txs) )
            acc = (candidate_ans_found[candidate_ans_str]) / (len(tx_set) - len(out_of_scope_txs))
            logging.info("candidate accuracy: %.3f" % (acc) )
            if acc >= check_pass_treshold:
                return True
            else:
                return False

        def _pack_all_candidates_to_dict(voting_keys):
            fields_chosen_by_llm = []
            for _k in C.PRESET_KEYS:
                fields_chosen_by_llm.append(
                    _get_fields_sort_by_conf_scores(voting_keys[_k]))
            fields_chosen_by_llm_dict = dict(zip(C.PRESET_KEYS, fields_chosen_by_llm))
            return fields_chosen_by_llm_dict

        def _extract_all_values_by_candidates_in_one_trx(all_candidates:dict, tx_instance, ):
            map_of_values = { k:[] for k in C.PRESET_KEYS }
            new_all_candidates = {}
            for preset_key, candidates in all_candidates.items():
                new_all_candidates[preset_key] = [] 
                for candidate in candidates:
                    if candidate == 'transaction.to': continue # TODO: Remove it, only debugging
                    value_map = TL.extract_values_by_given_fields(tx_instance[0], tx_instance[1:], {preset_key:[candidate]})
                    try: 
                        value_map[preset_key]
                        new_all_candidates[preset_key].append(candidate)
                    except: 
                        continue
                    if preset_key in (C.PRESET_KEYS_TO, C.PRESET_KEYS_TOKEN): 
                        value = _normalize_address_type(value_map[preset_key])
                    else:
                        value = value_map[preset_key]
                    map_of_values[preset_key].append(value)
            return new_all_candidates, map_of_values

        def check_consistency_of_multi_candidates(tx_set, candidate_set:dict, key_name):
            for tx_hash in tx_set:
                tx_instance = new_bridge_data[chain][role][meta_structure][tx_hash]
                _, values = _extract_all_values_by_candidates_in_one_trx(candidate_set, tx_instance)
                values = values[key_name]
                last = None
                for v in values:
                    if last is None: last = v 
                    elif last != v: 
                        return False 
                    last = v
            return True 

        def get_diversity_of_multi_candidates(tx_set, candidate_set:dict, key_name):
            diversity = dd(set)
            candidate_list = candidate_set[key_name]
            for tx_hash in tx_set:
                tx_instance = new_bridge_data[chain][role][meta_structure][tx_hash]
                _, values = _extract_all_values_by_candidates_in_one_trx(candidate_set, tx_instance)
                values = values[key_name]
                for i,v in enumerate(values): 
                    diversity[candidate_list[i]].add(v)
            return diversity

        def check_consistency_and_diversity(all_candidates, possible_candidate):
            ret = {k:list(v) for (k, v) in possible_candidate.items()}
            additional_tx = new_bridge_data[chain][role][meta_structure]
            new_ret:T.Dict[str, list] = {}
            for k, ret_candidates in ret.items():
                new_ret[k] = copy.deepcopy(ret_candidates[:])
                if not len(new_ret[k]): 
                    new_ret[k] = copy.deepcopy(all_candidates[k])
                if len(new_ret[k]) <= 1:continue 
                if check_consistency_of_multi_candidates(additional_tx.keys(), {k:new_ret[k]}, k ): continue
                dv = get_diversity_of_multi_candidates(additional_tx.keys(), {k:new_ret[k]}, k)
                flag_all_unique = all(len(x) == 1 for x in dv.values())
                if k == C.PRESET_KEYS_TOKEN and flag_all_unique: 
                    # it is possible that all tokens are same in a business logic
                    continue
                for ret_candidate in ret_candidates:
                    if len(dv[ret_candidate]) == 1:
                        new_ret[k].remove(ret_candidate)
                if len(new_ret[k]) > 1:
                    new_ret[k] = [sorted(new_ret[k], key=lambda x: len(dv[x]), reverse=True)[0]]
            return new_ret

        def _wrap_check_all_dst_candidate_for_tx_set(all_candidates:dict, tx_set):
            possible_candidate = dd(set)
            _valid_tx_cnt = 0
            for tx_hash in tx_set:
                if _valid_tx_cnt > self.QUERY_TIMES_FOR_VOTING: break
                if tx_hash not in new_bridge_data[chain][role][meta_structure]: continue
                tx_instance = new_bridge_data[chain][role][meta_structure][tx_hash]
                map_of_values = { k:[] for k in C.PRESET_KEYS }
                possible_candidate_one_tx = dd(set)
                # 1. 根据$\{F_d^1,F_d^2,...\},...,\{F_{ts}^1,F_{ts}^2,...\}$ 分别提取$\{V_d^1, V_d^2...\},...,\{V_{ts}^1,V_{ts}^2,...\}$
                all_candidates, map_of_values = _extract_all_values_by_candidates_in_one_trx(all_candidates, tx_instance)
                
                # 2. 分析$I$的asset flow，获得所有可能的 Dest, amount, Token，记为$\{D_i,A_i, T_i\}$
                tracer = TCE.get_tracer()
                asset_flows = tracer.get_trace_common(chain, tx_hash, 'asset')
                flag_tested_af = False
                _valid_tx_cnt += 1
                # 3. 缩小A/T的范围： 将这些$\{A_i, T_i\}$ 与 $\{V_A^1, V_A^2...\},\{V_T^1,V_T^2,...\}$ 相互比较，这里的$V_A^x, V_T^x$ 分别代表LLM给出的A和T的候选在这个transaction中所对应的Value，如果数值相等，则认为找到了一个有可能是正确的A/T
                for af in asset_flows:
                    af = ed(af)
                    af.value = TL.save_value_int(af.value)
                    if af.type == 'cash': 
                        af_token_name, af_token_symbol, af_token_decimal = 'eth', 'eth', 18
                    elif af.type == 'token': 
                        # af.addr is the addr of Token
                        af_token_name, af_token_symbol, af_token_decimal = TL.get_token_name_symbol_decimals(af.addr, chain, TL.get_rpc_endpoints(chain, True, True))
                    af_value = af.value / (10**af_token_decimal)
                    if af_value == 0: continue
                    for (Didx, Dest),(Aidx, A), (Tidx, TOKEN) in itertools.product(
                            enumerate(map_of_values[C.PRESET_KEYS_TO]), enumerate(map_of_values[C.PRESET_KEYS_AMOUNT]), enumerate(map_of_values[C.PRESET_KEYS_TOKEN])):
                        if TL.save_value_int(Dest) != TL.save_value_int(af.to):
                            continue
                        try:
                            value_token_name, value_token_symbol, value_token_decimal = TL.get_token_name_symbol_decimals(TOKEN, chain, TL.get_rpc_endpoints(chain, True, True))
                        except: 
                            continue
                        flag_tested_af = True
                        A = TL.save_value_int(A) 
                        if not isinstance(A, (int,float)): continue
                        A = A / (10 ** value_token_decimal)
                        if not TL.check_whether_token_name_and_symbol_match(af_token_name, af_token_symbol, value_token_name, value_token_symbol, strict_mode=False): 
                            continue
                        if abs(A - af_value) / af_value > C.HYPERPARAMETER_FEE_RATE: continue
                        possible_candidate_one_tx[C.PRESET_KEYS_TO].add(all_candidates[C.PRESET_KEYS_TO][Didx])
                        possible_candidate_one_tx[C.PRESET_KEYS_AMOUNT].add(all_candidates[C.PRESET_KEYS_AMOUNT][Aidx])
                        possible_candidate_one_tx[C.PRESET_KEYS_TOKEN].add(all_candidates[C.PRESET_KEYS_TOKEN][Tidx])
                for preset_key in (C.PRESET_KEYS_TO, C.PRESET_KEYS_AMOUNT, C.PRESET_KEYS_TOKEN):
                    if not len(possible_candidate[preset_key]):
                        possible_candidate[preset_key] = copy.deepcopy(set(all_candidates[preset_key]))
                    new_set = possible_candidate[preset_key].intersection(possible_candidate_one_tx[preset_key])
                    if flag_tested_af:
                        possible_candidate[preset_key] = new_set
                
            # 这个步骤是将上面已经检验过的 Destination 作为跨链验证的根据，而不是用LLM给出的（因为这可能不太准确）                
            if flag_tested_af: all_candidates[C.PRESET_KEYS_TO] = list(possible_candidate[C.PRESET_KEYS_TO])
            # all_candidates, map_of_values = _extract_all_values_by_candidates_in_one_trx(all_candidates, tx_instance)
            addtional_tx = list(new_bridge_data[chain][role][meta_structure].keys())
            for tx_hash in addtional_tx[:min(100, len(addtional_tx))]:
                tx_instance = new_bridge_data[chain][role][meta_structure][tx_hash]
                # ! 这个地方注意，tx_instance 已经改变！
                _, map_of_values = _extract_all_values_by_candidates_in_one_trx(all_candidates, tx_instance)
                flag_tested_src_trx = False
                # 4. 根据 $\{F_c^1,F_c^2,...\}$ 到源链上寻找位于 $\{F_{ts}^1,F_{ts}^2,...\}$一定时间窗口内的交易，
                for (Cidx, CHAIN), (Didx, Dest_in_dst), (TsIdx, Ts) in itertools.product(
                        enumerate(map_of_values[C.PRESET_KEYS_CHAIN]), enumerate(map_of_values[C.PRESET_KEYS_TO]), enumerate(map_of_values[C.PRESET_KEYS_TIMESTAMP])):
                    src_chain_name = DOC.chain_id_to_name[C.current_bridge].get(CHAIN, '')
                    if src_chain_name == chain: continue 
                    if src_chain_name not in C.chains_to_pair: continue

                    src_trxs = filter_bridge_data_by_role_and_ts(new_bridge_data, src_chain_name, 'src', Ts, C.HYPERPARAMETER_TIME_WINDOW, reverse=True)
                    for src_trx in src_trxs:
                        # 4.1 提取$\{V_d^1, V_d^2...\},...,\{V_{ts}^1,V_{ts}^2,...\}$ （目标链上的）
                        flag_tested_src_trx = True
                        _meta_struct = self._get_dependable_meta_structure(src_chain_name, src_trx[0], src_trx[1:])
                        candidate_of_this_src_trx = self.final_chosen_keys[chain]['src'][_meta_struct] or _pack_all_candidates_to_dict(self.voting_keys[src_chain_name]['src'][_meta_struct])
                        _, map_of_dst_values = _extract_all_values_by_candidates_in_one_trx(candidate_of_this_src_trx, src_trx)
                        # 4.2 将源链与目标链上的 $V_{Di}$ 进行比较，筛选D
                        for a in map_of_dst_values[C.PRESET_KEYS_TO]:
                            if not isinstance(a, (str, bytes)) or not isinstance(Dest_in_dst, (str, bytes)): continue
                            if a.lower() != Dest_in_dst.lower(): continue 
                            possible_candidate_one_tx[C.PRESET_KEYS_CHAIN].add(all_candidates[C.PRESET_KEYS_CHAIN][Cidx])
                            possible_candidate_one_tx[C.PRESET_KEYS_TO].add(all_candidates[C.PRESET_KEYS_TO][Didx])
                            possible_candidate_one_tx[C.PRESET_KEYS_TIMESTAMP].add(all_candidates[C.PRESET_KEYS_TIMESTAMP][TsIdx])
                            break 
                for preset_key in (C.PRESET_KEYS_CHAIN, C.PRESET_KEYS_TO, C.PRESET_KEYS_TIMESTAMP):
                    if not len(possible_candidate[preset_key]):
                        possible_candidate[preset_key] = copy.deepcopy(set(all_candidates[preset_key]))
                    new_set = possible_candidate[preset_key].intersection(possible_candidate_one_tx[preset_key])
                    if flag_tested_src_trx:
                        possible_candidate[preset_key] = new_set

            return check_consistency_and_diversity(all_candidates, possible_candidate)

        def _wrap_check_all_src_candidate_for_tx_set(all_candidates:dict, tx_set):
            possible_candidate = dd(set)
            for tx_hash in tx_set[:min(5, len(tx_set))]:
                tx_instance = new_bridge_data[chain][role][meta_structure][tx_hash]
                map_of_values = { k:[] for k in C.PRESET_KEYS }
                possible_candidate_one_tx = dd(set)
                # 1. 根据$\{F_d^1,F_d^2,...\},...,\{F_{ts}^1,F_{ts}^2,...\}$ 分别提取$\{V_d^1, V_d^2...\},...,\{V_{ts}^1,V_{ts}^2,...\}$
                all_candidates, map_of_values = _extract_all_values_by_candidates_in_one_trx(all_candidates, tx_instance)
                
                # 2. 分析$I$的asset flow，获得所有可能的amount, Token，记为$\{A_i, T_i\}$
                tracer = TCE.get_tracer()
                asset_flows = tracer.get_trace_common(chain, tx_hash, 'asset')
                flag_tested_af = False
                # 3. 缩小A/T的范围： 将这些$\{A_i, T_i\}$ 与 $\{V_A^1, V_A^2...\},\{V_T^1,V_T^2,...\}$ 相互比较，这里的$V_A^x, V_T^x$ 分别代表LLM给出的A和T的候选在这个transaction中所对应的Value，如果数值相等，则认为找到了一个有可能是正确的A/T
                for af in asset_flows:
                    af = ed(af)
                    af.value = TL.save_value_int(af.value)
                    if af.type == 'cash': 
                        af_token_name, af_token_symbol, af_token_decimal = 'eth', 'eth', 18
                    elif af.type == 'token': 
                        # af.addr is the addr of Token
                        af_token_name, af_token_symbol, af_token_decimal = TL.get_token_name_symbol_decimals(af.addr, chain, TL.get_rpc_endpoints(chain, True, True))
                    af_value = af.value / (10**af_token_decimal)
                    for (Aidx, A), (Tidx, TOKEN) in itertools.product(enumerate(map_of_values[C.PRESET_KEYS_AMOUNT]), enumerate(map_of_values[C.PRESET_KEYS_TOKEN])):
                        try:
                            value_token_name, value_token_symbol, value_token_decimal = TL.get_token_name_symbol_decimals(TOKEN, chain, TL.get_rpc_endpoints(chain, True, True))
                        except:
                            continue
                        flag_tested_af = True
                        A = TL.save_value_int(A) / (10 ** value_token_decimal)
                        if not TL.check_whether_token_name_and_symbol_match(af_token_name, af_token_symbol, value_token_name, value_token_symbol, strict_mode=False): 
                            continue
                        if abs(A - af_value) / af_value > C.HYPERPARAMETER_FEE_RATE: continue
                        possible_candidate_one_tx[C.PRESET_KEYS_AMOUNT].add(all_candidates[C.PRESET_KEYS_AMOUNT][Aidx])
                        possible_candidate_one_tx[C.PRESET_KEYS_TOKEN].add(all_candidates[C.PRESET_KEYS_TOKEN][Tidx])
                for preset_key in (C.PRESET_KEYS_AMOUNT, C.PRESET_KEYS_TOKEN):
                    if not len(possible_candidate[preset_key]):
                        possible_candidate[preset_key] = copy.deepcopy(set(all_candidates[preset_key]))
                    new_set = possible_candidate[preset_key].intersection(possible_candidate_one_tx[preset_key])
                    if flag_tested_af and len(new_set):
                        possible_candidate[preset_key] = new_set
            
            addtional_tx = list(new_bridge_data[chain][role][meta_structure].keys())
            for tx_hash in addtional_tx[:min(100, len(addtional_tx))]:
                tx_instance = new_bridge_data[chain][role][meta_structure][tx_hash]
                # ! 这个地方需要注意，tx_instance 已经被改变了！！
                _, map_of_values = _extract_all_values_by_candidates_in_one_trx(all_candidates, tx_instance)
                flag_tested_dst_trx = False
                # 4. 根据 $\{F_c^1,F_c^2,...\}$ 到目标链上寻找位于 $\{F_{ts}^1,F_{ts}^2,...\}$一定时间窗口内的交易，
                for (Cidx, CHAIN), (Didx, Dest_in_src), (TsIdx, Ts) in itertools.product(
                        enumerate(map_of_values[C.PRESET_KEYS_CHAIN]), enumerate(map_of_values[C.PRESET_KEYS_TO]), enumerate(map_of_values[C.PRESET_KEYS_TIMESTAMP])):
                    dest_chain_name = DOC.chain_id_to_name[C.current_bridge].get(CHAIN, '')
                    if dest_chain_name == chain: continue 
                    if dest_chain_name not in C.chains_to_pair: continue

                    dst_trxs = filter_bridge_data_by_role_and_ts(new_bridge_data, dest_chain_name, 'dst', Ts, C.HYPERPARAMETER_TIME_WINDOW)
                    for dst_trx in dst_trxs:
                        # 4.1 提取$\{V_d^1, V_d^2...\},...,\{V_{ts}^1,V_{ts}^2,...\}$ （目标链上的）
                        flag_tested_dst_trx = True
                        _meta_struct = self._get_dependable_meta_structure(dest_chain_name, dst_trx[0], dst_trx[1:])
                        _, map_of_dst_values = _extract_all_values_by_candidates_in_one_trx(_pack_all_candidates_to_dict(self.voting_keys[dest_chain_name]['dst'][_meta_struct]), dst_trx)
                        # 4.2 将源链与目标链上的 $V_{Di}$ 进行比较，筛选D
                        for a in map_of_dst_values[C.PRESET_KEYS_TO]:
                            if not isinstance(a, (str, bytes)) or not isinstance(Dest_in_src, (str, bytes)): continue
                            if a.lower() != Dest_in_src.lower(): continue 
                            possible_candidate_one_tx[C.PRESET_KEYS_CHAIN].add(all_candidates[C.PRESET_KEYS_CHAIN][Cidx])
                            possible_candidate_one_tx[C.PRESET_KEYS_TO].add(all_candidates[C.PRESET_KEYS_TO][Didx])
                            possible_candidate_one_tx[C.PRESET_KEYS_TIMESTAMP].add(all_candidates[C.PRESET_KEYS_TIMESTAMP][TsIdx])
                            break 
                for preset_key in (C.PRESET_KEYS_CHAIN, C.PRESET_KEYS_TO, C.PRESET_KEYS_TIMESTAMP):
                    if not len(possible_candidate[preset_key]):
                        possible_candidate[preset_key] = copy.deepcopy(set(all_candidates[preset_key]))
                    new_set = possible_candidate[preset_key].intersection(possible_candidate_one_tx[preset_key])
                    if flag_tested_dst_trx and len(new_set):
                        possible_candidate[preset_key] = new_set

            # print(possible_candidate)
            return check_consistency_and_diversity(all_candidates, possible_candidate)

        #  ================
        filter1, filter2 = dd(lambda:0), dd(lambda:0)
        for role, chain in itertools.product(['src', 'dst'], self.voting_keys.keys()): 
            num_of_filter1, num_of_filter2 = 0, 0
            if not len(self.voting_keys[chain][role]): continue
            for meta_structure in self.voting_keys[chain][role]: 
                num_within_one_meta = 1
                for _k in self.voting_keys[chain][role][meta_structure]:
                    num_within_one_meta *= len(self.voting_keys[chain][role][meta_structure][_k])
                num_of_filter2 += num_within_one_meta
                # 从一个类别中随机拿一个
                tx_instance = list(new_bridge_data[chain][role][meta_structure].values())[0]
                flatten_keys = _get_meta_structure(tx_instance[0], tx_instance[1:], False, [], False)
                L = len(flatten_keys)
                comb = (L)*(L-4)*(L-3)*(L-2)*(L-1) / (5*4*3*2)
                num_of_filter1 += comb
            filter1[role] += num_of_filter1
            filter2[role] += num_of_filter2
            logging.info(
                f"{chain}, {role}, type_of_message:{len(self.voting_keys[chain][role])}, total_possible_before_LLM:{num_of_filter1}, average:{num_of_filter1 / len(self.voting_keys[chain][role])},total_possible_answer:{num_of_filter2},average:{num_of_filter2 / len(self.voting_keys[chain][role])}")
        logging.info(f"Total: {str(TL.defaultdict_to_dict(filter1))}, {str(TL.defaultdict_to_dict(filter2))}")

        for role, chain in itertools.product(['src', 'dst'], self.voting_keys.keys()): 
            for meta_structure in self.voting_keys[chain][role]:
                if (chain in self.final_chosen_keys and role in self.final_chosen_keys[chain] and 
                        meta_structure in self.final_chosen_keys[chain][role] and 
                        (len(self.final_chosen_keys[chain][role][meta_structure]))  ): 
                    continue
                choices_of_tx_hash = tuple(new_bridge_data[chain][role][meta_structure].keys())

                fields_chosen_by_llm_dict = _pack_all_candidates_to_dict(self.voting_keys[chain][role][meta_structure])
                _md5 = TL.md5_encryption(meta_structure)
                if _md5 not in asked_tx_hash_set[chain][role]: 
                    continue
                if role == 'src':
                    possible_answer_for_this_meta = _wrap_check_all_src_candidate_for_tx_set(fields_chosen_by_llm_dict, asked_tx_hash_set[chain][role][_md5])
                elif role == 'dst':
                    possible_answer_for_this_meta = _wrap_check_all_dst_candidate_for_tx_set(fields_chosen_by_llm_dict, asked_tx_hash_set[chain][role][_md5])

                final_ans = {}
                for preset_key, possible_answers in possible_answer_for_this_meta.items(): 
                    final_ans[preset_key] = possible_answers
                    if len(possible_answers) <= 1: continue
                    sorted_possible_answers = sorted(possible_answers, key=lambda x: self.voting_keys[chain][role][meta_structure][preset_key][x], reverse=True)
                    final_ans[preset_key] = sorted_possible_answers
                
                if len(final_ans):
                    self.final_chosen_keys[chain][role][meta_structure] = final_ans
                else:
                    pass
                self._save_final_keys()

    def _pack_assistant_str(self, last_chosen_key:dict, last_extrat_res:dict):
        """
            When re-asking LLM, we need to add (virtual) answer as "assistant" prompt, 
            such that can avoid LLM to answer the wrong key. 
        """
        assistant_str = {}
        for key in C.PRESET_KEYS_TO_PROMPT:
            LLM_resp_decoded = {key: {"chosen_key":last_chosen_key[key],"value":last_extrat_res[key]}}
            assistant_str.update(LLM_resp_decoded)
        return json.dumps(assistant_str)


def filter_bridge_data_by_role_and_ts(new_bridge_data, chain, role, timestamp, treshold:int=3600, reverse:bool=False):
    ret = []
    for meta_structure in new_bridge_data[chain][role]:
        for tx_hash, tx_instance in new_bridge_data[chain][role][meta_structure].items():
            trx = tx_instance[0]
            if not len(trx): continue
            ts = trx.get('timeStamp', 0) or trx.get('timestamp', 0)
            ts, timestamp = TL.save_value_int(ts), TL.save_value_int(timestamp)
            diff = (ts - timestamp) if not reverse else (timestamp - ts)
            if diff >= 0 and diff <= treshold: 
                ret.append(tx_instance)
    return sorted(ret, key=lambda x: TL.save_value_int(x[0].get('timestamp', False) or x[0].get('timeStamp')))


def check_to_token_amount(check_existance:bool, af, ext_tuples, af_type, src_chain, dst_chain, w3=None):
    if not (isinstance(ext_tuples[2], str) and isinstance(ext_tuples[0], str) and isinstance(ext_tuples[1], (int, float)) ): 
        return False
    if not check_existance:
        res1 = TL.save_value_int(af.to) == TL.save_value_int(ext_tuples[0]) and \
            ( af.value == ext_tuples[1] 
                # or abs( af.value - ext_tuples[1]) / af.value  < C.HYPERPARAMETER_FEE_RATE 
             )
        if af_type == 'cash': return res1
        return res1 and TL.save_value_int(ext_tuples[2]) == TL.save_value_int(af.addr)
    else:
        if not TL.save_value_int(af.to) == TL.save_value_int(ext_tuples[0]): return False
        if not af.value or not ext_tuples[1]: return False
        if af_type == 'cash':
            return abs(af.value - ext_tuples[1]) / af.value < C.HYPERPARAMETER_FEE_RATE
        if TL.save_value_int(ext_tuples[2]) == 0: 
            return False
        dst_name, dst_symbol, dst_decimal = TL.get_token_name_symbol_decimals(af.addr, dst_chain, TL.get_rpc_endpoints(dst_chain, True, True))
        src_name, src_symbol, src_decimal = TL.get_token_name_symbol_decimals(ext_tuples[2], src_chain, TL.get_rpc_endpoints(src_chain, True, True))
        if '' in (dst_name, dst_symbol, src_name, src_symbol):
            # One of the token info getting failed 
            return False
        dst_name, src_name = dst_name.strip(), src_name.strip() 
        if TL.check_whether_token_name_and_symbol_match(src_name, src_symbol,dst_name, dst_symbol,): 
            ext_value = ext_tuples[1] / (10 ** src_decimal)
            af_value = af.value / (10 ** dst_decimal)
            return (abs(af_value - ext_value) / af_value < C.HYPERPARAMETER_FEE_RATE)
        return False


def wrap_extract_from_trace(tx_instance, extract_res, 
        chosen_fields, chain_of_dst:str, check_existance, chain_of_src:str='', w3=None):
    tracer = TCE.get_tracer()
    ret = {a:'found' for a in C.PRESET_KEYS}
    try:
        ext_amount, ext_to, ext_token = (
            TL.save_value_int(extract_res[C.PRESET_KEYS_AMOUNT]), extract_res[C.PRESET_KEYS_TO], extract_res[C.PRESET_KEYS_TOKEN])
    except KeyError:
        logging.debug("check fail reason: `to`,`token`, `amount` extract failed")
        ret[C.PRESET_KEYS_AMOUNT] = ret[C.PRESET_KEYS_TO] = ret[C.PRESET_KEYS_TOKEN] = 'unfound'
        return ret # cannot extract certain fields and values from the answer given by LLM
    ext_to = _normalize_address_type(ext_to)
    ext_token = _normalize_address_type(ext_token)
    asset_flows = tracer.get_trace_common(chain_of_dst, tx_instance[0]['hash'], 'asset')
    for af in get_final_flows(asset_flows):
        af = ed(af)
        af.value = TL.save_value_int(af.value)
        if af.value in (0, 0.0):
            continue
        if check_to_token_amount(check_existance, af, 
                                    (ext_to, ext_amount, ext_token), af.type, chain_of_src, chain_of_dst, w3):
            return {a:'found' for a in C.PRESET_KEYS} # correct (in terms of native asset)
    if not check_existance: # check on dst. For src chain, we have logged in upstream function
        logging.debug("check fail reason: `to`,`token`, `amount` check fail")
    return {a:'unfound' for a in C.PRESET_KEYS}


def _filter_dst_chain_matching_trxs(tx_instance_set:list, possible_answer_for_dst:str, callback=None):
    """
        在为src上的trx寻找匹配的dst trx时，常常会找到非常多的候选trx，但是这些trx并不都来自src chain
        这个函数会筛选掉这些不来自于`src_chain`的trx，以大幅减少要检查的trx数量
    """
    if not len(possible_answer_for_dst): 
        return tx_instance_set
    ret = []
    possible_answer_for_dst = json.loads(possible_answer_for_dst)
    for tx_instance in tx_instance_set:
        a = TL.extract_values_by_given_fields(tx_instance[0], 
            tx_instance[1:] if len(tx_instance) > 1 else [], 
            {C.PRESET_KEYS_CHAIN: [possible_answer_for_dst[C.PRESET_KEYS_CHAIN]]}, True) 
        try:
            chain_id = a[C.PRESET_KEYS_CHAIN]
            if callback(chain_id): 
                ret.append(tx_instance)
        except KeyError: 
            ret.append(tx_instance)
    return ret

def _flatten_all_keys(d:dict, cur_key:str=''):
    tmp_ret = []
    ret = []
    for subkey in d.keys():
        value = d[subkey]
        if isinstance(value, dict):
            tmp_ret.extend(_flatten_all_keys(value, subkey))
        else:
            tmp_ret.append(subkey)
    if len(cur_key) and len(tmp_ret):
        for r in tmp_ret:
            ret.append(cur_key + '.' + r)
        return ret
    else:
        return tmp_ret


def _get_meta_structure(trx:dict, logs:T.List[dict], return_str:bool=False, str_filter:list=[], return_combined_keys:bool=True):
    if len(trx):
        func_name = trx.get('pureFunctionName', None) or trx.get('methodId') or trx['input'][:10]
        all_keys = _flatten_all_keys(trx['args'], 'transaction[%s].args' % func_name)
        transaction_combined = TL.group_strings_by_hierarchy(all_keys)
        combined_keys = [transaction_combined]
    else:
        combined_keys = ['transaction[none]()']
        all_keys = []
    for log_entry in logs:
        log_keys =  _flatten_all_keys(log_entry['args'], 'log[%s].args' % (log_entry['event']))
        log_keys_combine = TL.group_strings_by_hierarchy(log_keys)
        combined_keys.append(log_keys_combine)
        all_keys.extend(log_keys)
    combined_keys = sorted(list(a for a in (combined_keys) if a not in str_filter))
    if return_combined_keys:
        if return_str: 
            return ';'.join(combined_keys)
        return combined_keys
    all_keys = sorted(list(a for a in (all_keys) if a not in str_filter))
    if return_str: 
        return ';'.join(all_keys)
    return all_keys


def verify_one_result(all_bridge_data, chain_of_tx, tx_instance, chosen_fields:dict, check_existance:bool=True, w3=None, possible_answer_for_dst:str=''):
    chosen_fields_wrap = {a:[b] for a, b in chosen_fields.items()}
    extract_res = TL.extract_values_by_given_fields(tx_instance[0], tx_instance[1:] if len(tx_instance) > 1 else [], chosen_fields_wrap, True)
    if any(v is None for v in extract_res.values()): 
        # the provided fields are wrong here  
        logging.debug("check fail reason: extract fail from trx instance")
        return {a:'unfound' for a in C.PRESET_KEYS}
    ret = {a:'found' for a in C.PRESET_KEYS}
    try:
        chain_id = extract_res[C.PRESET_KEYS_CHAIN]
    except:
        ret[C.PRESET_KEYS_CHAIN] = 'unfound'
        logging.debug("check fail reason: `chain` field")
        return ret
    chain_name = DOC.chain_id_to_name[C.current_bridge].get(chain_id, None)
    if chain_name not in C.chains_to_pair: 
        ret[C.PRESET_KEYS_CHAIN] = 'out_of_scope'
        return ret # chain not in our pairing list

    if chain_name == chain_of_tx: 
        ret[C.PRESET_KEYS_CHAIN] = 'unfound'
        logging.debug("check fail reason: `chain` field")
        return ret 
    
    if check_existance:
        ts = tx_instance[0].get('timestamp') or tx_instance[0].get('timeStamp')
        treshold = C.HYPERPARAMETER_TIME_WINDOW 
        candidate_data = filter_bridge_data_by_role_and_ts(all_bridge_data, chain_name, 'dst', ts, treshold)
        candidate_data = _filter_dst_chain_matching_trxs(candidate_data, possible_answer_for_dst, lambda x: DOC.chain_id_to_name[C.current_bridge].get(x)==chain_of_tx)
        while len(candidate_data) > 50 and treshold >= 3600 / 2: # 20 minutes
            treshold = (treshold * 2) // 3
            candidate_data = filter_bridge_data_by_role_and_ts(all_bridge_data, chain_name, 'dst', ts, treshold)
            candidate_data = _filter_dst_chain_matching_trxs(candidate_data, possible_answer_for_dst, lambda x: DOC.chain_id_to_name[C.current_bridge].get(x)==chain_of_tx)
        i = 0
        for i, cdata in enumerate(TL.mid_for(candidate_data)):
            check_status = wrap_extract_from_trace(cdata, 
                        extract_res, chosen_fields, chain_name, check_existance, chain_of_tx, w3)
            all_found = all(v == 'found' for k, v in check_status.items())
            if all_found: 
                return check_status
        for k in (C.PRESET_KEYS_TO, C.PRESET_KEYS_TOKEN, C.PRESET_KEYS_AMOUNT): 
            ret[k] = 'unfound'
        if not len(candidate_data):
            logging.debug("check fail reason: timestamp error or data not found in time period")
        else:
            logging.debug("check fail reason: `to`,`token`, `amount` check fail")
        return ret
    else:
        return wrap_extract_from_trace(tx_instance, extract_res, 
                    chosen_fields, chain_of_tx, check_existance, '', w3)


def _normalize_address_type(s:str):
    if isinstance(s, str) and not all(c in string.printable for c in s):
        return TL.normalize_address_type(TL.byte_to_hex(s, True))
    return s


def _init_meta_structure_trees_from_file(filename):
    if os.path.exists(filename):
        with open(filename) as f: obj = json.load(f)
        meta_structures = {}
        for chain, chain_data in obj.items():
            meta_structures[chain] = {
                'keys': chain_data['keys'], 
                'tree': AncestorTree.from_data(chain_data['tree']),
                'ignore': chain_data['ignore']
            }
        return meta_structures
    else:
        meta_structures = {}
    return meta_structures


def calc_net_asset_flow(asset_flows:T.List[dict]):
    """
    给定一些asset flow，计算每个地址的纯转入/转出金额。
    
    @param: `asset_flow`: List[AssetFlow]
        where `AssetFlow` is a dict defined as: 
        {
            "from": str , // optional, an wallet or contract address, and could be empty string
            "to":str,  //  an wallet or contract address
            "type": str , // one of "cash" or "token"
            "addr": str ,  // if "type" == 'token', then this is the address of token
            "value": int
        }
    """
    net_flow = dd(int)
    for flow in asset_flows:
        from_addr = flow.get("from")
        to_addr = flow["to"]
        value = flow["value"]

        if from_addr:
            net_flow[from_addr] -= value
        net_flow[to_addr] += value
    return net_flow

def get_final_flows(asset_flows):
    af_for_types = dd(list)
    for af in asset_flows:
        if af['type'] == 'token' and af.get('symbol') == 'eth': 
            af_for_types['cash'].append(af)
        else:
            af_for_types[af['type']].append(af)
    ret = []
    for _t in af_for_types:
        ret.append(af_for_types[_t][-1])
    return ret

def load_GT():
    if os.path.exists(C.paths().gt): 
        with open(C.paths().gt) as f: 
            ret = json.load(f)
    else:
        ret = {'gt':{}, 'not_in':{}, 'fail':[]}
    return ret

def get_transaction_hash_in_gt_scope(GT:dict):
    ret = dd(set)

    
    for chain in GT:
        if chain not in C.chains_to_pair: continue
        gt_of_chain = GT[chain]['gt']
        for key, value in gt_of_chain.items():
            ret[chain].add(key.lower())
            if value[0] not in C.chains_to_pair: continue
            ret[value[0]].add(value[1].lower())

    return ret