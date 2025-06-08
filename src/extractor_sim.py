
from src.extractor import ExtractorLLM
from src import tools as TL, config as C, secret as S, doc_info as DOC, tracer as TCE
import typing as T
from sentence_transformers import util, SentenceTransformer
import string

class ExtractorSim(ExtractorLLM):
    
    def __init__(self, sim_model, model_type:str='gpt-4o-mini', check_rpc:bool=True):
        self.encoder:SentenceTransformer = sim_model
        self.model:SentenceTransformer = self.encoder
        self.pre_encoded_queries = []
        super().__init__(model_type, check_rpc)

    def _nest_dict(self, d:dict, cur_key='')->T.List[str]:
        tmp_ret = []
        ret = []
        for subkey in d.keys():
            value = d[subkey]
            if isinstance(value, dict):
                tmp_ret.extend(self._nest_dict(value, subkey))
            else:
                value_type = TL.guess_para_type(value)
                tmp_ret.append(subkey + ' ' + value_type)
        if len(cur_key) and len(tmp_ret):
            for r in tmp_ret:
                ret.append(cur_key + ' ' + r)
            return ret
        else:
            return tmp_ret
    
    def _prepare_preset_queies(self, ):
        if len(self.pre_encoded_queries): 
            return self.pre_encoded_queries
        
        prompts = []
        for k in C.PRESET_KEYS:
            prompts.append(C.PRESET_KEYS_TO_PROMPT[k])
        self.pre_encoded_queries = self.encoder.encode(prompts)
        return self.pre_encoded_queries

    def _prepare_all_keys(self, trx:dict, logs:T.List[dict]):
        """
            Prepare a set of keys from transaction's function name, function arguments,
            and log event's names + arguments
        """
        # deal transactions 
        if len(trx):
            func_name = trx.get('pureFunctionName', None) or trx.get('methodId') or trx['input'][:10]
            trx_keys = _flatten_all_keys(trx, 'transaction %s' % func_name)
        else:
            trx_keys = []
        # deal logs
        log_keys = []
        for log_entry in logs:
            if not len(log_entry): continue
            one_log_keys = _flatten_all_keys(log_entry, 'log %s ' % (log_entry['event']))
            # one_log_keys = map(lambda x: "log %s %s" % (log_entry['event'] , x), one_log_keys)
            log_keys.extend(one_log_keys)

        all_keys_old:T.List[str] = trx_keys + log_keys
        # extend the semantics from template 
        # the keys follow the forms of: 
        # <source>(transaction/log), <name>, <arguments component1>[,<arguments component2>, ...]

        return all_keys_old

    def _convert_fields_to_dot(self, fields:str):
        """
            fields used in similarity search are splitted by space. 
            This function convert it to dot-connected
        """
        tmp = fields.split(' ')
        tmp = list(t for t in tmp if len(t))
        if tmp[0] == 'transaction': header = tmp[0] 
        else: header = "%s[%s]" % (tmp[0], tmp[1])
        return '.'.join( [header] + tmp[2:] )

    def _get_transaction_hash_set_from_LLM_records(self, new_bridge_data):
        ret = {}
        for chain in new_bridge_data:
            if chain not in ret:  ret[chain] = {}
            for role in new_bridge_data[chain]: 
                if role not in ret[chain]: ret[chain][role] = {}
                for meta_structure in new_bridge_data[chain][role]: 
                    _md5 = TL.md5_encryption(meta_structure)
                    if _md5 not in ret[chain][role]: 
                        ret[chain][role][_md5] = list(v for v in new_bridge_data[chain][role][meta_structure].keys()  )
        return ret

    def _encode_and_extract_from_given_keys_queries(
            self, trx:dict, logs:T.List[dict],
            candidate_keys:T.List[str], 
            query_values:T.List[T.Any] = ())->dict:
        """
            encode all keys from {name-value} set, and extract the target keys(Queries) from 
            the most close embeddings
        """
        all_keys_old = candidate_keys
        if not len(all_keys_old): 
            return {}
        all_keys = list(map(lambda x: _cut_varnames(x), all_keys_old))
        # for k in all_keys:
        #     self.all_tokens.add(' '.join(self.encoder.tokenizer.tokenize(k)))
        all_keys_embedding = self.encoder.encode(all_keys, show_progress_bar=False)
        # this stores tuple (extracted value, scores, the top5 result)
        chosen_keys = { k:dict() for k in C.PRESET_KEYS }
        
        for i, p in enumerate(query_values):
            all_keys_sim_scores = {all_keys[i] : float(util.cos_sim(p, x).squeeze()) for (i, x) in enumerate(all_keys_embedding)}
            sorted_keys = sorted(all_keys, key=lambda x: all_keys_sim_scores[x], reverse=True)
            top5_keys = sorted_keys[:min(5, len(sorted_keys))]
            # self._add_extracted_top5(C.PRESET_KEYS[i], top5_keys)
            top5_record:T.Dict[str, T.Any] = {} # from key to value
            for _key_in_top5 in top5_keys:
                index = all_keys.index(_key_in_top5)
                assert index != -1, "Error in getting keys when extracting:" + _key_in_top5
                _key_old = all_keys_old[index]
                _keys_dotted = self._convert_fields_to_dot(_key_old)
                chosen_keys[C.PRESET_KEYS[i]][_keys_dotted] = all_keys_sim_scores[_key_in_top5]
        return chosen_keys

    def _encode_and_extract(self, trx:dict, logs:T.List[dict])->dict:
        return self._encode_and_extract_from_given_keys_queries(
                trx, logs, 
                self._prepare_all_keys(trx, logs),self._prepare_preset_queies())

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

        left_times = self.first_query_times[chain][ancestor]
        input_id = "%s.%s.%s" % (chain, role, trx['hash'])
        
        if ancestor in self.voting_keys[chain][role] and len(self.voting_keys[chain][role][ancestor]) == len(self.PRESET_KEYS):
            # this has been voted
            return 1, ''

        # Query DL model 
        fields1 = self._encode_and_extract(trx, logs)
        self.voting_keys[chain][role][ancestor] = fields1
        return 1, ''


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


def _flatten_all_keys(d:dict, cur_key:str='', connnector:str=' '):
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
            ret.append(cur_key + connnector + r)
        return ret
    else:
        return tmp_ret