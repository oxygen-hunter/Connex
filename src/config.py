# -*- coding:utf-8 -*-

import os
from easydict import EasyDict as ed

# current_bridge = 'Multichain-2023-7-7'
# chains_to_pair = ['Ethereum', 'BSC', 'Avalanche', 'Polygon', 'Fantom']

# current_bridge = 'OptimismBridge-2023-7-14'
# chains_to_pair = ['Ethereum', 'Optimism']

# current_bridge = 'Multichain-2023-10-26'
# chains_to_pair = ['Ethereum', 'BSC', 'Avalanche', 'Polygon', 'Fantom']

# current_bridge = 'OptimismBridge-2023-10-29'
# chains_to_pair = ['Ethereum', 'Optimism']

# current_bridge = 'Stargate-2023-11-6'
# chains_to_pair = ['Ethereum', 'BSC', 'Avalanche', 'Polygon', 'Fantom']


# done by lixiao
date_after = '2024-3-1'
date_before = '2024-3-11'
# date_after = '2024-3-1'
timing = 'earliest' # earliest与date_after配合使用，收取某个日期后最早的require条tx。latest收取截止目前最新的require条tx
tx_num_require = 3000 # 所需tx数量
# current_bridge = 'Across' + '-' + date_after
# current_bridge = 'Stargate' + '-' + date_after
# current_bridge = 'Portal' + '-' + date_after
current_bridge = 'DLN' + '-' + date_after
# chains_to_pair = ['Ethereum', 'Arbitrum', 'Base', 'Optimism', 'BSC']
chains_to_pair = ['Ethereum','Arbitrum', 'Base', 'Optimism', 'BSC', 'Avalanche','Polygon', 'Fantom']

num_threads = 10

answers_wanted = 1
force_proxy = False
force_reload = False

HYPERPARAMETER_TIME_WINDOW = 7200
HYPERPARAMETER_FEE_RATE = 0.2

MAX_CONTEXT_SIZE = 128000
TRESHOLD_OF_INVALID_ANSWER_RATE = 0.5


Transfer_abi = {'anonymous': False, 'inputs': [{'indexed': True, 'internalType': 'address', 'name': 'from', 'type': 'address'}, {'indexed': True, 'internalType': 'address', 'name': 'to', 'type': 'address'}, {'indexed': False, 'internalType': 'uint256', 'name': 'value', 'type': 'uint256'}], 'name': 'Transfer', 'type': 'event'}

def paths(LLM_sub_dir:str=''):
    if globals().get('config_paths', False):
        return globals().get('config_paths')
    p = ed({
    # general
    'abis': os.path.join('data', 'abis.json'),
    "signatures":os.path.join('data', 'signatures.json'),
    "event_signatures":os.path.join('data', 'signatures_event.json'),
    "tokens": os.path.join('data', 'tokens.json'),
    'logs':os.path.join('logs'),
    'tx_requirements': os.path.join('data', 'tx_requirements.json'),
    'rpc_endpoints': os.path.join('data', 'rpc_endpoints.json'),
    'etherscan_like_api_keys': os.path.join('data', 'etherscan_like_api_keys.json'),
    'etherscan_like_api_urls': os.path.join('data', 'etherscan_like_api_urls.json'),
    'proxy': os.path.join('data', 'proxy.json'),
    'handmade_token_abi': os.path.join('data', 'handmade_token_abi.json'),
    "tokenizer":os.path.join('model', 'oobabooga', 'llama-tokenizer', 'tokenizer.json'), 
    "model":{
        "sentence_bert":os.path.join('model', 'sentence-transformers', 'all-MiniLM-L6-v2'), 
        "bge":os.path.join('model', 'BAAI', 'bge-base-en-v1.5'), 
    },
    "LLM_prompt":os.path.join('data', 'LLM_template.json'), 
    "LLM_config_dir":os.path.join('data', 'LLM_request_config'),
    "LLM_prompt_score":os.path.join('data', 'LLM_template_score.json'), 
    "LLM_benchmark":os.path.join('data','LLM_benchmark'), 
    "running_config":os.path.join('data', 'running_config'),

    # bridge specific data
    'bridge_tx_data': os.path.join('data', 'bridge_data', current_bridge, 'tx'),
    'bridge_log_data': os.path.join('data','bridge_data', current_bridge, 'log'),
    # bridge specific results
    'bridge_results': os.path.join('results', current_bridge),
    "decoded_data":os.path.join('results', current_bridge, 'decoded_data.json'),
    "diversed_fields":os.path.join('results', current_bridge, 'divesed_fields.json'),
    "chosen_keys":os.path.join('results', current_bridge, LLM_sub_dir, 'chosen_keys.json'),
    "candidate_chekcking_res":os.path.join('results', current_bridge, LLM_sub_dir, 'candidate_chekcking_res.json'),
    "unfound_trxs":os.path.join('results', current_bridge, LLM_sub_dir, 'unfound_trxs.json'),
    "meta_structure":os.path.join('results', current_bridge, 'meta_structure.json'),
    "key_info":os.path.join('results', current_bridge, LLM_sub_dir, 'chains_key_info.json'),
    "trace_cache": os.path.join('results', current_bridge, 'trace_cache.json'),
    "key_info_rule":os.path.join('results', current_bridge, 'chains_key_info_rule.json'),
    "extracted_tokens":os.path.join('results', current_bridge, 'extracted_tokens.txt'),
    "extract_statis":os.path.join('results', current_bridge, 'extract_statis.json'),
    "pair_res":os.path.join('results', current_bridge, LLM_sub_dir, 'pairing_res.json'), 
    "statis":os.path.join('results', current_bridge, LLM_sub_dir, 'statis.json'),
    "statis_excel":os.path.join('results', current_bridge, LLM_sub_dir, 'statis.xlsx'),
    "TP":os.path.join('results', current_bridge, LLM_sub_dir, 'TP.json'),
    "FP":os.path.join('results', current_bridge, LLM_sub_dir, 'FP.json'),
    "FN":os.path.join('results', current_bridge, LLM_sub_dir, 'FN.json'),
    "TP_plus":os.path.join('results', current_bridge, LLM_sub_dir, 'TP_plus.json'),
    "FP_plus":os.path.join('results', current_bridge, LLM_sub_dir, 'FP_plus.json'),
    "FN_plus":os.path.join('results', current_bridge, LLM_sub_dir, 'FN_plus.json'),
    "FN_rules":os.path.join('results', current_bridge, LLM_sub_dir, 'FN_rules.json'),
    "analysis_TP":os.path.join('results', current_bridge, LLM_sub_dir, 'TP_analysis.json'),
    "analysis_FP":os.path.join('results', current_bridge, LLM_sub_dir, 'FP_analysis.json'),
    "analysis_FN_token":os.path.join('results', current_bridge, LLM_sub_dir, 'analysis_FN_token.json'),
    "analysis_FN_token_excel":os.path.join('results', current_bridge, LLM_sub_dir, 'analysis_FN_token.xlsx'),
    "gt":os.path.join('results', current_bridge, 'gt.json'),
    "gt_from_tx": os.path.join('results', current_bridge, 'gt_from_tx.json'),
    "gt_from_api": os.path.join('results', current_bridge, 'gt_from_api.json'),
    "gt_classified_by_chain": os.path.join('results', current_bridge, 'gt_classified_by_chain.json'),
    "paired_tokens_from_gt": os.path.join('results', current_bridge, 'paired_tokens_from_gt.json'),
    "LLM_log":os.path.join('results', current_bridge, LLM_sub_dir, 'LLM_records.json'),
})
    globals()['config_paths'] = p 
    return p

def reset_paths(*args):
    del globals()['config_paths']
    paths(*args)


def get_globals(name:str, _default=None):
    return globals().get(name, _default)

def set_globals(name:str, value):
    globals()[name] = value

PRESET_KEYS_TO = 'to'
PRESET_KEYS_TIMESTAMP = 'timestamp'
PRESET_KEYS_TOKEN = 'token'
PRESET_KEYS_TOKEN_TYPE = 'token_type'
PRESET_KEYS_CHAIN = 'chain'
PRESET_KEYS_AMOUNT = 'amount'

PRESET_KEYS_TO_PROMPT = {
    PRESET_KEYS_TO:"target address of this cross chain transaction(\"to\" for short)",
    PRESET_KEYS_TOKEN:"the address of swapped token(\"token\" for short)",
    # PRESET_KEYS_TOKEN_TYPE:"the type of swapped token(\"token_type\" for short), choose from [\"native currency\", \"token by smart contract\"]",
    PRESET_KEYS_AMOUNT:"the amount of swapped token(\"amount\" for short)",
    PRESET_KEYS_CHAIN:"the {role} chain id(\"chain\" for short)",
    PRESET_KEYS_TIMESTAMP:"the timestamp of this transaction(\"timestamp\" for short)"
}

PRESET_KEYS = tuple(PRESET_KEYS_TO_PROMPT.keys())