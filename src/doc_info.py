"""
    Information extracted from documents.
    This step can be done by LLM or human effort
"""

function_name_to_role = {
    "Multichain-2023-7-7":{
        "anySwapOutUnderlying":"src",
        "anySwapOut":"src",
        "anySwapOutAndCall":"src",
        "anySwapOutNative":"src",
        "anySwapInAuto":"dst",
        "anySwapIn":"dst",
        "anySwapInUnderlyingAndExec":"dst",
    },
    'Stargate-2024-3-1':{
        "swap":"src",
        "0x9fbf10fc":"src",
        "validateTransactionProofV1":"dst",
        "validateTransactionProofV2":"dst",
        "0x252f7b01":"dst",
        "0x0508941e":"dst",
    },
    "Portal-2024-3-1":{
        "completeTransfer":'dst', 
        "transferTokens":'src'
    },
    "DLN-2024-3-1":{
        'fulfillOrder':'dst', 'strictlySwapAndCallDln':'dst', 
        'createSaltedOrder':'src', 'strictlySwapAndCall':'src'
    },
    "Across-2024-3-1":{
        'fillV3Relay':'dst',
        '0x2e378115':'dst',
    },
    "Celer":{
        'deposit':'src', 'send':'src', 'sendNative':'src', 
        'mint':'dst', 'relay':'dst'
    },
    "Multi":{
        "anySwapOutUnderlying":"src",
        "anySwapOut":"src",
        "anySwapOutAndCall":"src",
        "anySwapOutNative":"src",
        "anySwapInAuto":"dst",
        "anySwapIn":"dst",
        "anySwapInUnderlyingAndExec":"dst",
    },
    'Poly':{
        'lock':'src',
        'verifyHeaderAndExecuteTx':'dst',
    },
}


chain_id_to_name = {
    "Multichain-2023-6-1": {
        43114: 'Avalanche', 56: 'BSC', 1: 'Ethereum',250: 'Fantom', 137: 'Polygon'},
    "Stargate-2024-3-1":{
        101:'Ethereum', 102:"BSC", 106:"Avalanche", 109:"Polygon", 110:"Arbitrum", 112:"Fantom", 111:"Optimism", 184:"Base"
    },
    "Portal-2024-3-1":{
        2:"Ethereum", 4:"BSC", 5:"Polygon", 6:"Avalanche", 10:"Fantom", 30:"Base", 23:"Arbitrum", 24:"Optimism", 3:"Terra", 1:"Solana", 
    },
    "DLN-2024-3-1":{
        1:"Ethereum", 42161:"Arbitrum", 56:"BSC", 137:"Polygon", 8453:"Base", 10:"Optimism", 
    },
    'Across-2024-3-1':{
        1:"Ethereum", 42161:"Arbitrum", 56:"BSC", 137:"Polygon", 8453:"Base", 10:"Optimism", 
    },
    "Celer": {
        43114: 'Avalanche', 56: 'BSC', 1: 'Ethereum',250: 'Fantom', 137: 'Polygon'
    },
    "Multi": {
        43114: 'Avalanche', 56: 'BSC', 1: 'Ethereum',250: 'Fantom', 137: 'Polygon'
    },
    "Poly": {
        17: "Polygon", 6:"BSC", 2:"Ethereum"
    },
}

chain_currency = {
    "Ethereum":"ETH",
    "BSC":"BNB",
    "Base":"ETH",
    "Arbitrum":"ETH",
    "Optimism":"ETH",
    "Polygon":"MATIC"
}


# estimate the block range when getting event data

timestamp_anchor = {
    'Ethereum': {1699260371:18511840},
    'Avalanche':{1699261519: 37403796},
    'BSC':{1699261486: 33252052},
    'Fantom':{1699261618:70254725},
    'Polygon':{1699261555:49607172} 
}

seconds_per_block = {
    'Ethereum':[6,15],
    'BSC':[2, 7],
    "Polygon":[2,7],
    'Avalanche':[1,5],
    'Fantom':[1,9],
}

# See: https://stargateprotocol.gitbook.io/stargate/developers/v1-supported-networks-and-assets
# and: https://stargateprotocol.gitbook.io/stargate/v2-developer-docs/technical-reference/v2-supported-networks-and-assets#iota
STABLE_COIN_SYMBOLS = ['USDT', 'USDC', 'USDD', "DAI", 'FRAX', 'sUSD','LUSD','BUSD', 'USDC.e','USDbC', 'lzUSDC']

ETH_TOKEN_SYMBOLS = ['SGETH', 'mETH', 'eth']

def get_token_symbol_type(synmbol:str):
    if synmbol in STABLE_COIN_SYMBOLS: 
        return 'USD'
    if synmbol in ETH_TOKEN_SYMBOLS: 
        return 'eth'
    if not len(synmbol): 
        return 'err'
    return 'other'

def search_doc_variable_by_bridge_name(bridge_name:str, var_name:str):
    g = globals()
    for k, v in g[var_name].items():
        if k.startswith(bridge_name):
            return v
    return {}