# -*- coding:utf-8 -*-

from .tools import *
import json 
from easydict import EasyDict as ed

class Matcher():
    def __init__(self) -> None:
        pass


class MatcherMultichain(Matcher):

    API_URL = 'https://scanapi.multichain.org/v3/tx/'

    def __init__(self) -> None:
        """
            Multichain的匹配。给定一个source transaction hash, 返回其dest transaction hash
        """
        super().__init__()

    def go(self, src_tx_hash: str)->str:
        tx = src_tx_hash.lower()
        resp = get_url(self.API_URL + tx)
        obj = ed(json.loads(resp))
        # 3 status: Success, Failure, Pending
        # Success: return obj.msg.swaptx
        # Failure/Pending: obj.msg don't have attribute 'swaptx'
        if obj.msg != 'Success' or 'swaptx' not in obj.msg:
            return ''
        return obj.info.swaptx.lower()

