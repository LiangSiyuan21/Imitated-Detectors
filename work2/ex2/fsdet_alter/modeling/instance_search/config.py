from fsdet_alter.config import CfgNode as CN


def add_instance_search_config(cfg):
    _C = cfg
    _C.MODEL.INSTANCE_SEARCH = CN()
    _C.MODEL.INSTANCE_SEARCH.NAME = ""
    _C.MODEL.INSTANCE_SEARCH.CHANNEL_GATE_NAME = ""