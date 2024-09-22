# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .build import META_ARCH_REGISTRY, build_model  # isort:skip

# import all the meta_arch, so they will be registered
from .rcnn import GeneralizedRCNN, ProposalNetwork
from .retinanet import RetinaNet
from .semircnn import SemiGeneralizedRCNNv1, OSGeneralizedRCNNFPN, SemiGeneralizedRCNNv2, SemiHAP2SGeneralizedRCNNv1, SemiHAP2SGeneralizedRCNNv2, OSHAP2SGeneralizedRCNNFPN, OSGeneralizedRCNN50, OSGeneralizedRCNN101
from .osrcnn import OSGeneralizedRCNN
