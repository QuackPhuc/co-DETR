"""
Detection heads for Co-DETR.

This module provides detection heads for object detection:
- CoDeformDETRHead: Main DETR detection head with Hungarian matching
- RPNHead: Region Proposal Network for anchor-based proposals
- RoIHead: RoI head for region-based detection
- ATSSHead: ATSS head with adaptive training sample selection
"""

from .detr_head import CoDeformDETRHead
from .rpn_head import RPNHead, AnchorGenerator
from .roi_head import RoIHead
from .atss_head import ATSSHead

__all__ = [
    "CoDeformDETRHead",
    "RPNHead",
    "AnchorGenerator",
    "RoIHead",
    "ATSSHead",
]
