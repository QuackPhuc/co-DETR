"""
Matching Algorithms for Object Detection.

This module provides bipartite matching algorithms for DETR-based detectors:
- Hungarian Matcher: Optimal bipartite matching using Hungarian algorithm
- Simple Matcher: Lightweight IoU-based matching for baselines

The matchers assign predicted boxes to ground truth targets during training
to enable computation of detection losses.
"""

from codetr.models.matchers.hungarian_matcher import (
    HungarianMatcher,
    SimpleMatcher,
)

__all__ = [
    "HungarianMatcher",
    "SimpleMatcher",
]
