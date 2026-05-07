"""SegFix: post-hoc boundary refinement for OCRNet on ADE20K (20%).

Importing this package side-effect-registers the custom transform,
segmentor, and metric in mmseg's registries. Configs that use SegFix
should include ``custom_imports = dict(imports=['segfix'])``.
"""

from .metric import SegFixOffsetMetric  # noqa: F401
from .model import SegFixOffsetModel  # noqa: F401
from .transforms import ComputeOffsetsFromSeg, PackSegFixInputs  # noqa: F401

__all__ = [
    'SegFixOffsetMetric',
    'SegFixOffsetModel',
    'ComputeOffsetsFromSeg',
    'PackSegFixInputs',
]
