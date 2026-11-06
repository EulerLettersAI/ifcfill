"""
ifcfill — Transform tabular data into Integer, Float and Categorical (IFC)
variables and fill/impute missing data.
"""

from ._selection import ModeSelection, missingness_dependence_gap, select_mode
from ._transformer import IFCTransformer

__version__ = "0.6.0"
__all__ = [
    "IFCTransformer",
    "ModeSelection",
    "missingness_dependence_gap",
    "select_mode",
]
