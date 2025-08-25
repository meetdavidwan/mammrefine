"""
MammRefine source module
"""

from .model import Model
from .detect import detect_factual_consistency
from .detect_debate import detect_factual_consistency_debate
from .critique import generate_critiques
from .critique_debate import generate_critiques_debate
from .refine import refine_summaries
from .refine_debate import refine_summaries_debate
from .pipeline import run_pipeline

__all__ = [
    "Model",
    "detect_factual_consistency",
    "detect_factual_consistency_debate", 
    "generate_critiques",
    "generate_critiques_debate",
    "refine_summaries",
    "refine_summaries_debate",
    "run_pipeline"
] 