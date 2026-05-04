"""Training task abstractions for timme.

This module provides task-based abstractions for training loops where each task
encapsulates both the forward pass and loss computation, returning a dictionary
with loss components and outputs for logging.
"""

from .task import TrainingTask
from ._helpers import resume_task_checkpoint, load_task_ema_checkpoint
from .classification import ClassificationTask
from .distillation import DistillationTeacher, LogitDistillationTask, FeatureDistillationTask
from .eval_task import ClassificationEvalTask, EvalTask, SSLEvalTask
from .lejepa import LeJEPATask, SIGReg
from .nepa import NEPATask
from .token_distillation import TokenDistillationTeacher, TokenDistillationTask

__all__ = [
    'TrainingTask',
    'resume_task_checkpoint',
    'load_task_ema_checkpoint',
    'ClassificationTask',
    'EvalTask',
    'ClassificationEvalTask',
    'SSLEvalTask',
    'DistillationTeacher',
    'LogitDistillationTask',
    'FeatureDistillationTask',
    'LeJEPATask',
    'SIGReg',
    'NEPATask',
    'TokenDistillationTeacher',
    'TokenDistillationTask',
]
