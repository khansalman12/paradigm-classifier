"""Central config — seeds, model name, threshold."""

import os
import random

import numpy as np
import torch


class Config:

    RANDOM_SEED = 42
    MODEL_NAME = "all-MiniLM-L6-v2"
    CONFIDENCE_THRESHOLD = 0.28

    DOCUMENT_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "programming_types.md",
    )

    PARADIGM_LABELS = [
        "Functional Programming",
        "Procedural Programming",
        "Object-Oriented Programming",
    ]

    @classmethod
    def set_seeds(cls):
        """Lock down all random sources for reproducibility."""
        random.seed(cls.RANDOM_SEED)
        np.random.seed(cls.RANDOM_SEED)
        torch.manual_seed(cls.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(cls.RANDOM_SEED)
