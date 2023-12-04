#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 20-Oct-2023
# 
# ---------------------------------------------------------------------------
# File contains base class for all networks.
# ---------------------------------------------------------------------------

import torch.nn as nn
import numpy as np


class BaseNet(nn.Module):
    """Base class for all neural networks."""

    def __init__(self):
        super().__init__()

    def forward(self, *input):
        """Forward pass logic

        Raises:
            NotImplementedError
        """
        raise NotImplementedError