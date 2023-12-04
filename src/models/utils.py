#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 20-Oct-2023
# 
# ---------------------------------------------------------------------------
# File contains utility functions for the models.
# ---------------------------------------------------------------------------

import torch

from pathlib import Path
from .basenet import BaseNet

def save_model(model: BaseNet, path: Path):

    return torch.save(model.state_dict(), path)

def load_model(model: BaseNet, path: Path):

    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    return model