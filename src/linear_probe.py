import os
from functools import partial
from dataclasses import dataclass
from typing import List
from tqdm import tqdm
import tyro

import numpy as np
import jax
import jax.numpy as jnp
from jax.random import split
import optax
from flax.training.train_state import TrainState

from substrates.gol import GameOfLife
from rollout import rollout_simulation

from model import ConvNet
import util

from main import ModelArgs

@dataclass
class Args:
    model: ModelArgs = ModelArgs()
    load_dir: str | None = None

def main(args: Args):
    pass

if __name__ == "__main__":
    main(tyro.cli(Args))