import argparse
import logging
import os
from pathlib import Path

import torch

from guacamol.assess_distribution_learning import assess_distribution_learning
from guacamol.utils.helpers import setup_default_logger

from rnn_sampler import SmilesRnnSampler
from rnn_utils import load_rnn_model, set_random_seed
from smiles_rnn_generator import SmilesRnnGenerator

model_def = "model_final_0.532.json"
model_path = "model_final_0.532.pt"
device = "cuda"

model = load_rnn_model(model_def, model_path, device, copy_to_cpu=True)
generator = SmilesRnnGenerator(model=model, device=device)

generator.generate(10)