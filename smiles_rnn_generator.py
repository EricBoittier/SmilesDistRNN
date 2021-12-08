from typing import List

from guacamol.distribution_matching_generator import DistributionMatchingGenerator

from rnn_model import SmilesRnn
from rnn_sampler import SmilesRnnSampler


class SmilesRnnGenerator(DistributionMatchingGenerator):
    """
    Wraps SmilesRnn in a class satisfying the DistributionMatchingGenerator interface.
    """

    def __init__(self, model: SmilesRnn, device: str) -> None:
        self.model = model
        self.device = device

    def generate(self, number_samples: int, max_seq_len=100) -> List[str]:
        sampler = SmilesRnnSampler(device=self.device)
        return sampler.sample(model=self.model, num_to_sample=number_samples, max_seq_len=max_seq_len)
