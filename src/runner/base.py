"""
Base Runner class for GRAM
Defines common interface for all runner implementations
"""

from abc import ABC, abstractmethod
import logging


class BaseRunner(ABC):
    """Base class for all runners"""

    def __init__(
        self,
        model_rec,
        model_gen,
        tokenizer,
        train_loader_id,
        train_loader_rec,
        valid_loader,
        device,
        args,
        rank=0,
    ):
        self.model_rec = model_rec
        self.model_gen = model_gen
        self.tokenizer = tokenizer
        self.train_loader_id = train_loader_id
        self.train_loader_rec = train_loader_rec
        self.valid_loader = valid_loader
        self.device = device
        self.args = args
        self.rank = rank

        # Common attributes
        self.cur_model_path = None
        self.best_model_path = None

    @abstractmethod
    def train_generator(self):
        """Train the generator model"""
        pass

    @abstractmethod
    def test(self, model_path):
        """Test the model"""
        pass

    def save_model(self, model, path):
        """Save model to path"""
        if self.rank == 0:
            logging.info(f"Saving model to {path}")
        # Implementation depends on specific runner

    def load_model(self, model, path):
        """Load model from path"""
        if self.rank == 0:
            logging.info(f"Loading model from {path}")
        # Implementation depends on specific runner
