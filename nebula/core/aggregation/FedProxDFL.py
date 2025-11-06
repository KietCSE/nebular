import gc
import torch
import math
import logging
from nebula.core.aggregation.aggregator import Aggregator


class FedProxDFL(Aggregator):

    def __init__(self, config=None, mu=0.5, alpha=0.2, **kwargs):
        super().__init__(config, **kwargs)
        self.mu = mu        # Proximal strength
        self.alpha = alpha  # weight giữa local và neighbor
        logging.info(f"[{self.__class__.__name__}] Initialized with mu={self.mu}, alpha={self.alpha}")

    def get_local_model(self, models):
        if self._addr not in models:
            raise ValueError(f"[{self.__class__.__name__}] Local model ({self._addr}) not found in models")
        return models[self._addr]

    def compute_prox_center(self, models):
        """
        Compute r_i = average of all neighbor models (excluding self)
        """
        local_addr = self._addr
        neighbor_models = [params for addr, (params, _) in models.items() if addr != local_addr]
        if not neighbor_models:
            return self.get_local_model(models)[0]

        with torch.no_grad():
            r_i = {}
            num_neighbors = len(neighbor_models)
            for layer in neighbor_models[0]:
                r_i[layer] = torch.zeros_like(neighbor_models[0][layer])
                for model in neighbor_models:
                    r_i[layer] += model[layer] / num_neighbors
        return r_i

    def run_aggregation(self, models):
        """
        FedProx DFL Aggregation:
        w_i ← α·w_i + (1−α)·r_i − μ·(w_i − r_i)
        """
        super().run_aggregation(models)

        local_model, _ = self.get_local_model(models)
        prox_center = self.compute_prox_center(models)

        with torch.no_grad():
            updated_model = {}
            for layer in local_model:
                prox_term = self.mu * (local_model[layer] - prox_center[layer])
                updated_model[layer] = (
                    self.alpha * local_model[layer] +
                    (1 - self.alpha) * prox_center[layer] -
                    prox_term
                )

        del models
        gc.collect()
        logging.info(f"[{self.__class__.__name__}] FedProx DFL aggregation completed.")
        return updated_model