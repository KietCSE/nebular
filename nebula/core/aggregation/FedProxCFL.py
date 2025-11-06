import gc
import torch
from nebula.core.aggregation.aggregator import Aggregator


class FedProxCFL(Aggregator):
    def __init__(self, config=None, mu=0.01, **kwargs):
        super().__init__(config, **kwargs)
        self.mu = mu

    def run_aggregation(self, models):
        super().run_aggregation(models)

        models = list(models.values())
        total_samples = float(sum(weight for _, weight in models))

        if total_samples == 0:
            raise ValueError("Total number of samples must be greater than zero.")

        last_model_params = models[-1][0]
        accum = {layer: torch.zeros_like(param, dtype=torch.float32) for layer, param in last_model_params.items()}

        with torch.no_grad():
            for model_parameters, weight in models:
                normalized_weight = weight / total_samples
                for layer in accum:
                    accum[layer].add_(
                        model_parameters[layer].to(accum[layer].dtype),
                        alpha=normalized_weight,
                    )

        try:
            prev_global = self.prev_model
        except AttributeError:
            prev_global = {layer: torch.zeros_like(param) for layer, param in accum.items()}

        with torch.no_grad():
            for layer in accum:
                accum[layer] -= self.mu * (accum[layer] - prev_global[layer])

        self.prev_model = {layer: accum[layer].clone().detach() for layer in accum}

        del models
        gc.collect()

        return accum
