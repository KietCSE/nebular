import gc
import torch
import math
import logging

from nebula.core.aggregation.aggregator import Aggregator

class BalanceFedProxPlus(Aggregator):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        # Hyperparameters 
        self.A = 1.5           # balance filtering constant
        self.K = 1.0           # decay factor
        self.a = 0.4           # weight between local and neighbors
        self.base_mu = 0.1     # base FedProx regularization strength
        # New params
        self.trim_ratio = 0.0  # trimmed-mean ratio (0.0..0.5). Nếu nhỏ dataset, set 0
        self.max_layer_norm = 10.0  # clip norm per-layer to reduce outliers
        logging.info(f"[{self.__class__.__name__}] Initialized with A={self.A}, K={self.K}, a={self.a}, base_mu={self.base_mu}, trim_ratio={self.trim_ratio}")

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

    def remove_malicious_models(self, models, prox_center):
        """
        Filter models based on prox-aware distance:
        ||wj - r_i|| <= A * exp(-K * t / T) * ||r_i||
        """
        try:
            current_round = self.engine.round + 1
            total_rounds = self.engine.total_rounds
        except AttributeError as e:
            logging.error(f"[{self.__class__.__name__}] Failed to get round info: {e}")
            return models

        # Compute ||r_i||
        prox_norm = 0.0
        for param in prox_center.values():
            prox_norm += torch.norm(param, p=2).item() ** 2
        prox_norm = math.sqrt(prox_norm)

        threshold = self.A * math.exp(-self.K * current_round / total_rounds) * prox_norm

        filtered_models = {}
        for node_addr, (model_params, weight) in models.items():
            if node_addr == self._addr:
                continue

            # Compute ||wj - r_i||
            distance = 0.0
            for layer in prox_center:
                diff = prox_center[layer] - model_params[layer]
                distance += torch.norm(diff, p=2).item() ** 2
            distance = math.sqrt(distance)

            if distance <= threshold:
                filtered_models[node_addr] = (model_params, weight)
                logging.debug(f"[{self.__class__.__name__}] Model {node_addr} accepted (dist={distance:.4f} ≤ thr={threshold:.4f})")
            else:
                logging.debug(f"[{self.__class__.__name__}] Model {node_addr} rejected (dist={distance:.4f} > thr={threshold:.4f})")

        return filtered_models

    def _elementwise_trimmed_mean(self, tensor_list, trim_ratio):
        """
        tensor_list: list of tensors with same shape
        trim_ratio: fraction to trim on each side (0..0.5). If 0 -> simple mean.
        Returns tensor of same shape.
        """
        if not tensor_list:
            raise ValueError("Empty tensor_list for trimmed mean")

        if trim_ratio <= 0.0:
            # weighted mean fallback handled elsewhere; here do simple mean
            acc = torch.zeros_like(tensor_list[0])
            for t in tensor_list:
                acc.add_(t)
            return acc / len(tensor_list)

        # stack to shape (N, *shape)
        stacked = torch.stack(tensor_list, dim=0)  # (N, ...)
        N = stacked.shape[0]
        k = int(math.floor(trim_ratio * N))
        if 2 * k >= N:
            # too aggressive trim -> just mean
            return torch.mean(stacked, dim=0)

        # compute element-wise sort across dim 0 by values
        # To keep memory controlled we sort along dim 0 using torch.topk twice:
        # But simpler: sort
        sorted_vals, _ = torch.sort(stacked, dim=0)  # (N, ...)
        trimmed = sorted_vals[k:N - k, ...]  # remove k smallest and k largest
        return torch.mean(trimmed, dim=0)

    def _clip_by_norm(self, tensor, max_norm):
        norm = torch.norm(tensor, p=2)
        if norm > max_norm:
            return tensor * (max_norm / (norm + 1e-12))
        return tensor

    def run_aggregation(self, models):
        """
        Improved FedProx-enhanced Balance aggregation:
        - compute prox center (same)
        - filter models using prox-aware condition (same)
        - aggregate with:
            * trimmed-mean (element-wise) when enough models
            * weighted-average fallback
            * per-layer clipping
            * adaptive mu based on average distance to prox_center
        """
        super().run_aggregation(models)

        local_model, _ = self.get_local_model(models)
        prox_center = self.compute_prox_center(models)
        filtered_models = self.remove_malicious_models(models, prox_center)

        if not filtered_models:
            logging.debug(f"[{self.__class__.__name__}] No models passed filtering; returning local model.")
            return local_model

        # Make list of (params, weight)
        filtered_items = list(filtered_models.items())  # [(addr, (params, weight)), ...]
        S = len(filtered_items)

        # Estimate attack strength from average distance to prox_center (used to adapt mu)
        avg_distance = 0.0
        for _, (params, _) in filtered_items:
            d = 0.0
            for layer in prox_center:
                d += torch.norm(prox_center[layer] - params[layer], p=2).item() ** 2
            avg_distance += math.sqrt(d)
        avg_distance /= S

        # compute prox_norm for scaling
        prox_norm = 0.0
        for param in prox_center.values():
            prox_norm += torch.norm(param, p=2).item() ** 2
        prox_norm = math.sqrt(prox_norm) + 1e-12

        # adaptive mu: if average distance relatively large, increase mu up to 5x
        attack_factor = max(0.0, (avg_distance / prox_norm) - 0.1)  # little margin
        adaptive_mu = self.base_mu * (1.0 + min(4.0, attack_factor))  # between base_mu and 5*base_mu
        logging.debug(f"[{self.__class__.__name__}] avg_dist={avg_distance:.6f}, prox_norm={prox_norm:.6f}, adaptive_mu={adaptive_mu:.6f}")

        # Prepare accum result per-layer
        accum = {}
        # For weighted average we sum weights
        total_weight = 0.0
        for _, (_, w) in filtered_items:
            try:
                total_weight += float(w)
            except Exception:
                total_weight += 1.0

        # If no valid total_weight, fallback to equal weights
        if total_weight <= 0:
            total_weight = float(S)

        with torch.no_grad():
            for layer in local_model:
                # collect tensors for this layer
                tensors = []
                weights = []
                for _, (params, w) in filtered_items:
                    tensors.append(params[layer])
                    try:
                        weights.append(float(w))
                    except Exception:
                        weights.append(1.0)

                # If enough models and trim_ratio > 0 -> elementwise trimmed mean
                if S >= 5 and self.trim_ratio > 0.0:
                    # trimmed mean ignores weights (robust). If you prefer weighted trimmed, more complex.
                    layer_avg = self._elementwise_trimmed_mean(tensors, self.trim_ratio)
                else:
                    # weighted mean
                    layer_avg = torch.zeros_like(local_model[layer])
                    for t, wt in zip(tensors, weights):
                        layer_avg.add_(t, alpha=wt / total_weight)

                # Clip the averaged layer to avoid single-layer huge values
                layer_avg = self._clip_by_norm(layer_avg, self.max_layer_norm)

                # FedProx proximal correction: prox_term = mu * (wi - r_i)
                prox_term = adaptive_mu * (local_model[layer] - prox_center[layer])
                # Clip prox_term as well
                prox_term = self._clip_by_norm(prox_term, self.max_layer_norm)

                # Final combination: a * wi + (1-a) * avg_wj - prox_term
                result_layer = (
                    self.a * local_model[layer] +
                    (1.0 - self.a) * layer_avg -
                    prox_term
                )

                # optional final clip
                result_layer = self._clip_by_norm(result_layer, self.max_layer_norm)

                accum[layer] = result_layer

        # cleanup
        del models, filtered_models
        gc.collect()
        logging.info(f"[{self.__class__.__name__}] BalanceFedProx aggregation completed (S={S}).")
        return accum
