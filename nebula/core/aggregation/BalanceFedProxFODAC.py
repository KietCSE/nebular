import gc
import torch
import math
import logging
import numpy as np

from nebula.core.aggregation.aggregator import Aggregator


class BalanceFedProxFODAC(Aggregator):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

        # Hyperparameters
        self.A = 0.7           # balance filtering constant
        self.K = 1.0           # decay factor
        self.a = 0.2           # weight between local and neighbors
        self.mu = 0.05         # FedProx regularization strength

        self.adj = kwargs.get("adjacency_matrix", None)
        self.prev_local_model = None
        self.prev_prox_center = None

        self.W = self._build_W_from_topology(self.adj)

        logging.info(f"[{self.__class__.__name__}] Initialized with A={self.A}, K={self.K}, a={self.a}, mu={self.mu}")

    def _sinkhorn_knopp(self, A, max_iter=1000, tol=1e-8):
        A = np.array(A, dtype=float)
        for _ in range(max_iter):
            A /= A.sum(axis=1, keepdims=True)
            A /= A.sum(axis=0, keepdims=True)
            if np.allclose(A.sum(axis=1), 1, atol=tol) and np.allclose(A.sum(axis=0), 1, atol=tol):
                break
        return A

    def _build_W_from_topology(self, adj):
        if adj is None:
            logging.warning(f"[{self.__class__.__name__}] No adjacency matrix provided. Using uniform weights.")
            return None

        A = np.maximum(adj, adj.T)
        np.fill_diagonal(A, 1)
        W = self._sinkhorn_knopp(A)
        W_dict = {(i, j): float(W[i, j]) for i in range(W.shape[0]) for j in range(W.shape[1])}
        logging.info(f"[{self.__class__.__name__}] Mixing matrix W built via Sinkhorn–Knopp normalization.")
        return W_dict

    def get_local_model(self, models):
        if self._addr not in models:
            raise ValueError(f"[{self.__class__.__name__}] Local model ({self._addr}) not found in models")
        return models[self._addr]

    def compute_prox_center(self, models):
        local_addr = self._addr
        local_model, _ = models[local_addr]

        if self.prev_prox_center is None or self.prev_local_model is None:
            logging.debug(f"[{self.__class__.__name__}] Initializing prox_center using neighbor average.")
            prox_center = self._compute_neighbor_average(models)
            self.prev_prox_center = {addr: params for addr, (params, _) in models.items()}
            self.prev_local_model = {k: v.clone() for k, v in local_model.items()}
            return prox_center

        with torch.no_grad():
            neighbor_centers = [
                (addr, self.prev_prox_center[addr])
                for addr in self.prev_prox_center if addr != local_addr
            ]

            if not neighbor_centers:
                return self._compute_neighbor_average(models)

            r_i_new = {}
            for layer in local_model:
                # Consensus part: sum_j w_ij * r_j(t)
                consensus_part = torch.zeros_like(local_model[layer])
                for addr, r_j in neighbor_centers:
                    try:
                        node_i = int(local_addr)
                        node_j = int(addr)
                    except:
                        node_i, node_j = local_addr, addr

                    if self.W:
                        w_ij = self.W.get((node_i, node_j), 0.0)
                    else:
                        w_ij = 1.0 / len(neighbor_centers)

                    consensus_part += w_ij * r_j[layer]

                # Dynamic part: (w_i^(t+1) - w_i^t)
                delta = local_model[layer] - self.prev_local_model[layer]
                r_i_new[layer] = consensus_part + delta

        self.prev_local_model = {k: v.clone() for k, v in local_model.items()}
        self.prev_prox_center[local_addr] = {k: v.clone() for k, v in r_i_new.items()}

        logging.debug(f"[{self.__class__.__name__}] Prox center updated via FODAC dynamic consensus.")
        return r_i_new

    def _compute_neighbor_average(self, models):
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
        try:
            current_round = self.engine.round + 1
            total_rounds = self.engine.total_rounds
        except AttributeError as e:
            logging.error(f"[{self.__class__.__name__}] Failed to get round info: {e}")
            return models

        prox_norm = math.sqrt(sum(torch.norm(param, p=2).item() ** 2 for param in prox_center.values()))
        threshold = self.A * math.exp(-self.K * current_round / total_rounds) * prox_norm

        filtered_models = {}
        for node_addr, (model_params, weight) in models.items():
            if node_addr == self._addr:
                continue

            distance = math.sqrt(sum(torch.norm(prox_center[layer] - model_params[layer], p=2).item() ** 2
                                     for layer in prox_center))
            if distance <= threshold:
                filtered_models[node_addr] = (model_params, weight)
                logging.debug(f"[{self.__class__.__name__}] Model {node_addr} accepted (dist={distance:.4f} ≤ thr={threshold:.4f})")
            else:
                logging.debug(f"[{self.__class__.__name__}] Model {node_addr} rejected (dist={distance:.4f} > thr={threshold:.4f})")

        return filtered_models

    def run_aggregation(self, models):
        super().run_aggregation(models)

        local_model, _ = self.get_local_model(models)
        prox_center = self.compute_prox_center(models)
        filtered_models = self.remove_malicious_models(models, prox_center)

        if not filtered_models:
            logging.debug(f"[{self.__class__.__name__}] No models passed filtering; returning local model.")
            return local_model

        filtered_models = list(filtered_models.values())
        S = len(filtered_models)
        accum = {layer: torch.zeros_like(param) for layer, param in local_model.items()}

        with torch.no_grad():
            for params, _ in filtered_models:
                for layer in accum:
                    accum[layer].add_(params[layer], alpha=1.0 / S)

            # result = a * wi + (1-a)*avg(wj) - mu*(wi - r_i)
            for layer in accum:
                prox_term = self.mu * (local_model[layer] - prox_center[layer])
                accum[layer] = (
                    self.a * local_model[layer]
                    + (1 - self.a) * accum[layer]
                    - prox_term
                )

        del models, filtered_models
        gc.collect()
        logging.info(f"[{self.__class__.__name__}] BalanceFedProxFODAC aggregation completed.")
        return accum
