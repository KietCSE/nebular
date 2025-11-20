import gc
import torch
import math
import logging
import numpy as np
from copy import deepcopy

from nebula.core.aggregation.aggregator import Aggregator


class BalanceFedProxFODAC(Aggregator):

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

        # Hyperparameters
        self.A = float(kwargs.get("A", 1.5))       # balance filtering constant
        self.K = float(kwargs.get("K", 1.0))       # decay factor
        self.a = float(kwargs.get("a", 0.4))      # weight giữa local và neighbor
        self.base_mu = float(kwargs.get("mu", 0.1))    # base FedProx regularization
        self.clip_norm = float(kwargs.get("clip_norm", 10.0))

        # topology
        self.adj = kwargs.get("adjacency_matrix", None)
        self.W = self._build_W(self.adj)

        # FODAC states
        self.x = None          # dynamic consensus state
        self.r_prev = None     # previous reference (local model)
        self.prev_local = None # previous local model
        self.device = torch.device(kwargs.get("device", "cpu"))

        logging.info(f"[{self.__class__.__name__}] Init A={self.A}, K={self.K}, a={self.a}, base_mu={self.base_mu}, clip_norm={self.clip_norm}")

    def _build_W(self, adj):
        if adj is None:
            return None
        A = np.array(adj, dtype=float)
        A = np.maximum(A, A.T)
        np.fill_diagonal(A, 1)
        # guard divide by zero
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        W = A / row_sums
        return W

    def _clone_state(self, state):
        return {k: v.detach().clone().to(self.device) for k, v in state.items()}

    def _zeros_like_state(self, state):
        return {k: torch.zeros_like(v) for k, v in state.items()}

    def _l2_distance(self, s1, s2):
        total = 0.0
        for k in s1:
            diff = s1[k].float() - s2[k].float()
            total += float(torch.norm(diff, p=2).item() ** 2)
        return math.sqrt(total)

    def compute_prox_center(self, models):
        local_addr = self._addr
        local_state, _ = models[local_addr]
        local_state = self._clone_state(local_state)

        # init lần đầu
        if self.x is None or self.r_prev is None:
            self.x = self._clone_state(local_state)
            self.r_prev = self._clone_state(local_state)
            self.prev_local = self._clone_state(local_state)
            return deepcopy(self.x)

        # compute Δr_i(t)
        delta_r = {k: (local_state[k] - self.r_prev[k]).detach().clone() for k in local_state}

        # neighbor consensus
        addrs = list(models.keys())
        neighbors = [addr for addr in addrs if addr != local_addr]
        delta_consensus = self._zeros_like_state(self.x)

        with torch.no_grad():
            for nb in neighbors:
                nb_state, _ = models[nb]
                nb_state = {k: v.to(self.device) for k, v in nb_state.items()}
                if self.W is not None:
                    try:
                        idx_i = addrs.index(local_addr)
                        idx_j = addrs.index(nb)
                        w_ij = float(self.W[idx_i, idx_j])
                    except ValueError:
                        w_ij = 1.0 / max(1, len(neighbors))
                else:
                    w_ij = 1.0 / max(1, len(neighbors))
                for k in delta_consensus:
                    delta_consensus[k] += w_ij * (nb_state[k] - self.x[k])

            # update x
            x_new = {}
            for k in self.x:
                x_new[k] = (self.x[k] + delta_consensus[k] + delta_r[k]).detach().clone()

            self.x = deepcopy(x_new)
            self.r_prev = deepcopy(local_state)
            self.prev_local = deepcopy(local_state)

        return deepcopy(self.x)

    def remove_malicious_models(self, models, prox_center):
        """
        Filter models based on prox-aware distance:
        ||wj - r_i|| <= A * exp(-K * t / T) * ||r_i||
        (UNCHANGED)
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

    def _clip_tensor(self, t, max_norm):
        if max_norm is None or max_norm <= 0:
            return t
        norm = torch.norm(t).item()
        if norm > max_norm:
            return t * (max_norm / (norm + 1e-12))
        return t

    def run_aggregation(self, models):
        super().run_aggregation(models)

        # local clone on device
        local_model, _ = models[self._addr]
        local_model = self._clone_state(local_model)

        # 1. prox center via FODAC (unchanged)
        prox_center = self.compute_prox_center(models)

        # 2. filter theo balance gốc (unchanged)
        filtered = self.remove_malicious_models(models, prox_center)

        if not filtered:
            logging.debug(f"[{self.__class__.__name__}] No valid neighbor; return local model.")
            return deepcopy(local_model)

        # 3. compute weighted average of neighbors (respect weight and topology W)
        # Build list of neighbors' (state, weight)
        addrs = list(models.keys())
        S = len(filtered)
        accum = self._zeros_like_state(local_model)

        # Prepare weights: consider both provided weight and topology W (if available)
        raw_weights = []
        states = []
        for nb_addr, (nb_state, nb_w) in filtered.items():
            # ensure tensor device
            nb_state = {k: v.to(self.device) for k, v in nb_state.items()}
            states.append(nb_state)
            try:
                w_val = float(nb_w)
            except Exception:
                w_val = 1.0
            # combine with topology weight if W provided
            if self.W is not None:
                try:
                    idx_i = addrs.index(self._addr)
                    idx_j = addrs.index(nb_addr)
                    topo_w = float(self.W[idx_i, idx_j])
                except ValueError:
                    topo_w = 1.0 / max(1, S)
                combined = w_val * topo_w
            else:
                combined = w_val
            raw_weights.append(max(0.0, combined))

        total_w = sum(raw_weights)
        if total_w <= 0:
            # fallback equal weights
            raw_weights = [1.0 for _ in raw_weights]
            total_w = float(len(raw_weights))

        # accumulate weighted average
        with torch.no_grad():
            for s, w in zip(states, raw_weights):
                for k in accum:
                    accum[k].add_(s[k], alpha=(w / total_w))

            # compute average distance to prox_center to adapt mu
            avg_distance = 0.0
            for s in states:
                d_sq = 0.0
                for k in prox_center:
                    d_sq += float(torch.norm(prox_center[k] - s[k], p=2).item() ** 2)
                avg_distance += math.sqrt(d_sq)
            avg_distance = avg_distance / max(1, len(states))

            # prox_norm for scaling
            prox_norm = 0.0
            for k in prox_center:
                prox_norm += float(torch.norm(prox_center[k], p=2).item() ** 2)
            prox_norm = math.sqrt(prox_norm) + 1e-12

            # adaptive mu (between base_mu and 5*base_mu)
            attack_factor = max(0.0, (avg_distance / prox_norm) - 0.1)
            adaptive_mu = self.base_mu * (1.0 + min(4.0, attack_factor))
            logging.debug(f"[{self.__class__.__name__}] avg_dist={avg_distance:.6f}, prox_norm={prox_norm:.6f}, adaptive_mu={adaptive_mu:.6f}")

            # final combine per-layer with clipping
            result = {}
            for k in accum:
                prox_term = adaptive_mu * (local_model[k] - prox_center[k])
                prox_term = self._clip_tensor(prox_term, self.clip_norm)

                mixed = self.a * local_model[k] + (1.0 - self.a) * accum[k] - prox_term
                mixed = self._clip_tensor(mixed, self.clip_norm)
                result[k] = mixed.clone()

        # cleanup
        del models, filtered
        gc.collect()
        logging.info(f"[{self.__class__.__name__}] Aggregation done. Neighbors={S}, adaptive_mu={adaptive_mu:.6f}")
        return result
