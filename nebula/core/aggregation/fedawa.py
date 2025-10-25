import gc
import torch
import math
from nebula.core.aggregation.aggregator import Aggregator
import logging


class FedAWA(Aggregator):
    """
    Aggregator: Federated Averaging (FedAvg)
    Authors: McMahan et al.
    Year: 2016
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.adaptive_weights = {}
        logging.debug(f"[{self.__class__.__name__}] Start FedAWA")


    def get_local_model(self, models):
        if self._addr not in models:
            raise ValueError(f"[{self.__class__.__name__}] Local model ({self._addr}) not found in models")
        return models[self._addr]


    def calculate_adaptive_weight(self, client_vector, global_vector):
        """
        Tính trọng số thích nghi cho từng client vector dựa trên công thức:
        λ_k = 1 / ||τ_k^t - τ_g^t||_2 / Σ_j 1 / ||τ_j^t - τ_g^t||_2

        Args:
            client_vector (dict): Dictionary with node_addr as keys and state_dicts (model_params - local_model) as values.
            global_vector (dict): State_dict chứa trung bình có trọng số của các vector hiệu.

        Returns:
            dict: Dictionary with node_addr as keys and adaptive weights as values.
        """
        if not client_vector:
            return {}

        inverse_distances = {}  # lưu 1 / distance cho từng client
        total_inverse_distance = 0.0

        # --- Vòng lặp 1: Tính khoảng cách (distance) và nghịch đảo ---
        for node_addr, client_vec in client_vector.items():

            # Tính norm bình phương (L2)
            distance_sq = 0.0
            for layer, global_param in global_vector.items():
                diff = client_vec[layer] - global_param
                # norm() trả về scalar tensor, .item() để lấy float
                distance_sq += torch.norm(diff, p=2).item() ** 2

            distance = math.sqrt(distance_sq)
            if distance == 0:
                distance = 1e-10  # tránh chia cho 0
            logging.debug(f"[{self.__class__.__name__}] Distance for node {node_addr} at round {self.engine.round}: {distance}")

            inverse_distance = 1.0 / distance
            inverse_distances[node_addr] = inverse_distance
            total_inverse_distance += inverse_distance
        logging.debug(f"[{self.__class__.__name__}] Total inverse distance at round {self.engine.round}: {total_inverse_distance}")

        # --- Vòng lặp 2: Chuẩn hóa trọng số ---
        weights = {
            node_addr: inv_d / total_inverse_distance
            for node_addr, inv_d in inverse_distances.items()
        }
        logging.debug(f"[{self.__class__.__name__}] Adaptive weights calculated at round {self.engine.round}: {weights}")
        return weights


    def calculate_client_vector_and_global_vector(self, models):
        """
        accum = global_vector = {
            "layer1.weight": torch.Tensor([...])  # weighted sum of all (client_param - local_param)
            "layer1.bias": torch.Tensor([...]),
            ...
        }
        client_vector = {
            "node_addr": {
                "layer1.weight": tensor([...]),
                "layer1.bias": tensor([...]),
                ...
            }, ...
        }
        """
        client_vector = {}
        local_model, _ = self.get_local_model(models)
        total_samples = float(sum(weight for _, weight in models.values()))
        if total_samples == 0:
            raise ValueError("Total number of samples must be greater than zero.")
        accum = {layer: torch.zeros_like(param, dtype=torch.float32) for layer, param in local_model.items()}

        with torch.no_grad():
            for node_addr, (model_params, weight) in models.items():
                # Skip local model
                logging.debug(f"[{self.__class__.__name__}] Calculating vector for node: {node_addr}")

                if node_addr == self._addr:
                    continue
                if node_addr in self.adaptive_weights:
                    normalized_weight = self.adaptive_weights[node_addr]
                    logging.debug(f"[{self.__class__.__name__}] Round {self.engine.round}. Using adaptive weight {normalized_weight} for node: {node_addr}")
                else:
                    normalized_weight = weight / total_samples
                    logging.debug(f"[{self.__class__.__name__}] Round {self.engine.round}. Using uniform weight {normalized_weight} for node: {node_addr}")

                vector = {}
                for layer in local_model:
                    vector[layer] = model_params[layer] - local_model[layer]
                    accum[layer].add_(
                            vector[layer].to(accum[layer].dtype),
                            alpha=normalized_weight,
                        )
                client_vector[node_addr]  = vector
                logging.debug(f"[{self.__class__.__name__}] Vector calculated successfully for node: {node_addr}")

        logging.debug(f"[{self.__class__.__name__}] Vector client {list(client_vector.keys())} and global vector calculated {list(accum.keys())}")
        gc.collect()  # Giải phóng bộ nhớ
        return client_vector, accum


    """
    models = {
        "client1": [state_dict1, num_samples1],
        "client2": [{layer_name1: torch.Tensor, layer_name2: torch.Tensor}, num_samples2],
        ...
    }
    """
    def run_aggregation(self, models):
        super().run_aggregation(models)
        vector_client, merge_global_vector = self.calculate_client_vector_and_global_vector(models)
        adaptive_weights = self.calculate_adaptive_weight(vector_client, merge_global_vector)
        # Save adaptive weights for next round
        self.adaptive_weights = adaptive_weights
        logging.debug(f"[{self.__class__.__name__}] Save adaptive weights for next round: {self.adaptive_weights}")

        # Aggregation with adaptive weights
        accum = {layer: torch.zeros_like(param, dtype=torch.float32) for layer, param in merge_global_vector.items()}

        with torch.no_grad():
            for node_addr, (model_parameters, _) in models.items():
                if node_addr == self._addr:
                    continue

                normalized_weight = adaptive_weights.get(node_addr, 0.0)
                if normalized_weight == 0.0:
                    logging.debug(f"[{self.__class__.__name__}] No adaptive weight for {node_addr}, skipping in weighted aggregation.")
                    continue

                for layer in accum:
                    accum[layer].add_(
                        model_parameters[layer].to(accum[layer].dtype),
                        alpha=normalized_weight,
                    )

        logging.debug(f"[{self.__class__.__name__}] Aggregation done with adaptive weights at round {self.engine.round}: {adaptive_weights}")

        # calculate alpha*accum + (1-alpha)*local_model
        local_model, _ = self.get_local_model(models)
        alpha = 0.5
        for layer in accum:
            accum[layer].mul_(alpha)
            accum[layer].add_(local_model[layer], alpha=(1-alpha))

        logging.debug(f"[{self.__class__.__name__}] Final result FedAWA done at round {self.engine.round} with alpha: {alpha}")

        gc.collect()  # Giải phóng bộ nhớ
        return accum
