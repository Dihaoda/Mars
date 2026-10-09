from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .legacy_tensor_ops import StateDict, average_updates, flatten_update


@dataclass
class HFedSAConfig:
    beta_explore: float = 0.35
    malicious_penalty: float = 0.8
    detection_threshold: float = 0.5
    gmm_reg_covar: float = 1e-6
    weight_ema_enabled: bool = False
    weight_ema_momentum: float = 0.8
    posterior_clip_enabled: bool = False
    posterior_min: float = 0.001
    posterior_max: float = 0.999
    weight_bounds_enabled: bool = False
    min_weight: float = 0.0
    max_weight: float = 1.0
    retention_floor_enabled: bool = False
    retention_floor: float = 0.0
    retention_mode: str = "global"
    retention_risk_gamma: float = 1.0
    retention_pmal_threshold: float = 0.3
    retention_topk: int = 10
    beta_decay_enabled: bool = False
    beta_initial: float = 0.35
    beta_final: float = 0.10
    beta_decay_rounds: int = 20
    ddpg_beta_decay_enabled: bool = False
    ddpg_beta_final_max: float = 0.05
    ddpg_beta_decay_rounds: int = 100
    beta_mode: str = "static"
    normalization_mode: str = "none"
    formula_mode: str = "current"
    cluster_role_strategy: str = "current"
    malicious_hard_gate_enabled: bool = False
    malicious_hard_gate_threshold: float = 0.9
    ddpg_security_penalty: float = 0.2
    ddpg_residual_penalty: float = 0.0
    ddpg_aggregated_norm_penalty: float = 0.0
    ddpg_aggregated_norm_target: float = 1.0
    normalize_update_to_root: bool = False
    update_norm_clip_enabled: bool = False
    update_norm_clip_ratio: float = 1.0


class HFedSAStatic:
    def __init__(self, config: HFedSAConfig, name: str = "hfedsa_static", beta_agent=None) -> None:
        self.name = name
        self.config = config
        self.beta_agent = beta_agent
        self.last_fallback_reason: str | None = None
        self.round_index = 0
        self._ema_weights: dict[int, float] = {}
        self.last_beta_trace: dict = {}
        self._previous_test_loss: float | None = None
        self._previous_test_accuracy: float | None = None
        self.last_gmm_means = np.zeros((3, 3), dtype=float)
        self.last_gmm_variances = np.ones((3, 3), dtype=float)

    def aggregate(
        self,
        updates: list[tuple[int, StateDict]],
        root_update: StateDict | None = None,
        sample_counts: dict[int, int] | None = None,
        **_: object,
    ) -> tuple[StateDict, list[dict]]:
        self.round_index += 1
        self.last_fallback_reason = None
        if root_update is None:
            raise ValueError("H-FedSA requires a trusted root update.")
        if len(updates) < 3:
            self.last_fallback_reason = "fewer_than_three_updates"
            return average_updates([update for _, update in updates]), []

        root_flat = flatten_update(root_update)
        root_norm = torch.linalg.norm(root_flat).item()
        features, cosines, norms, feature_nan_flags, feature_inf_flags = self._extract_features(updates, root_flat, root_norm)
        posteriors, role_indices = self._fit_gmm(features)
        state_vector = self._ddpg_state_vector(features, posteriors)

        weighted_updates = []
        weights = []
        weighted_client_ids = []
        diagnostics = []
        previous_beta = float(self.last_beta_trace.get("beta_explore", self.config.beta_explore)) if self.last_beta_trace else self.config.beta_explore
        beta_explore = self._current_beta_explore(state_vector)
        retention_topk_indices = self._retention_topk_indices(posteriors, role_indices)
        self.last_beta_trace = {
            "round": self.round_index,
            "defense": self.name,
            "beta_mode": self.config.beta_mode,
            "previous_beta": previous_beta,
            "beta_explore": beta_explore,
            "state_mean": float(np.mean(state_vector)) if state_vector.size else 0.0,
            "state_std": float(np.std(state_vector)) if state_vector.size else 0.0,
            "state_vector": state_vector.tolist(),
            "reward": "NA",
            "actor_loss": "NA",
            "critic_loss": "NA",
            "replay_size": "NA",
        }
        for row_idx, (client_id, update) in enumerate(updates):
            p_benign = float(posteriors[row_idx, role_indices["benign"]])
            p_unseen = float(posteriors[row_idx, role_indices["unseen"]])
            p_malicious = float(posteriors[row_idx, role_indices["malicious"]])
            clipped_posteriors = self._clip_posteriors(np.asarray([p_benign, p_unseen, p_malicious], dtype=float))
            posterior_clipped = bool(not np.allclose(clipped_posteriors, np.asarray([p_benign, p_unseen, p_malicious], dtype=float)))
            p_benign, p_unseen, p_malicious = [float(value) for value in clipped_posteriors]
            cosine = float(cosines[row_idx])
            raw_weight = self._raw_weight(cosine, p_benign, p_unseen, p_malicious, beta_explore)
            bounded_weight, weight_clipped, retention_applied, effective_retention_floor = self._bound_weight(
                raw_weight,
                p_malicious=p_malicious,
                topk_eligible=row_idx in retention_topk_indices,
            )
            ema_weight = "NA"
            weight = bounded_weight
            if self.config.weight_ema_enabled:
                previous = self._ema_weights.get(int(client_id), bounded_weight)
                weight = self.config.weight_ema_momentum * previous + (1.0 - self.config.weight_ema_momentum) * bounded_weight
                self._ema_weights[int(client_id)] = weight
                ema_weight = weight
            hard_gated = bool(
                self.config.malicious_hard_gate_enabled
                and p_malicious >= self.config.malicious_hard_gate_threshold
            )
            if hard_gated:
                weight = 0.0
            diagnostics.append(
                {
                    "client_id": client_id,
                    "weight": weight,
                    "raw_weight": raw_weight,
                    "bounded_weight": bounded_weight,
                    "ema_weight": ema_weight,
                    "p_benign": p_benign,
                    "p_unseen": p_unseen,
                    "p_malicious": p_malicious,
                    "direction": float(features[row_idx, 0]),
                    "intensity": float(features[row_idx, 1]),
                    "sparsity": float(features[row_idx, 2]),
                    "update_norm": float(norms[row_idx]),
                    "predicted_malicious": p_malicious >= self.config.detection_threshold,
                    "gmm_fallback": self.last_fallback_reason or "",
                    "gmm_fallback_used": bool(self.last_fallback_reason),
                    "gmm_fallback_reason": self.last_fallback_reason or "",
                    "posterior_clipped": posterior_clipped,
                    "weight_clipped": weight_clipped,
                    "retention_applied": retention_applied,
                    "effective_retention_floor": effective_retention_floor,
                    "retention_topk_eligible": row_idx in retention_topk_indices,
                    "hard_gated": hard_gated,
                    "feature_nan_flag": bool(feature_nan_flags[row_idx]),
                    "feature_inf_flag": bool(feature_inf_flags[row_idx]),
                    "beta_explore": beta_explore,
                }
            )
            transformed_update, transformed_update_norm, update_scale, update_clipped = self._transform_update_for_aggregation(
                update,
                float(norms[row_idx]),
                root_norm,
            )
            diagnostics[-1]["transformed_update_norm"] = transformed_update_norm
            diagnostics[-1]["normalized_update_norm"] = transformed_update_norm
            diagnostics[-1]["raw_update_norm"] = float(norms[row_idx])
            diagnostics[-1]["root_update_norm"] = float(root_norm)
            diagnostics[-1]["update_scale"] = update_scale
            diagnostics[-1]["update_norm_clipped"] = bool(update_clipped)
            diagnostics[-1]["normalization_mode"] = self.config.normalization_mode
            diagnostics[-1]["formula_mode"] = self.config.formula_mode
            diagnostics[-1]["cluster_role_strategy"] = self.config.cluster_role_strategy
            diagnostics[-1]["malicious_hard_gate_enabled"] = self.config.malicious_hard_gate_enabled
            diagnostics[-1]["malicious_hard_gate_threshold"] = self.config.malicious_hard_gate_threshold
            diagnostics[-1]["retention_floor_enabled"] = self.config.retention_floor_enabled
            diagnostics[-1]["retention_floor"] = self.config.retention_floor
            diagnostics[-1]["retention_mode"] = self.config.retention_mode
            diagnostics[-1]["retention_risk_gamma"] = self.config.retention_risk_gamma
            diagnostics[-1]["retention_pmal_threshold"] = self.config.retention_pmal_threshold
            diagnostics[-1]["retention_topk"] = self.config.retention_topk
            diagnostics[-1]["normalize_update_to_root"] = self._normalizes_to_root()
            diagnostics[-1]["update_norm_clip_enabled"] = self._clips_update_norm()
            if weight > 0.0:
                weighted_updates.append(transformed_update)
                weights.append(weight)
                weighted_client_ids.append(client_id)

        if not weighted_updates or sum(weights) <= 0:
            self.last_fallback_reason = self.last_fallback_reason or "all_hfedsa_weights_zero"
            for item in diagnostics:
                item["gmm_fallback"] = self.last_fallback_reason
                item["gmm_fallback_used"] = bool(self.last_fallback_reason)
                item["gmm_fallback_reason"] = self.last_fallback_reason
                item["weight"] = 1.0 / max(len(updates), 1)
                item["num_effective_clients"] = len(updates)
            return average_updates([update for _, update in updates]), diagnostics
        normalizer = sum(weights)
        normalized_weights = [w / normalizer for w in weights]
        normalized_by_client = {client_id: normalized_weights[idx] for idx, client_id in enumerate(weighted_client_ids)}
        for item in diagnostics:
            item["weight"] = float(normalized_by_client.get(item["client_id"], 0.0))
            item["risk_weighted_contribution"] = float(item["weight"]) * float(item["p_malicious"])
            item["num_effective_clients"] = len(weighted_updates)
        return average_updates(weighted_updates, normalized_weights), diagnostics

    def _raw_weight(
        self,
        cosine: float,
        p_benign: float,
        p_unseen: float,
        p_malicious: float,
        beta_explore: float,
    ) -> float:
        relu_cosine = max(cosine, 0.0)
        mode = self.config.formula_mode.lower()
        if mode == "paper":
            return max(0.0, p_benign * relu_cosine + p_unseen * beta_explore)
        if mode == "direction_factor":
            base = max(0.0, p_benign + beta_explore * p_unseen - self.config.malicious_penalty * p_malicious)
            return max(0.0, relu_cosine * base)
        if mode == "direction_squared":
            base = max(0.0, p_benign + beta_explore * p_unseen - self.config.malicious_penalty * p_malicious)
            return max(0.0, (relu_cosine ** 2) * base)
        if mode == "direction_risk":
            base = max(0.0, p_benign + beta_explore * p_unseen)
            return max(0.0, relu_cosine * (1.0 - p_malicious) * base)
        if mode == "direction_risk_squared":
            base = max(0.0, p_benign + beta_explore * p_unseen)
            return max(0.0, relu_cosine * ((1.0 - p_malicious) ** 2) * base)
        if mode == "pmal_suppression":
            base = max(0.0, p_benign + beta_explore * p_unseen)
            return max(0.0, relu_cosine * base * (1.0 - p_malicious))
        if mode == "pmal_suppression_squared":
            base = max(0.0, p_benign + beta_explore * p_unseen)
            return max(0.0, relu_cosine * base * ((1.0 - p_malicious) ** 2))
        if mode == "pmal_threshold_05":
            risk = (1.0 - p_malicious) if p_malicious > 0.5 else 1.0
            base = max(0.0, p_benign + beta_explore * p_unseen - self.config.malicious_penalty * p_malicious)
            return max(0.0, relu_cosine * base * risk)
        if mode == "pmal_threshold_07":
            risk = (1.0 - p_malicious) if p_malicious > 0.7 else 1.0
            base = max(0.0, p_benign + beta_explore * p_unseen - self.config.malicious_penalty * p_malicious)
            return max(0.0, relu_cosine * base * risk)
        if mode == "cosine_one_minus_pmalicious":
            return max(0.0, relu_cosine * (1.0 - p_malicious))
        if mode == "cosine_pbenign":
            return max(0.0, relu_cosine * p_benign)
        alignment_floor = 0.05 if p_unseen > p_benign else 0.0
        alignment = max(cosine, alignment_floor)
        probability_weight = max(
            0.0,
            p_benign + beta_explore * p_unseen - self.config.malicious_penalty * p_malicious,
        )
        return max(0.0, alignment * probability_weight)

    def _transform_update_for_aggregation(
        self,
        update: StateDict,
        client_norm: float,
        root_norm: float,
    ) -> tuple[StateDict, float, float, bool]:
        if client_norm <= 1e-12 or root_norm <= 1e-12:
            return update, client_norm, 1.0, False
        scale = 1.0
        clipped = False
        if self._normalizes_to_root():
            scale = root_norm / (client_norm + 1e-12)
        if self._clips_update_norm():
            max_norm = max(root_norm * self.config.update_norm_clip_ratio, 1e-12)
            clip_scale = max_norm / (client_norm + 1e-12)
            if clip_scale < scale:
                clipped = True
                scale = clip_scale
        if np.isclose(scale, 1.0):
            return update, client_norm, 1.0, clipped
        transformed = {name: tensor * scale for name, tensor in update.items()}
        return transformed, float(client_norm * scale), float(scale), clipped

    def _normalizes_to_root(self) -> bool:
        mode = self.config.normalization_mode.lower()
        return bool(self.config.normalize_update_to_root or mode in {"root_norm", "root_norm_then_clip"})

    def _clips_update_norm(self) -> bool:
        mode = self.config.normalization_mode.lower()
        return bool(self.config.update_norm_clip_enabled or mode in {"clip", "root_norm_then_clip"})

    def _extract_features(
        self,
        updates: list[tuple[int, StateDict]],
        root_flat: torch.Tensor,
        root_norm: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        feature_rows = []
        cosines = []
        norms = []
        nan_flags = []
        inf_flags = []
        sqrt_dim = np.sqrt(float(root_flat.numel()))
        for _, update in updates:
            client_flat = flatten_update(update)
            client_norm = torch.linalg.norm(client_flat).item()
            if client_norm <= 1e-12 or root_norm <= 1e-12:
                cosine = 0.0
            else:
                cosine = torch.nn.functional.cosine_similarity(root_flat, client_flat, dim=0).item()
            l1_norm = torch.linalg.norm(client_flat, ord=1).item()
            normalized_sparsity = l1_norm / (sqrt_dim * client_norm + 1e-12)
            direction_distance = 1.0 - cosine
            intensity = np.log((client_norm + 1e-12) / (root_norm + 1e-12))
            raw_row = np.asarray([direction_distance, intensity, normalized_sparsity], dtype=np.float64)
            nan_flags.append(bool(np.isnan(raw_row).any()))
            inf_flags.append(bool(np.isinf(raw_row).any()))
            feature_rows.append(raw_row.tolist())
            cosines.append(cosine)
            norms.append(client_norm)
        features = np.asarray(feature_rows, dtype=np.float64)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        cosines_array = np.nan_to_num(np.asarray(cosines), nan=0.0, posinf=0.0, neginf=0.0)
        norms_array = np.nan_to_num(np.asarray(norms), nan=0.0, posinf=0.0, neginf=0.0)
        return features, cosines_array, norms_array, np.asarray(nan_flags, dtype=bool), np.asarray(inf_flags, dtype=bool)

    def observe_round(self, train_loss: float, test_loss: float, test_accuracy: float, diagnostics: list[dict]) -> dict:
        if self.config.beta_mode != "ddpg" or self.beta_agent is None:
            return {}
        malicious_weight = 0.0
        malicious_residual = 0.0
        aggregated_update_norm = 0.0
        aggregated_norm_spike = 0.0
        if self._previous_test_loss is None:
            reward = 0.0
        else:
            loss_improvement = self._previous_test_loss - float(test_loss)
            accuracy_improvement = (float(test_accuracy) - float(self._previous_test_accuracy or test_accuracy)) / 100.0
            malicious_weight = np.mean(
                [float(item["weight"]) for item in diagnostics if bool(item.get("predicted_malicious", False)) and item.get("weight") not in {"", "NA", None}]
            ) if diagnostics else 0.0
            malicious_residual = float(
                sum(
                    float(item.get("weight", 0.0)) * float(item.get("p_malicious", 0.0))
                    for item in diagnostics
                    if item.get("weight") not in {"", "NA", None} and item.get("p_malicious") not in {"", "NA", None}
                )
            )
            aggregated_update_norm = float(
                next(
                    (
                        item.get("aggregated_update_norm")
                        for item in diagnostics
                        if item.get("aggregated_update_norm") not in {"", "NA", None}
                    ),
                    0.0,
                )
            )
            aggregated_norm_spike = max(0.0, aggregated_update_norm - self.config.ddpg_aggregated_norm_target)
            beta = float(self.last_beta_trace.get("beta_explore", self.config.beta_explore))
            previous_beta = float(self.last_beta_trace.get("previous_beta", beta))
            smoothness_penalty = abs(beta - previous_beta)
            reward = float(
                loss_improvement
                + accuracy_improvement
                - self.config.ddpg_security_penalty * malicious_weight
                - self.config.ddpg_residual_penalty * malicious_residual
                - self.config.ddpg_aggregated_norm_penalty * aggregated_norm_spike
                - 0.05 * smoothness_penalty
            )
        self._previous_test_loss = float(test_loss)
        self._previous_test_accuracy = float(test_accuracy)
        next_state = np.asarray(self.last_beta_trace.get("state_vector", np.zeros(24)), dtype=float)
        train_info = self.beta_agent.observe(reward, next_state)
        self.last_beta_trace.update(
            {
                "reward": reward,
                "actor_loss": train_info.get("actor_loss", "NA"),
                "critic_loss": train_info.get("critic_loss", "NA"),
                "replay_size": train_info.get("replay_size", "NA"),
                "predicted_malicious_mean_weight": malicious_weight,
                "malicious_weighted_contribution": malicious_residual,
                "aggregated_update_norm": aggregated_update_norm,
                "aggregated_norm_spike": aggregated_norm_spike,
            }
        )
        return self.last_beta_trace

    def _current_beta_explore(self, state_vector: np.ndarray | None = None) -> float:
        if self.config.beta_mode == "ddpg" and self.beta_agent is not None and state_vector is not None:
            previous_beta = float(self.last_beta_trace.get("beta_explore", self.config.beta_explore)) if self.last_beta_trace else self.config.beta_explore
            beta = float(self.beta_agent.select_beta(state_vector, train=True))
            if self.config.ddpg_beta_decay_enabled:
                decay_rounds = max(1, int(self.config.ddpg_beta_decay_rounds))
                progress = min(max(self.round_index - 1, 0), decay_rounds - 1) / float(max(decay_rounds - 1, 1))
                initial_max = float(self.beta_agent.config.max_beta)
                dynamic_max = initial_max + progress * (self.config.ddpg_beta_final_max - initial_max)
                beta = min(beta, max(self.beta_agent.config.min_beta, dynamic_max))
            self.last_beta_trace["previous_beta"] = previous_beta
            return beta
        if not self.config.beta_decay_enabled:
            return self.config.beta_explore
        decay_rounds = max(1, int(self.config.beta_decay_rounds))
        if decay_rounds == 1:
            return self.config.beta_final
        progress = min(max(self.round_index - 1, 0), decay_rounds - 1) / float(decay_rounds - 1)
        return float(self.config.beta_initial + progress * (self.config.beta_final - self.config.beta_initial))

    def _clip_posteriors(self, posteriors: np.ndarray) -> np.ndarray:
        if not self.config.posterior_clip_enabled:
            return posteriors
        clipped = np.clip(posteriors, self.config.posterior_min, self.config.posterior_max)
        total = float(clipped.sum())
        if total <= 0.0 or not np.isfinite(total):
            return np.asarray([1.0, 0.0, 0.0], dtype=float)
        return clipped / total

    def _bound_weight(
        self,
        weight: float,
        p_malicious: float = 0.0,
        topk_eligible: bool = True,
    ) -> tuple[float, bool, bool, float]:
        if not self.config.weight_bounds_enabled and not self.config.retention_floor_enabled:
            return float(weight), False, False, 0.0
        minimum = self.config.min_weight if self.config.weight_bounds_enabled else 0.0
        effective_retention_floor = self._effective_retention_floor(p_malicious, topk_eligible)
        minimum = max(minimum, effective_retention_floor)
        maximum = self.config.max_weight if self.config.weight_bounds_enabled else 1.0
        bounded = float(np.clip(weight, minimum, maximum))
        retention_applied = bool(effective_retention_floor > 0.0 and weight < effective_retention_floor)
        return bounded, bool(not np.isclose(bounded, weight)), retention_applied, effective_retention_floor

    def _effective_retention_floor(self, p_malicious: float, topk_eligible: bool) -> float:
        if not self.config.retention_floor_enabled:
            return 0.0
        mode = self.config.retention_mode.lower()
        if mode == "low_pmal_only":
            return self.config.retention_floor if p_malicious < self.config.retention_pmal_threshold else 0.0
        if mode == "scaled_by_risk":
            gamma = max(float(self.config.retention_risk_gamma), 0.0)
            return self.config.retention_floor * (max(0.0, 1.0 - p_malicious) ** gamma)
        if mode == "topk_low_risk":
            return self.config.retention_floor if topk_eligible else 0.0
        return self.config.retention_floor

    def _retention_topk_indices(self, posteriors: np.ndarray, role_indices: dict[str, int]) -> set[int]:
        if not self.config.retention_floor_enabled or self.config.retention_mode.lower() != "topk_low_risk":
            return set(range(len(posteriors)))
        count = min(max(int(self.config.retention_topk), 0), len(posteriors))
        if count <= 0:
            return set()
        p_malicious = posteriors[:, role_indices["malicious"]]
        return set(int(index) for index in np.argsort(p_malicious)[:count])

    def _fit_gmm(self, features: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        n_components = min(3, len(features))
        if n_components < 3:
            self.last_fallback_reason = "fewer_than_three_samples_for_gmm"
            return self._uniform_benign_posteriors(len(features)), {"benign": 0, "unseen": 1, "malicious": 2}
        if not np.isfinite(features).all():
            self.last_fallback_reason = "non_finite_hfedsa_features"
            return self._uniform_benign_posteriors(len(features)), {"benign": 0, "unseen": 1, "malicious": 2}
        if np.allclose(features.var(axis=0), 0.0):
            self.last_fallback_reason = "collapsed_hfedsa_features"
            return self._uniform_benign_posteriors(len(features)), {"benign": 0, "unseen": 1, "malicious": 2}

        normalized = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        try:
            raw_posteriors, centers, variances = self._diagonal_gmm(normalized, n_components=3, reg_covar=self.config.gmm_reg_covar)
        except FloatingPointError:
            self.last_fallback_reason = "diagonal_gmm_numerical_failure"
            return self._uniform_benign_posteriors(len(features)), {"benign": 0, "unseen": 1, "malicious": 2}
        if not np.isfinite(raw_posteriors).all() or not np.isfinite(centers).all():
            self.last_fallback_reason = "diagonal_gmm_non_finite_output"
            return self._uniform_benign_posteriors(len(features)), {"benign": 0, "unseen": 1, "malicious": 2}
        self.last_gmm_means = centers.copy()
        self.last_gmm_variances = variances.copy()

        benign_cluster, malicious_cluster = self._assign_cluster_roles(centers)
        if malicious_cluster == benign_cluster:
            malicious_cluster = int(np.argmax(centers[:, 0]))
        unseen_candidates = [idx for idx in range(3) if idx not in {benign_cluster, malicious_cluster}]
        unseen_cluster = unseen_candidates[0] if unseen_candidates else int(np.argmax(centers[:, 1]))

        posteriors = np.zeros_like(raw_posteriors)
        posteriors[:, 0] = raw_posteriors[:, benign_cluster]
        posteriors[:, 1] = raw_posteriors[:, unseen_cluster]
        posteriors[:, 2] = raw_posteriors[:, malicious_cluster]
        return posteriors, {"benign": 0, "unseen": 1, "malicious": 2}

    def _assign_cluster_roles(self, centers: np.ndarray) -> tuple[int, int]:
        strategy = self.config.cluster_role_strategy.lower()
        if strategy == "composite_sparsity":
            malicious_score = 1.5 * centers[:, 2] + 0.35 * centers[:, 0] + 0.15 * np.abs(centers[:, 1])
            malicious_cluster = int(np.argmax(malicious_score))
            benign_score = centers[:, 0] + 0.25 * np.abs(centers[:, 1]) + 0.15 * np.maximum(centers[:, 2], 0.0)
            benign_cluster = int(np.argmin(benign_score))
            if benign_cluster == malicious_cluster:
                benign_score = np.where(np.arange(len(centers)) == malicious_cluster, np.inf, benign_score)
                benign_cluster = int(np.argmin(benign_score))
            return benign_cluster, malicious_cluster
        benign_cluster = int(np.argmin(centers[:, 0] + 0.25 * np.abs(centers[:, 1])))
        malicious_cluster = int(np.argmax(centers[:, 0] + 0.5 * np.abs(centers[:, 1])))
        return benign_cluster, malicious_cluster

    def _uniform_benign_posteriors(self, count: int) -> np.ndarray:
        posteriors = np.zeros((count, 3), dtype=np.float64)
        posteriors[:, 0] = 1.0
        return posteriors

    def _diagonal_gmm(self, x: np.ndarray, n_components: int, reg_covar: float, max_iter: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        order = np.argsort(x[:, 0])
        init_positions = np.linspace(0, len(order) - 1, n_components).astype(int)
        means = x[order[init_positions]].copy()
        variances = np.tile(x.var(axis=0) + reg_covar, (n_components, 1))
        priors = np.full(n_components, 1.0 / n_components)

        for _ in range(max_iter):
            log_probs = []
            for component in range(n_components):
                var = variances[component] + reg_covar
                log_det = np.sum(np.log(var))
                mahalanobis = np.sum(((x - means[component]) ** 2) / var, axis=1)
                log_probs.append(np.log(priors[component] + 1e-12) - 0.5 * (log_det + mahalanobis))
            log_probs = np.vstack(log_probs).T
            log_probs -= log_probs.max(axis=1, keepdims=True)
            responsibilities = np.exp(log_probs)
            responsibilities /= responsibilities.sum(axis=1, keepdims=True)

            counts = responsibilities.sum(axis=0) + 1e-8
            priors = counts / len(x)
            means = (responsibilities.T @ x) / counts[:, None]
            for component in range(n_components):
                diff = x - means[component]
                variances[component] = (responsibilities[:, component][:, None] * diff * diff).sum(axis=0) / counts[component]
                variances[component] += reg_covar

        return responsibilities, means, variances

    def _ddpg_state_vector(self, features: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        previous_beta = float(self.last_beta_trace.get("beta_explore", self.config.beta_explore)) if self.last_beta_trace else self.config.beta_explore
        previous_loss = float(self._previous_test_loss) if self._previous_test_loss is not None else 0.0
        previous_accuracy = float(self._previous_test_accuracy) / 100.0 if self._previous_test_accuracy is not None else 0.0
        pieces = [
            self.last_gmm_means.reshape(-1),
            self.last_gmm_variances.reshape(-1),
            posteriors.mean(axis=0) if posteriors.size else np.zeros(3),
            features.mean(axis=0) if features.size else np.zeros(3),
            np.asarray([previous_loss, previous_accuracy, previous_beta, float(self.round_index)], dtype=float),
        ]
        state = np.concatenate([np.asarray(piece, dtype=float).reshape(-1) for piece in pieces])
        self.last_beta_trace["state_vector"] = state.tolist() if self.last_beta_trace else state.tolist()
        return state
