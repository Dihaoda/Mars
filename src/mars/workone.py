"""Auditable adapter around the supplied work-one hfedsa_ddpg implementation.

The original GMM, role mapping, weighting and DDPG code are retained. Only
feature computation is adapted to LoRA; aggregation is performed by Mars.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from .geometry import delta, norm, cosine, raw_delta
from .legacy_hfedsa import HFedSAConfig, HFedSAStatic
from .legacy_ddpg import DDPGBetaAgent, DDPGBetaConfig


def workone_features(local, base, root, scale, representation):
    if representation == 'raw':
        # Match the original implementation's float32 vector arithmetic.
        x, y = raw_delta(local, base).float(), raw_delta(root, base).float()
        length, reference_length = float(x.norm()), float(y.norm())
        direction = float(torch.nn.functional.cosine_similarity(x, y, dim=0)) if min(length, reference_length) > 1e-12 else 0.0
        sparsity = float(x.abs().sum()) / (math.sqrt(x.numel()) * length + 1e-12)
    else:
        x, y = delta(local, base, scale), delta(root, base, scale)
        length, reference_length = norm(x), norm(y)
        direction = cosine(x, y) if min(length, reference_length) > 1e-12 else 0.0
        dimension = sum(m.u.shape[0] * m.v.shape[1] for m in x.values())
        sparsity = sum(m.l1() for m in x.values()) / (math.sqrt(dimension) * length + 1e-12)
    feature = [1.0 - direction, math.log((length + 1e-12) / (reference_length + 1e-12)), sparsity]
    if not np.isfinite(feature).all():
        raise FloatingPointError('Non-finite work-one LoRA features')
    return feature, direction, length, reference_length


class WorkOneDefense(HFedSAStatic):
    def __init__(self, cfg):
        spec = cfg['workone']
        mode = 'ddpg' if cfg['defense']['name'] == 'hfedsa_ddpg' else 'static'
        config = HFedSAConfig(beta_mode=mode, beta_explore=spec['static_beta'],
                             malicious_penalty=spec['malicious_penalty'],
                             detection_threshold=cfg['defense']['threshold'],
                             gmm_reg_covar=spec['gmm_reg_covar'],
                             ddpg_security_penalty=spec['security_penalty'])
        agent = DDPGBetaAgent(DDPGBetaConfig(**spec['ddpg']), seed=cfg['seed'] + 901) if mode == 'ddpg' else None
        super().__init__(config, name=cfg['defense']['name'], beta_agent=agent)
        self.representation = cfg['defense']['representation']
        self._feature_cache = None
        self.controller_updates = 0
        self.pending_reward = None

    def _current_beta_explore(self, state_vector=None):
        # Complete (s[t-1], a[t-1], r[t-1], s[t]) before selecting a[t].
        # Supplied code instead stored s[t-1] twice, creating self-transitions.
        if self.beta_agent is not None and self.pending_reward is not None:
            self.beta_agent.observe(self.pending_reward, state_vector)
            if self.beta_agent.last_actor_loss != 'NA':
                self.controller_updates += 1
            self.pending_reward = None
        return super()._current_beta_explore(state_vector)

    def _extract_features(self, updates, root_flat, root_norm):
        if self._feature_cache is None:
            raise RuntimeError('LoRA features must be computed before scoring')
        return self._feature_cache

    def score(self, states, base, root, scale, selected):
        values = [workone_features(s, base, root, scale, self.representation) for s in states]
        x = np.asarray([v[0] for v in values])
        self._feature_cache = (x, np.asarray([v[1] for v in values]), np.asarray([v[2] for v in values]),
                               np.zeros(len(values), dtype=bool), np.zeros(len(values), dtype=bool))
        # Scalar proxies avoid materializing hundreds of millions of matrix
        # entries. Their aggregate is discarded; only original-code weights are used.
        proxies = [(i, {'proxy': torch.zeros(1)}) for i in selected]
        _, rows = super().aggregate(proxies, {'proxy': torch.tensor([values[0][3]])})
        self._feature_cache = None
        for row in rows:
            row['p_heterogeneous'] = row['p_unseen']
            row['probed'] = False
        return [r['weight'] for r in rows], rows, {
            'name': self.name, 'representation': self.representation,
            'fallback': self.last_fallback_reason, 'probe_count': 0,
            'gmm_means_standardized': self.last_gmm_means.tolist(),
            'gmm_diagonal_variances': self.last_gmm_variances.tolist(),
            'beta': self.last_beta_trace['beta_explore']}

    def observe_validation(self, validation, diagnostics):
        if self.beta_agent is None:
            return {}
        loss = float(np.mean([v['loss'] for v in validation.values()]))
        accuracy = 100.0 * float(np.mean([v['accuracy'] for v in validation.values()]))
        # Preserve work-one reward terms, but fix its empty-mean NaN and the
        # zero-accuracy truthiness bug. Only validation and predicted risks enter.
        flagged_weights = [r['weight'] for r in diagnostics if r['predicted_malicious']]
        malicious_weight = float(np.mean(flagged_weights)) if flagged_weights else 0.0
        beta = self.last_beta_trace['beta_explore']
        previous_beta = self.last_beta_trace['previous_beta']
        reward = 0.0 if self._previous_test_loss is None else (
            self._previous_test_loss - loss + (accuracy - self._previous_test_accuracy) / 100.0
            - self.config.ddpg_security_penalty * malicious_weight - 0.05 * abs(beta - previous_beta))
        if not np.isfinite(reward):
            raise FloatingPointError('Non-finite DDPG reward')
        self._previous_test_loss, self._previous_test_accuracy = loss, accuracy
        self.pending_reward = reward
        info = {'actor_loss': self.beta_agent.last_actor_loss,
                'critic_loss': self.beta_agent.last_critic_loss, 'replay_size': len(self.beta_agent.replay)}
        if not all(torch.isfinite(p).all() for model in (self.beta_agent.actor, self.beta_agent.critic) for p in model.parameters()):
            raise FloatingPointError('Non-finite DDPG parameters')
        self.last_beta_trace.update(reward=reward, reward_source='validation_only',
                                    controller_updates=self.controller_updates, **info)
        return dict(self.last_beta_trace)
