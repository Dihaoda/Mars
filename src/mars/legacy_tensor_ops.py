from __future__ import annotations

from collections import OrderedDict

import torch


StateDict = OrderedDict[str, torch.Tensor]


def state_dict_delta(local_state: StateDict, global_state: StateDict) -> StateDict:
    return OrderedDict(
        (
            name,
            local_state[name] - global_state[name]
            if torch.is_floating_point(global_state[name])
            else torch.zeros_like(global_state[name]),
        )
        for name in global_state
    )


def add_update(model_state: StateDict, update: StateDict, scale: float = 1.0) -> StateDict:
    return OrderedDict(
        (
            name,
            tensor + scale * update[name] if torch.is_floating_point(tensor) else tensor,
        )
        for name, tensor in model_state.items()
    )


def zeros_like_update(reference: StateDict) -> StateDict:
    return OrderedDict((name, torch.zeros_like(tensor)) for name, tensor in reference.items())


def flatten_update(update: StateDict) -> torch.Tensor:
    return torch.cat([tensor.detach().reshape(-1).float().cpu() for tensor in update.values()])


def unflatten_update(flat: torch.Tensor, reference: StateDict) -> StateDict:
    update = OrderedDict()
    pointer = 0
    for name, tensor in reference.items():
        numel = tensor.numel()
        update[name] = flat[pointer : pointer + numel].reshape(tensor.shape).to(tensor.device).type_as(tensor)
        pointer += numel
    return update


def average_updates(updates: list[StateDict], weights: list[float] | None = None) -> StateDict:
    if not updates:
        raise ValueError("Cannot aggregate an empty update list.")
    if weights is None:
        weights = [1.0 / len(updates)] * len(updates)
    total = zeros_like_update(updates[0])
    for update, weight in zip(updates, weights):
        for name in total:
            if torch.is_floating_point(total[name]):
                total[name] += update[name] * float(weight)
    return total
