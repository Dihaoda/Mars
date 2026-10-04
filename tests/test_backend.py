import torch
import torch.nn.functional as F
import pytest

from mars.backend import Backend


def rows():
    return [{"text": "good nice enjoyable movie", "label": 1}, {"text": "bad dull awful movie", "label": 0}]


def test_client_reset_optimizer_isolation_and_frozen_base(small_config):
    backend = Backend(small_config)
    initial = backend.state()
    frozen = {k: v.detach().clone() for k, v in backend.model.named_parameters() if not v.requires_grad}
    first, _ = backend.train(initial, rows(), 71, max_steps=2)
    backend.train(first, list(reversed(rows())), 99, max_steps=2)
    repeated, _ = backend.train(initial, rows(), 71, max_steps=2)
    for key in first:
        torch.testing.assert_close(first[key], repeated[key], rtol=0, atol=0)
    assert any(not initial[k].equal(first[k]) for k in initial)
    for key, value in backend.model.named_parameters():
        if key in frozen:
            torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)


def test_answer_only_loss_and_probe_restore(small_config):
    backend = Backend(small_config)
    initial = backend.state()
    trained, _ = backend.train(initial, rows(), 1, max_steps=1)
    backend.load(trained)
    ids, labels = backend.encode(rows()[0])
    assert len(ids) == len(labels)
    assert -100 in labels
    assert [x for x in labels if x != -100] == backend.answers[1]
    result = backend.probe(initial, {"sst2": rows(), "imdb": rows()})
    assert all(x > 0 for x in result)
    for key, value in backend.state().items():
        torch.testing.assert_close(value, trained[key], rtol=0, atol=0)


@pytest.mark.parametrize("label", [None, 0, 1])
def test_selected_logits_preserve_losses_and_gradients(small_config, label):
    backend = Backend(small_config)
    examples = [{"text": "good movie", "label": 1},
                {"text": "bad dull awful movie with extra words", "label": 0}]
    trained, _ = backend.train(backend.state(), examples, 19, max_steps=1)
    backend.load(trained)  # Exercise nonzero gradients for both LoRA factors.
    backend.model.eval()
    inputs, labels = backend.batch(examples, label)
    target = labels[:, 1:]
    full = backend.model(**inputs).logits[:, :-1].float()
    expected = F.cross_entropy(full.reshape(-1, full.shape[-1]), target.reshape(-1),
                               reduction="none", ignore_index=-100).reshape(target.shape).sum(1)
    expected.sum().backward()
    gradients = {name: param.grad.clone() for name, param in backend.model.named_parameters()
                 if param.requires_grad}
    backend.model.zero_grad(set_to_none=True)
    actual, counts = backend._losses(examples, label)
    actual.sum().backward()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(counts, (target != -100).sum(1))
    for name, param in backend.model.named_parameters():
        if param.requires_grad:
            torch.testing.assert_close(param.grad, gradients[name], atol=1e-6, rtol=1e-5)


def test_evaluation_reuses_true_label_score(small_config):
    backend = Backend(small_config)
    examples = [{"text": "good movie", "label": 1}, {"text": "bad dull awful movie", "label": 0}]
    state = backend.state()
    backend.model.eval()
    with torch.no_grad():
        losses, counts = backend._losses(examples)
        expected = float(losses.sum()) / int(counts.sum())
    result = backend.evaluate(state, examples)
    assert result["loss"] == pytest.approx(expected, rel=1e-6)
