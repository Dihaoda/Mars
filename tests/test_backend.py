import torch

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
