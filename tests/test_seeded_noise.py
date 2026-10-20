import sys
from types import ModuleType

import pytest

from nodes.type_converter_node import MAITypeConverterNode
from utils.seeded_noise import MAISeededNoise


@pytest.mark.parametrize("seed,expected", [
    (0, 0), (42, 42), ((1 << 64) - 1, (1 << 64) - 1),
    (-1, (1 << 64) - 1), (1 << 64, 0),
])
def test_seed_uses_unsigned_64_bit_range(seed, expected):
    assert MAISeededNoise(seed).seed == expected


@pytest.mark.parametrize("batch_index", [None, [0, 2, 0]])
def test_noise_output_delegates_to_comfy_with_exact_seed_and_latent(monkeypatch, batch_index):
    calls = []
    generated = object()

    def prepare_noise(samples, seed, indices):
        calls.append((samples, seed, indices))
        return generated

    comfy = ModuleType("comfy")
    sample = ModuleType("comfy.sample")
    sample.prepare_noise = prepare_noise
    comfy.sample = sample
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.sample", sample)

    seed = (1 << 64) - 1
    noise = MAITypeConverterNode().convert(value=str(seed))[4]
    assert calls == []  # Conversion does not allocate noise or touch the RNG.
    samples = object()
    latent = {"samples": samples}
    if batch_index is not None:
        latent["batch_index"] = batch_index

    assert noise.generate_noise(latent) is generated
    assert noise.generate_noise(latent) is generated
    assert calls == [(samples, seed, batch_index)] * 2
    assert latent["samples"] is samples
