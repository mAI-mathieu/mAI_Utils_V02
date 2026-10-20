"""Seed-backed NOISE object compatible with ComfyUI custom samplers."""


class MAISeededNoise:
    def __init__(self, seed):
        # Keep every converted integer usable as an unsigned 64-bit seed.
        self.seed = seed % (1 << 64)

    def generate_noise(self, input_latent):
        # Delay the ComfyUI dependency until sampling actually needs the noise.
        from comfy.sample import prepare_noise

        return prepare_noise(
            input_latent["samples"], self.seed, input_latent.get("batch_index"),
        )
