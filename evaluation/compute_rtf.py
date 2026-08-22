import time
import torch


@torch.no_grad()
def measure_rtf(model, hifigan, texts, n_steps=2, device="cuda", n_warmup=5):
    from inference.synthesize import synthesize
    for i in range(n_warmup):
        synthesize(model, texts[i % len(texts)], hifigan, device=device, n_steps=n_steps)
    torch.cuda.synchronize()

    total_synth_time, total_audio_time = 0.0, 0.0
    for text in texts:
        start = time.time()
        wav, _ = synthesize(model, text, hifigan, device=device, n_steps=n_steps)
        torch.cuda.synchronize()
        total_synth_time += time.time() - start
        total_audio_time += len(wav) / 22050.0
    return total_synth_time / total_audio_time
