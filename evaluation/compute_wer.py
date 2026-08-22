import whisper
import jiwer


def compute_wer(wav_paths, references, whisper_model="medium"):
    model = whisper.load_model(whisper_model)
    hyps = []
    for wav_path in wav_paths:
        result = model.transcribe(wav_path, language="en")
        hyps.append(result["text"])
    return jiwer.wer(references, hyps)
