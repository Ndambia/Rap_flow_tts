# Wraps the pretrained NISQA model (Mittag et al., Interspeech 2021) referenced in the paper.
from nisqa.NISQA_model import nisqaModel


def compute_nisqa(wav_dir, pretrained_ckpt="weights/nisqa.tar"):
    args = {"mode": "predict_dir", "pretrained_model": pretrained_ckpt, "data_dir": wav_dir}
    nisqa = nisqaModel(args)
    df = nisqa.predict()
    return df["mos_pred"].mean(), df["mos_pred"].std()
