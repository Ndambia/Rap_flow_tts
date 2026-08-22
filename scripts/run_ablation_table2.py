"""Reproduces Table 2 ablation grid (A)-(H) from the paper."""

CONFIG_GRID = {
    "A": dict(run_stage2=False),
    "B": dict(run_stage2=True, freeze_encoder=False, shared_dropout=False,
              use_huber=False, delta_schedule=False, adversarial=False),
    "C": dict(run_stage2=True, freeze_encoder=True, shared_dropout=False,
              use_huber=False, delta_schedule=False, adversarial=False),
    "D": dict(run_stage2=True, freeze_encoder=True, shared_dropout=True,
              use_huber=False, delta_schedule=False, adversarial=False),
    "E": dict(run_stage2=True, freeze_encoder=True, shared_dropout=True,
              use_huber=True, delta_schedule=False, adversarial=False),
    "F": dict(run_stage2=True, freeze_encoder=True, shared_dropout=True,
              use_huber=False, delta_schedule=True, adversarial=False),
    "G": dict(run_stage2=True, freeze_encoder=True, shared_dropout=True,
              use_huber=False, delta_schedule=False, adversarial=True),
    "H": dict(run_stage2=True, freeze_encoder=True, shared_dropout=True,
              use_huber=True, delta_schedule=True, adversarial=True),
}

if __name__ == "__main__":
    for row, flags in CONFIG_GRID.items():
        print(f"Row ({row}): {flags}")
