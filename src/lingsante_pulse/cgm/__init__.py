"""Continuous-glucose representation, training and evaluation branch."""


def extract(
    payload, checkpoint_dir=None, cohort="external", device="cpu", memory_device="cpu"
):
    """Extract CGM features using the bundled deployment weights by default."""
    from .inference.pipeline import ROOT, extract as run

    return run(
        payload,
        ROOT / "checkpoints" if checkpoint_dir is None else checkpoint_dir,
        cohort,
        device,
        memory_device,
    )


__all__ = ["extract"]
