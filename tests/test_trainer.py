from modelforge.config import TrainingConfig
from modelforge.data import _synthetic_samples
from modelforge.trainer import train

def test_train_completes():
    cfg = TrainingConfig(num_epochs=1, max_samples=10, per_device_train_batch_size=2)
    samples = _synthetic_samples(10)
    run = train(cfg, samples)
    assert run.best_loss < 2.6
    assert run.end_time is not None
    assert len(run.metrics) > 0
