from modelforge.config import TrainingConfig, LoRAConfig

def test_default_config():
    cfg = TrainingConfig()
    assert cfg.num_epochs == 3
    assert cfg.lora.r == 16

def test_custom_lora():
    lora = LoRAConfig(r=64, lora_alpha=128)
    assert lora.r == 64
    assert lora.lora_alpha == 128
