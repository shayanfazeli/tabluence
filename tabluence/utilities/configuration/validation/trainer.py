from typing  import Dict, List


def validate_trainer_config(config: Dict) -> None:
    """
    Validates the trainer configuration.
    """
    assert 'optimizer' in config, "`optimizer` is missing in the configuration"
    assert 'task' in config, "`task` is missing in the configuration"
    assert "checkpointing" in config, "`checkpointing` is missing in the configuration"
    assert 'repo' in config['checkpointing'], "`repo` is missing in the `checkpointing` configuration"
    assert 'type' in config['task'], "`type` is missing in the `task` configuration"
    assert config['task']['type'] in ['classification', 'regression'], "`task` type must be either `classification` or `regression`"