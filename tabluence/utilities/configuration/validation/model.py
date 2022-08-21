from typing  import Dict, List


def early_fused_single_rnn_validate_model_config(config: Dict) -> None:
    """
    Validates the model configuration.
    """
    assert 'single_source' in config, 'Model configuration must contain `single_source` indicating the name of the single source on which the model must operate.'
    assert 'task' in config, 'Model configuration must contain `task`'
    for k in ['target_in_meta', 'type', 'loss_class', 'loss_args']:
        assert k in config['task'], f'`task` value in the model configuration must contain `{k}`'
    if config['task']['type'] == 'classification':
        assert 'label_layout' in config['task'], '`task` value in the model configuration must contain `label_layout`'
    elif config['task']['type'] == 'regression':
        assert 'regression_arms' in config['task'], '`task` value in the model configuration must contain `regression_arms`'
    regulatory_tasks_validation(config=config)


def late_fused_separate_rnn_validate_model_config(config: Dict) -> None:
    """
    Validates the model configuration.
    """
    assert 'branches' in config, 'Model configuration must contain `branches`'
    assert 'aggregation' in config, 'Model configuration must contain `aggregation`'
    assert 'task' in config, 'Model configuration must contain `task`'
    for k in ['target_in_meta', 'type', 'loss_class', 'loss_args']:
        assert k in config['task'], f'`task` value in the model configuration must contain `{k}`'
    if config['task']['type'] == 'classification':
        assert 'label_layout' in config['task'], '`task` value in the model configuration must contain `label_layout`'
    elif config['task']['type'] == 'regression':
        assert 'regression_arms' in config['task'], '`task` value in the model configuration must contain `regression_arms`'
    regulatory_tasks_validation(config=config)


def regulatory_tasks_validation(config: Dict) -> None:
    if 'regulatory_tasks' in config:
        assert type(config['regulatory_tasks']) == list, "regulatory_tasks must be a list"
        assert len(config['regulatory_tasks']) > 0, "regulatory_tasks must be a list of at least one task"
        for task in config['regulatory_tasks']:
            for k in ['type', 'target_in_meta', 'label_layout', 'loss_class', 'loss_args', 'objective', 'coefficient']:
                assert k in task, f"regulatory_tasks must contain {k}"
            assert task['type'] in ['classification', 'regression'], f"regulatory_tasks type {task['type']} is not supported"
            if task['type'] == 'classification':
                assert 'label_layout' in task, f"regulatory_tasks must contain label_layout"
            elif task['type'] == 'regression':
                assert 'regression_arms' in task, f"regulatory_tasks must contain regression_arms"
            else:
                raise ValueError(f"regulatory_tasks type {task['type']} is not supported")