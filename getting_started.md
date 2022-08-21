# Getting Started

## Writing a Configuration File
#### Step 1: Specifying the dataset/dataloaders

The first thing that a configuration file needs is the specification of datasets (original datasets).
In here, I give an example of our smartwatch study single slice dataset.
Please note that you can refer to `nowoe.deep_learning.data.dataset` for more details regarding available interfaces.
The code structure is designed in a way that for each dataset, developers are expected to create a 
subpackage (e.g., in this case the `nowoe.deep_learning.data.dataset.smartwatch_study.single_slice` which is a child of single slice datasets), in which
many the entire details are contained. However, for the encapsulation purposes, only the interface that
can provide the full dictionary of dataloaders (of type `Dict[str, torch.utils.data.dataloader.DataLoader]`) is required to be made
available outside of this package.
By refering to `nowoe.deep_learning.data.dataset`, you can see all the interfaces that are available. Alternatively,
you can follow the descriptions above to create your own.

Now, in the config file, you have to create the `data=dict()` config and in the `interface` key, give the name of this
interface, while passing all of the required information for it as `args` as shown below:
```python
data = dict(
    interface='smartwatch_study_single_slice_dataloaders',
    args=dict(batch_size=50,
                root_dir=__dataset_root_dir,
                subject_splits=dict(
                    train=[f'SWS_{e:02d}' for e in range(1, 10)],
                    test=[f'SWS_{e:02d}' for e in range(10, 15)]
                ),
                dataset_config=dict(
                    slice_lengths=[3600],
                    slice_time_step=(5 * 60),
                    label_milestone_per_window=1.0,
                    metadata_cache_filepath=__dataset_root_dir + 'metadata_cache.pkl',
                    no_cache=False,
                    parallel_threads=10
                ),
                sampler_configs=dict(
                    train=dict(
                        negative_sample_count=100,
                        positive_sample_count=50,
                        target_variable='overall_quantized_stress_value',
                        split_name="train"
                    ),
                    test=dict(
                        negative_sample_count=100,
                        positive_sample_count=50,
                        target_variable='overall_quantized_stress_value',
                        split_name="test"
                    )
                )
        )
)
```

todo: put the cache files for dataset definitions somewhere


#### Step 2: Specifying the dataside pipeline
In almost all cases of using a dataset in a machine learning-based inference pipeline, especially if neural networks are involved,
one would need many dataside operations to be conducted prior to the data being fed to the model.
In that regard, we have categorized these operations as the following groups:
* Augmentation [`nowoe.deep_learning.data.pipeline.augmentation`]
* Fusion [`nowoe.deep_learning.data.pipeline.fusion`]
* Preprocessing [`nowoe.deep_learning.data.pipeline.preprocessing`]

Hence, one can create a pipeline (similar to `torch.nn.Sequential`) as the configuration's
`dataside_pipeline` key. Here is an instance of how it is done:

```python
dataside_pipeline=dict(
    type='StandardDataSidePipeline',
    args=dict(
        module_configs=[
            dict(
                type='MinMaxSingleSliceNormalization',
                lib='preprocessing',
                config=dict()
            ),
            dict(
                type='GaussianMixturesSingleSliceAugmentation',
                lib='augmentation',
                config=dict(
                    gmm=dict(
                        n_components=2,
                        covariance_type='full'
                    )
                )
            )
        ]
    )
)
```




### Step 3: Model

Defining the model is perhaps the most critical step in determining the pipeline's operation.
To develop your own models, please refer to the documentations related to the
`nowoe.deep_learning.pipeline.model` package.

An example:
```python
model = dict(
    type='LateFusedSeparateRNNSliceModel',
    config=dict(
        branches=dict(
            daily=dict(
                rnn_model="LSTM",
                rnn_args=dict(
                    input_size=20,
                    hidden_size=64,
                    bidirectional=True,
                    batch_first=True,
                    bias=False,
                    dropout=0.5
                ),
                project_args=dict(
                    input_dim=128,
                    projection_dim=32
                ),  # will be projected to this dimension if not None.
            ),
            respiration=dict(
                rnn_model="LSTM",
                rnn_args=dict(
                    input_size=2,
                    hidden_size=8,
                    bidirectional=True,
                    batch_first=True,
                    bias=False,
                    dropout=0.2
                ),
                project_args=dict(
                    input_dim=16,
                    projection_dim=4
                ),  # will be projected to this dimension if not None.
            ),
            pulseOx=dict(
                rnn_model="LSTM",
                rnn_args=dict(
                    input_size=2,
                    hidden_size=8,
                    bidirectional=True,
                    batch_first=True,
                    bias=False,
                    dropout=0.2
                ),
                project_args=dict(
                    input_dim=16,
                    projection_dim=4
                ),  # will be projected to this dimension if not None.
            ),
            stress=dict(
                rnn_model="LSTM",
                rnn_args=dict(
                    input_size=2,
                    hidden_size=8,
                    bidirectional=True,
                    batch_first=True,
                    bias=False,
                    dropout=0.2
                ),
                project_args=dict(
                    input_dim=16,
                    projection_dim=4
                ),  # will be projected to this dimension if not None.
            )
        ),
    )
)
```


### Step 4: Trainer
Specifying the trainer is also an important step in designing the experiment. An example
is shown below:

```python
trainer = dict(
    config=dict(
        optimizer=dict(
            type="Adam",
            args=dict(
                lr=0.0001,
                weight_decay=0.0001
            )
        ),
        checkpointing=dict(
            checkpointing_interval=5,
            repo=__logdir,
        ),
    ),
)
```

Please note that critical information regarding optimization and checkpointing are configured by passing
the trainer configuration.

### Step 5: Run
The `tabluence_train` script is the main entry point for training. Please
refer to the documentation related to the arguments.
An example:

```bash
tabluence_train \
./configs/late_fused_separate_rnns/classification/two_bin/base.py \
--device=1 \
--seed=20 \
#--clean
```

### Extra: Evaluation
The `--eval` can indicate to the `tabluence_train` script to evaluate the model, which will use the `_latest` checkpoints.


