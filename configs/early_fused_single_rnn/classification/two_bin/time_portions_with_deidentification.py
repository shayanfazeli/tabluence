__code_root = '/home/shayan/phoenix/tabluence/tabluence/'
__warehouse_root = '/home/shayan/warehouse/tabluence/'

__logdir = __warehouse_root + 'early_fused_single_rnn/classification/two_bin/time_portions/all_with_deidentification'

__feature_names_per_data_source = dict(
    daily=[
        'heart_rate_tsvalue',
    ],
    pulseOx=[
        'spo2_tsvalue'
    ],
    respiration=[
        'epoch_to_breath_tsvalue'
    ],
    stress=[
        'stress_level_tsvalue'
    ]
)
__dataset_root_dir = __code_root + 'resources/warrior_wellness/Analysis/local_repo/'
__dataset_cache_filepath = __code_root + 'resources/smartwatch_study/dataset_cache/window_1hour_stride_1second.pkl.gz'
__data_manager_cache_filepath = __code_root + 'resources/smartwatch_study/dataset_cache/window_1hour_stride_1second-datamanger001-1_15wo10.pkl.gz'

__target_variable = 'overall_quantized_stress_value'
__overall_stress_quantization_bins = [0.0, 0.5, 10.0]
__specific_stress_quantization_bins = [0.0, 0.5, 10.0]
__timeline_portions = dict(train=0.8, test=0.2)

data = dict(
    interface='smartwatch_study_single_slice_dataloaders',
    args=dict(batch_size=50,
              data_manager_cache_filepath=__data_manager_cache_filepath,
              root_dir=__dataset_root_dir,
              subject_splits=dict(
                  train=[f'SWS_{e:02d}' for e in range(1, 15) if e != 10],
                  test=[f'SWS_{e:02d}' for e in range(1, 15) if e != 10]
              ),
              dataset_config=dict(
                  slice_lengths=[3600],
                  slice_time_step=1,
                  label_milestone_per_window=1.0,
                  metadata_cache_filepath=__dataset_cache_filepath,
                  no_cache=False,
                  parallel_threads=10,
                  overall_stress_quantization_bins=__overall_stress_quantization_bins,
                  specific_stress_quantization_bins=__specific_stress_quantization_bins
              ),
              sampler_configs=dict(
                  train=dict(
                      negative_sample_count=10000,
                      positive_sample_count=10000,
                      target_variable=__target_variable,
                      split_name="train",
                      timeline_portions=__timeline_portions
                  ),
                  test=dict(
                      negative_sample_count=1000,
                      positive_sample_count=1000,
                      target_variable=__target_variable,
                      split_name="test",
                      timeline_portions=__timeline_portions
                  )
              )
              )
)

dataside_pipeline=dict(
    type='StandardDataSidePipeline',
    args=dict(
        module_configs=[
            dict(
                type='MinMaxSingleSliceNormalization',
                lib='preprocessing',
                config=dict(
                    feature_names_per_data_source=__feature_names_per_data_source
                )
            ),
            dict(
                type='SliceToSliceFusion',
                lib='fusion',
                config=dict(
                    timestamp_column='utc_timestamp',
                    sources=dict(
                        all_timeseries=dict(
                            daily=['heart_rate_tsvalue'],
                            pulseOx=['spo2_tsvalue'],
                            respiration=['epoch_to_breath_tsvalue'],
                            stress=['stress_level_tsvalue'])
                    ),
                    nan_fill_method=['ffill', 'bfill', 'fill_constant_0']
                )
            )
        ]
    )
)

tensorizer = dict(
    type='CustomTensorizer',
    config=dict(
        timestamp_column='utc_timestamp',
        value_config=dict(
            all_timeseries=dict(
                bring=[
                    'heart_rate_tsvalue',
                    'spo2_tsvalue',
                    'epoch_to_breath_tsvalue',
                    'stress_level_tsvalue'
                ]
            ),
        )
    )
)

model = dict(
    type='EarlyFusedSingleRNNSliceModel',
    config=dict(
        single_source='all_timeseries',
        main_rnn=dict(
            rnn_model="LSTM",
            rnn_args=dict(
                input_size=4,
                num_layers=4,
                hidden_size=128,
                bidirectional=True,
                batch_first=True,
                bias=False,
                dropout=0.2
            ),
            project_args=dict(
                input_dim=256,
                projection_dim=128
            ),  # will be projected to this dimension if not None.
        ),
        task=dict(
            target_in_meta='overall_quantized_stress_value',
            type='classification',
            loss_class='CrossEntropyLoss',
            loss_args=dict(),
        ),
        regulatory_tasks=[
            dict(
                type='classification',
                target_in_meta='subject_id',
                label_layout=[f'SWS_{e:02d}' for e in range(1, 15) if e != 10],
                objective='minimize',
                loss_class='CrossEntropyLoss',
                loss_args=dict(),
                coefficient=1.,
                adversarial=True
            )
        ]
    )
)

trainer = dict(
    type='StandardTrainer',
    config=dict(
        optimizer=dict(
            type="Adam",
            args=dict(
                lr=0.001,
                weight_decay=1e-4,
            )
        ),
        epoch_scheduler=dict(
            type="CosineAnnealingLR",
            args=dict(
                T_max=20,
                eta_min=0,
                last_epoch=-1
            )
        ),
        max_epochs=20,
        checkpointing=dict(
            checkpointing_interval=50,
            repo=__logdir,
        ),
    ),
)
