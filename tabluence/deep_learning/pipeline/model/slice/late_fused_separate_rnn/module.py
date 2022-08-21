from typing import List, Dict, Any, Tuple

import numpy
import pandas
import torch
import torch.nn

from tabluence.deep_learning.pipeline.model.slice.base import SliceModelBase
from tabluence.utilities.configuration.validation import late_fused_separate_rnn_validate_model_config as validate_model_config
from tabluence.deep_learning.utilities.rnn.pack_unpack import pack_single_slice_batch_for_rnn


class LateFusedSeparateRNNSliceModel(SliceModelBase):
    """
    An example of a config for setting up an instance of :cls:`LateFusedSeparateRNNSliceModel`:

    ```python
    model = LateFusedSeparateRNNSliceModel(
        config=dict(
            branches=dict(
                daily=dict(
                    rnn_model="LSTM",
                    rnn_args=dict(
                        input_size=20,
                        hidden_size=32,
                        bidirectional=True,
                        batch_first=True,
                        bias=False,
                        dropout=0.2
                    ), #torch args
                    project_args=dict(
                        input_dim=64,
                        projection_dim=32
                    ),  # will be projected to this dimension if not None.
                ),
                respiration=dict(
                    rnn_model="LSTM",
                    rnn_args=dict(
                        input_size=2,
                        hidden_size=4,
                        bidirectional=True,
                        batch_first=True,
                        bias=False,
                        dropout=0.2
                    ), #torch args
                    project_args=dict(
                        input_dim=8,
                        projection_dim=4
                    ),  # will be projected to this dimension if not None.
                ),
                pulseOx=dict(
                    rnn_model="LSTM",
                    rnn_args=dict(
                        input_size=2,
                        hidden_size=4,
                        bidirectional=True,
                        batch_first=True,
                        bias=False,
                        dropout=0.2
                    ), #torch args
                    project_args=dict(
                        input_dim=8,
                        projection_dim=4
                    ),  # will be projected to this dimension if not None.
                ),
                stress=dict(
                    rnn_model="LSTM",
                    rnn_args=dict(
                        input_size=2,
                        hidden_size=4,
                        bidirectional=True,
                        batch_first=True,
                        bias=False,
                        dropout=0.2
                    ), #torch args
                    project_args=dict(
                        input_dim=8,
                        projection_dim=4
                    ),  # will be projected to this dimension if not None.
                )
            ),
            aggregation=dict(
                method="concatenate", # options are `mean` (this means all the branch reps have to be the same), `concatenate`
                project_args=dict(
                    input_dim=44,
                    projection_dim=50), # the output of the given `method` will be projected to it (if not None).
            ),
            task=dict(
                target_in_meta='overall_quantized_stress_value',
                type='classification',
                label_layout=[0.0,
                              0.2571428571428571,
                              0.5142857142857142,
                              0.7714285714285714,
                              1.0285714285714285,
                              1.2857142857142856,
                              1.5428571428571427,
                              1.7999999999999998,
                              2.057142857142857,
                              2.314285714285714,
                              2.571428571428571,
                              2.8285714285714283,
                              3.0857142857142854,
                              3.3428571428571425,
                              3.5999999999999996],
                loss_class='CrossEntropyLoss',
                loss_args=dict(),
            )
        )
    )
    ```

    One other note to point out is that we offer the option for "regulatory" losses. For instance,
    the following are those that are supported by this class:

    * `bottleneck_id_prediction`:
    The corresponding configuration:

    ```python
    regulatory_tasks = [
        dict(
            name="bottleneck_id_prediction",
            type="classification",
            target_in_meta="subject_id",
            label_layout=[f'SWS_{i:02d}' for i in range(1, 10)],
            loss_class='CrossEntropyLoss',
            loss_args=dict(),
            objective='maximize',
            weight=1.0
    ]
    ```
    """
    def __init__(self, *args, **kwargs):
        super(LateFusedSeparateRNNSliceModel, self).__init__(*args, **kwargs)
        self.aggregation_method = self.config['aggregation']['method']
        self.sanity_checks()

    def sanity_checks(self) -> None:
        """
        Post-initialization sanity checks
        """
        validate_model_config(self.config)

    def building_blocks(self) -> None:
        self.build_loss()
        self.softmax = torch.nn.Softmax(dim=-1)
        for branch in self.config['branches'].keys():
            self.add_module(f"branch_{branch}_rnn", getattr(torch.nn, self.config['branches'][branch]['rnn_model'])(**self.config['branches'][branch]['rnn_args']))
            if 'project_args' in self.config['branches'][branch]:
                self.add_module(
                    f"branch_{branch}_projection",
                    torch.nn.Sequential(
                        torch.nn.Linear(self.config['branches'][branch]['project_args']['input_dim'], self.config['branches'][branch]['project_args']['projection_dim']),
                        torch.nn.Dropout(p=0.2),
                        torch.nn.ReLU(),
                        torch.nn.Linear(self.config['branches'][branch]['project_args']['projection_dim'], self.config['branches'][branch]['project_args']['projection_dim']),
                    ))
                branch_final_dim = self.config['branches'][branch]['project_args']['projection_dim']
        self.add_module(
            f"aggregation_projection",
            torch.nn.Sequential(
                torch.nn.Linear(self.config['aggregation']['project_args']['input_dim'], self.config['aggregation']['project_args']['projection_dim'],),
                torch.nn.Dropout(p=0.2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.config['aggregation']['project_args']['projection_dim'], self.config['aggregation']['project_args']['projection_dim'],),
            ))
        self.process_config_for_possible_regulatory_tasks()

        if self.config['task']['type'] == 'classification':
            self.add_module("final_projection", torch.nn.Sequential(
                torch.nn.Linear(self.config['aggregation']['project_args']['projection_dim'], len(self.config['task']['label_layout']))
            ))
        elif self.config['task']['type'] == 'regression':
            self.add_module("final_projection", torch.nn.Sequential(
                torch.nn.Linear(self.config['aggregation']['project_args']['projection_dim'], self.config['task']['regression_arms'])
            ))
        else:
            raise NotImplementedError()

    def process_config_for_possible_regulatory_tasks(self) -> None:
        """
        Processing the configuration for the possible existence of regulatory tasks, currently as supervision
        or regression, enforced on the latent representations (specific to this model).

        The idea of regulatory tasks is to remove/reinforce the presence/absence of certain information
        from network bottleneck. This method, and how (+where in the pipeline) it is applied,
        is specific to this module.
        """
        if 'regulatory_tasks' in self.config:
            self.regulatory_task_losses = []
            for j, task in enumerate(self.config['regulatory_tasks']):
                input_dimension = self.config['aggregation']['project_args']['projection_dim']
                if task['type'] == 'classification':
                    self.regulatory_task_losses.append(
                        getattr(torch.nn, task['loss_class'])(**task['loss_args'])
                    )
                    self.add_module(
                        f"fc_regulatory_{j}",
                        torch.nn.Sequential(
                            torch.nn.LayerNorm(input_dimension),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(p=0.5),
                            torch.nn.Linear(input_dimension, input_dimension),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(p=0.5),
                            torch.nn.Linear(input_dimension, len(task['label_layout'])),
                        )
                    )
                elif task['type'] == 'regression':
                    self.regulatory_task_losses.append(
                        getattr(torch.nn, task['loss_class'])(**task['loss_args'])
                    )
                    self.add_module(
                        f"fc_regulatory_{j}",
                        torch.nn.Sequential(
                            torch.nn.LayerNorm(input_dimension),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(p=0.5),
                            torch.nn.Linear(input_dimension, input_dimension),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(p=0.5),
                            torch.nn.Linear(input_dimension, task['regression_arms']),
                        )
                    )
                else:
                    raise ValueError(f"Regulatory task {task} is not supported.")
            if len(self.regulatory_task_losses) == 0:
                self.regulatory_task_losses = None
        else:
            self.regulatory_task_losses = None

    def build_loss(self) -> None:
        """
        building the loss from config
        """
        self.criterion = getattr(torch.nn, self.config['task']['loss_class'])(**self.config['task']['loss_args'])

    def initialize_weights(self) -> None:
        for n, p in self.named_parameters():
            if 'weight' in n:
                torch.nn.init.xavier_uniform_(p)
            elif 'bias' in n:
                torch.nn.init.zeros_(p)
            elif 'void_rep' in n:
                torch.nn.init.xavier_uniform_(p)
            else:
                pass

    def get_branch_rnn_representations(self, branch_inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        branch_inputs: `Dict[str, torch.Tensor]`, required
            shape: (batch_size, max_num_timesteps, input_dim)
            The packed/processed inputs for the sequential processing branches

        Returns
        -------
        `Tuple[torch.Tensor, torch.Tensor]`: The  first element is the branch representations, of dimensions:
        `(batch_size, final_rep_dim)`. The second element indicates the branch repdims (currently not used).
        """
        # - preparing the placeholders
        branch_representations = []
        branch_repdims = []

        # - getting the batch_size
        batch_size = numpy.max([len(x) for x in branch_inputs.values()])
        assert batch_size > 0, "at least one sequence shall be non-empty"

        for branch in self.config['branches'].keys():
            tmp_outputs = getattr(self, f"branch_{branch}_rnn")(branch_inputs[branch].float())
            D = 2 if getattr(self, f"branch_{branch}_rnn").bidirectional else 1
            if isinstance(getattr(self, f"branch_{branch}_rnn"), torch.nn.LSTM):
                h_n = tmp_outputs[1][0]
                _, batch_size, rep_dim = h_n.shape
                h_n = h_n.reshape(D, -1, batch_size, rep_dim)
                h_n = h_n[:, -1, :, :].permute(1, 0, 2).flatten(1)  # shape: (batch_size, new rep_dim)

                if 'project_args' in self.config['branches'][branch]:
                    h_n = getattr(self, f"branch_{branch}_projection")(h_n)

                branch_repdims.append(h_n.shape[-1])
                branch_representations.append(h_n)
            else:
                raise NotImplementedError()

        if self.aggregation_method == 'mean':
            assert len(set(branch_repdims)) == 1, "all the dims must be the same for all branches"
            branch_representations = torch.mean(torch.stack(branch_representations, dim=0), dim=0)
        elif self.aggregation_method == 'concatenate':
            branch_representations = torch.cat(branch_representations, dim=-1)
        else:
            raise NotImplementedError()
        branch_repdims = torch.LongTensor(branch_repdims).to(branch_representations.device)
        return branch_representations, branch_repdims

    def preprocess_batch(self, batch: Dict[str, List[Dict[str, pandas.DataFrame]]]) -> Dict[str, Any]:
        """
        Processing batch for this model:
        * tensorization
        * rnn-packing of sequences (empty sequences will be set as a minus one tensor of length 1-element)
        * getting the targets from the metadata following the `config`.

        Parameters
        ----------
        batch: `Dict[str, List[Dict[str, pandas.DataFrame]]]`, required
            The batch to be processed.

        Returns
        -------
        `Dict[str, Any]`: The processed batch.
        """
        # - bringing in the tools needed for the preprocessing
        batch = self.tensorizer(batch)
        branch_inputs = pack_single_slice_batch_for_rnn(batch=batch, data_sources_to_pack=list(self.config['branches'].keys()))
        batch['branch_inputs'] = branch_inputs

        for k in batch['slice'][0].values():
            device = k.device
            break

        if self.config['task']['type'] == 'classification':
            y = torch.Tensor([self.config['task']['label_layout'].index(meta[self.config['task']['target_in_meta']]) for meta in batch['meta']]).long().to(device)
        elif self.config['task']['type'] == 'regression':
            y = torch.Tensor([meta[self.config['task']['target_in_meta']] for meta in batch['meta']]).float().to(device)
        else:
            raise NotImplementedError()
        batch['targets'] = y

        if self.regulatory_task_losses is not None:
            batch['regulatory_targets'] = []
            for j, task in enumerate(self.config['regulatory_tasks']):
                batch['regulatory_targets'].append(torch.Tensor([
                    task['label_layout'].index(meta[task['target_in_meta']])
                    for meta in batch['meta']]).long().to(device))
        return batch

    def loss(self, batch, model_outputs) -> Dict[str, torch.Tensor]:
        """
        Loss computation for this module.

        Parameters
        ----------
        batch: `Dict[str, Any]`, required
            The batch to be used for computing the loss.
        model_outputs: `Dict[str, torch.Tensor]`, required
            The model outputs to be used for computing the loss.

        Returns
        -------
        `Dict[str, torch.Tensor]`: The various losses for the batch.
        """
        loss_outputs = dict(
            loss=self.criterion(model_outputs['logits'], batch['targets']),
        )
        if self.regulatory_task_losses is not None:
            for j, task in enumerate(self.config['regulatory_tasks']):
                if task['objective'] == 'maximize':
                    coeff = -1
                elif task['objective'] == 'minimize':
                    coeff = +1
                else:
                    raise ValueError(f"Unknown objective {task['objective']}")
                loss_outputs[f'regulatory_loss_{j}'] = coeff * task['weight'] * self.regulatory_task_losses[j](model_outputs[f'regulatory_logits'][j], batch['regulatory_targets'][j])

        return loss_outputs

    def inference_main(self, batch: Any) -> Dict[str, Any]:
        """
        Main inference function for this model.

        Parameters
        ----------
        batch: `Dict[str, Any]`, required
            The batch to be used for inference.

        Returns
        -------
        `Dict[str, Any]`: The inference results, including the following:
            * `latent_representations`: `torch.Tensor` of shape `(batch_size, num_latent_dims)`
            * `logits`: `torch.Tensor` of shape `(batch_size, 1 or num_classes)` depending on the task
            * `regulatory_logits`: `List[torch.Tensor]` of length `num_regulatory_tasks`. This key only exists
            if `regulatory_tasks` is provided as a key in the configurations.
            * `targets`: `torch.Tensor` of shape `(batch_size, 1 or num_classes)` depending on the task
            * 'y_hat': numpy version of predicted indices or values for classification or regression (the target estimates)
        """
        # - passing the information to the rnns
        branch_representations, branch_repdims = self.get_branch_rnn_representations(
            branch_inputs=batch['branch_inputs'])
        # - passing the information to the aggregation layer
        latent_representations = getattr(self, f"aggregation_projection")(branch_representations)

        # - getting the logits by passing the latent representations through the final linear layer
        logits = self.final_projection(latent_representations)

        if self.config['task']['type'] == 'classification':
            y_hat = torch.argmax(logits, dim=-1)
        elif self.config['task']['type'] == 'regression':
            y_hat = logits
        else:
            raise NotImplementedError()

        output = {
            'latent_representations': latent_representations,
            'logits': logits,
            'targets': batch['targets'],
            'y_hat': y_hat,
            'y_score': self.softmax(logits)
        }

        if self.regulatory_task_losses is not None:
            output['regulatory_logits'] = []
            for j, task in enumerate(self.config['regulatory_tasks']):
                output['regulatory_targets'] = batch['regulatory_targets'][j]
                output['regulatory_logits'].append(
                    getattr(self, f"fc_regulatory_{j}")(latent_representations))

        return output

    def inference_eval(self, batch: Any) -> Dict[str, Any]:
        with torch.no_grad():
            return self.inference_main(batch=batch)

    def inference_train(self, batch: Any) -> Dict[str, Any]:
        return self.inference_main(batch=batch)
