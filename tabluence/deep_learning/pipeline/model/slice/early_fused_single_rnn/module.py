from typing import List, Dict, Any, Tuple

import numpy
import pandas
import torch
import torch.nn
import torch.nn.functional as F
from tabluence.deep_learning.pipeline.model.slice.base import SliceModelBase
from tabluence.utilities.configuration.validation import early_fused_single_rnn_validate_model_config as validate_model_config
from tabluence.deep_learning.utilities.rnn.pack_unpack import pack_single_slice_batch_for_rnn
import tabluence.deep_learning.optimization.loss

class EarlyFusedSingleRNNSliceModel(SliceModelBase):
    """
    An example of a config for setting up an instance of :cls:`EarlyFusedSingleRNNSliceModel`:

    """
    def __init__(self, *args, **kwargs):
        super(EarlyFusedSingleRNNSliceModel, self).__init__(*args, **kwargs)
        self.sanity_checks()

    def sanity_checks(self) -> None:
        """
        Post-initialization sanity checks
        """
        validate_model_config(self.config)

    def process_config_for_possible_regulatory_tasks(self) -> None:
        """
        Processing the configuration for the possible existence of regulatory tasks, currently as supervision
        or regression, enforced on the latent representations (specific to this model).

        The idea of regulatory tasks is to remove/reinforce the presence/absence of certain information
        from network bottleneck. This method, and how (+where in the pipeline) it is applied,
        is specific to this module.
        """
        latent_rep_dim = self.config['main_rnn']['project_args']['projection_dim']
        if 'regulatory_tasks' in self.config:
            self.regulatory_task_losses = []
            for j, task in enumerate(self.config['regulatory_tasks']):
                self.regulatory_task_losses.append(
                    getattr(tabluence.deep_learning.optimization.loss, task['loss_class'])(**task['loss_args'])
                )
                if task['type'] == 'classification':
                    self.add_module(
                        f"fc_regulatory_{j}",
                        torch.nn.Sequential(
                            torch.nn.LayerNorm(latent_rep_dim),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(p=0.5),
                            torch.nn.Linear(latent_rep_dim, latent_rep_dim),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(p=0.5),
                            torch.nn.Linear(latent_rep_dim, len(task['label_layout'])),
                        )
                    )
                elif task['type'] == 'regression':
                    self.add_module(
                        f"fc_regulatory_{j}",
                        torch.nn.Sequential(
                            torch.nn.LayerNorm(latent_rep_dim),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(p=0.5),
                            torch.nn.Linear(latent_rep_dim, latent_rep_dim),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(p=0.5),
                            torch.nn.Linear(latent_rep_dim, task['regression_arms']),
                        )
                    )
                else:
                    raise ValueError(f"Regulatory task {task} is not supported.")
            if len(self.regulatory_task_losses) == 0:
                self.regulatory_task_losses = None
        else:
            self.regulatory_task_losses = None


        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.negative_log_likelihood_loss = torch.nn.NLLLoss()

    def building_blocks(self) -> None:
        self.build_loss()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.add_module('initial_layernorm', torch.nn.LayerNorm(self.config['main_rnn']['rnn_args']['input_size']))
        self.add_module("main_rnn", getattr(torch.nn, self.config['main_rnn']['rnn_model'])(**self.config['main_rnn']['rnn_args']))
        if 'project_args' in self.config['main_rnn'].keys():
            self.add_module(
                "fc_latent",
                torch.nn.Sequential(
                    torch.nn.Linear(self.config['main_rnn']['project_args']['input_dim'], self.config['main_rnn']['project_args']['projection_dim']),
                    torch.nn.Dropout(p=0.2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.config['main_rnn']['project_args']['projection_dim'], self.config['main_rnn']['project_args']['projection_dim']),
                ))
            latent_rep_dim = self.config['main_rnn']['project_args']['projection_dim']
        else:
            raise "Currently, the projection layer must be specified for the main rnn. Please modify the config accordingly."

        if self.config['task']['type'] == 'classification':
            self.add_module("fc_logit", torch.nn.Sequential(
                torch.nn.Linear(latent_rep_dim, len(self.config['task']['label_layout']))
            ))
        elif self.config['task']['type'] == 'regression':
            self.add_module("final_projection", torch.nn.Sequential(
                torch.nn.Linear(latent_rep_dim, self.config['task']['regression_arms'])
            ))
        else:
            raise NotImplementedError()

        self.process_config_for_possible_regulatory_tasks()

    def build_loss(self) -> None:
        """
        building the loss from config
        """
        self.criterion = getattr(torch.nn, self.config['task']['loss_class'])(**self.config['task']['loss_args'])

    def initialize_weights(self) -> None:
        for n, p in self.named_parameters():
            if 'initial_layernorm' in n:
                pass
            elif 'weight' in n:
                try:
                    torch.nn.init.xavier_uniform_(p)
                except Exception as e:
                    continue
            elif 'bias' in n:
                torch.nn.init.zeros_(p)
            elif 'void_rep' in n:
                torch.nn.init.xavier_uniform_(p)
            else:
                pass

    def apply_main_rnn(self, rnn_input: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
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
        assert len(rnn_input.keys()) == 1, "Only one branch is supported for the `earlyfusedsinglernn` model"
        rnn_input = rnn_input[list(rnn_input.keys())[0]]  # packed tensor of shape (batch_size, max_num_timesteps, input_dim)
        tmp_outputs = self.main_rnn(rnn_input.float())
        D = 2 if self.main_rnn.bidirectional else 1
        if isinstance(self.main_rnn, torch.nn.LSTM):
            h_n = tmp_outputs[1][0]
            _, batch_size, rep_dim = h_n.shape
            h_n = h_n.reshape(D, -1, batch_size, rep_dim)
            h_n = h_n[:, -1, :, :].permute(1, 0, 2).flatten(1)  # shape: (batch_size, new rep_dim)

            if 'project_args' in self.config['main_rnn'].keys():
                h_n = self.fc_latent(h_n)
            else:
                raise Exception('The projection layer must be specified for the main rnn. Please modify the config accordingly.')
            latent_representation = h_n
        else:
            raise NotImplementedError()

        return latent_representation

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

        # for i in range(len(batch['slice'])):
        #     batch['slice'][i][self.config['single_source']] = self.initial_layernorm(batch['slice'][i][self.config['single_source']].float())

        branch_inputs = pack_single_slice_batch_for_rnn(batch=batch, data_sources_to_pack=[self.config['single_source']])
        batch['rnn_input'] = branch_inputs

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
                assert task['type'] == 'classification', f"the model currently does not support {task['type']}"
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
                if 'adversarial' not in task:
                    task['adversarial'] = False

                if task['objective'] == 'maximize':
                    coeff = -1
                elif task['objective'] == 'minimize':
                    coeff = +1
                else:
                    raise ValueError(f"Unknown objective {task['objective']}")

                if not task['adversarial']:
                    loss_outputs[f'regulatory_loss_{j}'] = coeff * task['coefficient'] * self.regulatory_task_losses[j](model_outputs[f'regulatory_logits'][j], batch['regulatory_targets'][j])
                elif task['type'] == 'classification':
                    logits_discriminator, logits_generator = model_outputs[f'regulatory_logits'][j]
                    criterion = self.regulatory_task_losses[j]

                    assert isinstance(criterion, torch.nn.CrossEntropyLoss), "currently, only crossentropyloss is supported for adversarial classification regulatory task."
                    targets = torch.eye(logits_discriminator.shape[-1]).to(logits_generator.device)[batch['regulatory_targets'][j]]
                    loss_outputs[f'regulatory_loss_{j}'] = coeff * task['coefficient'] * (
                            self.cross_entropy_loss(logits_discriminator, targets) -
                            (- ( 1. / float(logits_generator.shape[0])) * torch.sum((1 - self.softmax(logits_generator)).log() * targets))  #  generator loss
                    )
                else:
                    raise NotImplementedError()

        return loss_outputs

    def inference_main(self, batch: Any) -> Dict[str, Any]:
        # - passing the information to the rnn
        latent_representations = self.apply_main_rnn(
            rnn_input=batch['rnn_input']) # of shape (batch_size, final_rep_dim)

        # - getting the logits by passing the latent representations through the final linear layer
        logits = self.fc_logit(latent_representations)

        if self.config['task']['type'] == 'classification':
            y_hat = torch.argmax(logits, dim=-1)
        elif self.config['task']['type'] == 'regression':
            y_hat = logits
        else:
            raise NotImplementedError()

        outputs = {
            'latent_representations': latent_representations,
            'logits': logits,
            'targets': batch['targets'],
            'y_hat': y_hat,
            'y_score': self.softmax(logits)
        }

        if self.regulatory_task_losses is not None:
            outputs['regulatory_logits'] = []
            for j, task in enumerate(self.config['regulatory_tasks']):
                outputs['regulatory_targets'] = batch['regulatory_targets'][j].data.cpu().numpy()
                if 'adversarial' not  in task:
                    task['adversarial'] = False
                if task['adversarial']:
                    logits_discriminator = getattr(self, f"fc_regulatory_{j}")(latent_representations.detach())
                    # - temporarily  freezing
                    self.freeze_by_name(name=f"fc_regulatory_{j}")
                    logits_generator = getattr(self, f"fc_regulatory_{j}")(latent_representations)
                    self.unfreeze_by_name(name=f"fc_regulatory_{j}")
                    outputs['regulatory_logits'].append((logits_discriminator, logits_generator))
                else:
                    outputs['regulatory_logits'].append(
                        getattr(self, f"fc_regulatory_{j}")(latent_representations))

        return outputs

    def freeze_by_name(self, name):
        for n, p in self.named_parameters():
            if name in n:
                p.requires_grad = False

    def unfreeze_by_name(self, name):
        for n, p in self.named_parameters():
            if name in n:
                p.requires_grad = True

    def inference_eval(self, batch: Any) -> Dict[str, Any]:
        with torch.no_grad():
            return self.inference_main(batch=batch)

    def inference_train(self, batch: Any) -> Dict[str, Any]:
        return self.inference_main(batch=batch)
