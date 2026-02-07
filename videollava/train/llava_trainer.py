import os
import math
import json
import random
import pickle
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    # ShardedDDPOption,
    logger,
)
from transformers.training_args import ParallelMode
from transformers.utils import is_torch_tpu_available, is_torch_npu_available
from typing import Any, Dict, List, Optional, Union


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if getattr(self.args, "force_apex", False):
            if getattr(self.args, "bf16", False):
                logger.warning("force_apex requested but bf16 is enabled; skipping Apex fp16 setup.")
            else:
                self.use_apex = True

    def _load_rng_state(self, checkpoint):
        """Handle RNG state loading while working around PyTorch 2.6 weights_only defaults."""
        if checkpoint is None:
            return

        safe_globals_ctx = getattr(torch.serialization, "safe_globals", None)
        numpy_reconstruct = getattr(np.core.multiarray, "_reconstruct", None)

        if safe_globals_ctx is not None and numpy_reconstruct is not None:
            try:
                with safe_globals_ctx([numpy_reconstruct]):
                    return super()._load_rng_state(checkpoint)
            except pickle.UnpicklingError:
                logger.warning("torch.load RNG state failed even inside safe_globals context; retrying with weights_only=False")
            except Exception:
                raise

        try:
            return super()._load_rng_state(checkpoint)
        except pickle.UnpicklingError:
            self._load_rng_state_fallback_weights_only_false(checkpoint)

    def _load_rng_state_fallback_weights_only_false(self, checkpoint):
        if self.args.world_size > 1:
            process_index = self.args.process_index
            rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    f"Didn't find an RNG file for process {process_index}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed."
                )
                return

        try:
            checkpoint_rng_state = torch.load(rng_file, weights_only=False)
            logger.info("Loaded RNG state with weights_only=False fallback.")
        except TypeError:
            checkpoint_rng_state = torch.load(rng_file)
            logger.info("Loaded RNG state fallback without weights_only kwarg (older torch version).")
        except Exception as exc:
            logger.error(f"Failed to load RNG state from {rng_file} with weights_only=False: {exc}")
            raise

        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])

        if torch.cuda.is_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
            else:
                try:
                    torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])
                except Exception as e:
                    logger.info(
                        "Didn't manage to set back the RNG states of the GPU because of the following error:\n %s"
                        "\nThis won't yield the same results as if the training had not been interrupted.",
                        e,
                    )

        if is_torch_tpu_available():
            try:
                import torch_xla.core.xla_model as xm

                xm.set_rng_state(checkpoint_rng_state["xla"])
            except Exception as e:
                logger.info(
                    "Didn't manage to set back the RNG states of the TPU because of the following error:\n %s"
                    "\nThis won't yield the same results as if the training had not been interrupted.",
                    e,
                )

        if is_torch_npu_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                torch.npu.random.set_rng_state_all(checkpoint_rng_state["npu"])
            else:
                try:
                    torch.npu.random.set_rng_state(checkpoint_rng_state["npu"])
                except Exception as e:
                    logger.info(
                        "Didn't manage to set back the RNG states of the NPU because of the following error:\n %s"
                        "\nThis won't yield the same results as if the training had not been interrupted.",
                        e,
                    )

    def _should_monitor_grads(self, step: Optional[int] = None) -> bool:
        if not getattr(self.args, "monitor_grads", False):
            return False
        interval = max(1, getattr(self.args, "monitor_grads_every", 1))
        current_step = self.state.global_step if step is None else step
        return (current_step % interval == 0) and self.is_world_process_zero()

    def _log_loss_before_backward(self, loss: torch.Tensor) -> None:
        if not self._should_monitor_grads():
            return
        loss_value = loss.detach()
        try:
            loss_scalar = float(loss_value.item())
        except Exception:
            loss_scalar = float("nan")
        logger.info(f"[grad-monitor] step={self.state.global_step} loss={loss_scalar:.6f}")
        self._last_monitored_loss = loss_scalar

    def _log_gradient_statistics(self, model: torch.nn.Module) -> None:
        if not self._should_monitor_grads():
            return

        total_sq = 0.0
        sum_abs = 0.0
        max_abs = 0.0
        elem_count = 0

        for _, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            grad = param.grad.detach().float()
            total_sq += torch.sum(grad * grad).item()
            sum_abs += grad.abs().sum().item()
            max_abs = max(max_abs, grad.abs().max().item())
            elem_count += grad.numel()

        grad_total_norm = math.sqrt(total_sq) if total_sq > 0 else 0.0
        grad_mean_abs = (sum_abs / elem_count) if elem_count > 0 else 0.0
        loss_val = getattr(self, "_last_monitored_loss", float("nan"))

        logger.info(
            "[grad-monitor] step=%s grad_total_norm=%.6f grad_mean_abs=%.6e grad_max_abs=%.6e loss=%.6f",
            self.state.global_step,
            grad_total_norm,
            grad_mean_abs,
            max_abs,
            loss_val,
        )

    def _check_nan_inf_gradients(self, model: torch.nn.Module) -> None:
        if not self._should_monitor_grads():
            return

        any_bad = False
        raise_on_bad = getattr(self.args, "raise_on_nan_grad", True)
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            grad = param.grad
            if not torch.isfinite(grad).all():
                has_nan = torch.isnan(grad).any().item()
                has_inf = torch.isinf(grad).any().item()
                logger.error(
                    f"[grad-monitor] non-finite gradient detected: name={name} has_nan={has_nan} has_inf={has_inf}"
                )
                any_bad = True

        if any_bad and raise_on_bad:
            raise FloatingPointError("Detected non-finite (NaN/Inf) gradients after backward")

    def _dump_first_backward_grad_stats(self, model: torch.nn.Module) -> None:
        if not getattr(self.args, "dump_first_backward_grads", False):
            return
        if getattr(self, "_first_backward_grads_dumped", False):
            return
        if not self.is_world_process_zero():
            self._first_backward_grads_dumped = True
            return

        grad_stats: Dict[str, Dict[str, Any]] = {}

        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            grad = param.grad.detach().float()
            flat_grad = grad.view(-1)

            nan_count = torch.isnan(flat_grad).sum().item()
            inf_count = torch.isinf(flat_grad).sum().item()
            finite_mask = torch.isfinite(flat_grad)
            finite_grad = flat_grad[finite_mask]

            if finite_grad.numel() > 0:
                mean_val = finite_grad.mean().item()
                std_val = finite_grad.std(unbiased=False).item() if finite_grad.numel() > 1 else 0.0
                min_val = finite_grad.min().item()
                max_val = finite_grad.max().item()
                l2_norm = finite_grad.norm().item()
                abs_mean = finite_grad.abs().mean().item()
            else:
                mean_val = float("nan")
                std_val = float("nan")
                min_val = float("nan")
                max_val = float("nan")
                l2_norm = float("nan")
                abs_mean = float("nan")

            sample_count = min(10, flat_grad.numel())
            sample_values = flat_grad[:sample_count].tolist()

            grad_stats[name] = {
                "dtype": str(grad.dtype),
                "shape": list(grad.shape),
                "numel": grad.numel(),
                "mean": mean_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
                "l2_norm": l2_norm,
                "abs_mean": abs_mean,
                "nan_count": nan_count,
                "inf_count": inf_count,
                "sample": sample_values,
            }

        target_path = getattr(self.args, "dump_first_backward_grads_path", None)
        if not target_path:
            dump_dir = os.path.join(self.args.output_dir, "grad_monitor")
            os.makedirs(dump_dir, exist_ok=True)
            target_path = os.path.join(dump_dir, "first_backward_grads.json")
        else:
            dump_dir = os.path.dirname(target_path)
            if dump_dir != "":
                os.makedirs(dump_dir, exist_ok=True)

        with open(target_path, "w") as fp:
            json.dump(grad_stats, fp, indent=2)

        logger.info("[grad-monitor] wrote first backward gradient stats to %s", target_path)
        self._first_backward_grads_dumped = True

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        if is_sagemaker_mp_enabled():
            # Defer to the parent implementation for SageMaker Model Parallel to avoid breaking their hooks.
            return super().training_step(model, inputs)

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            print(f">>> loss = {loss}")
            if loss.isnan().any():
                print(f">>> loss is nan")
                raise ValueError("loss is nan")
            loss = loss.mean()

        self._log_loss_before_backward(loss)

        if self.use_apex:
            from apex import amp
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        self._check_nan_inf_gradients(model)
        self._log_gradient_statistics(model)
        self._dump_first_backward_grad_stats(model)

        return loss.detach() / self.args.gradient_accumulation_steps

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
        #     return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            #     self.optimizer = OSS(
            #         params=optimizer_grouped_parameters,
            #         optim=optimizer_cls,
            #         **optimizer_kwargs,
            #     )
            # else:
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            self.model.generation_config.do_sample = True            # <------------------------------------ Here
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            self.model.generation_config.do_sample = True  # <------------------------------------- Here, too
            super(LLaVATrainer, self)._save(output_dir, state_dict)