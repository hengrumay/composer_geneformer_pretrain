

##1 Set parameters

import datetime

# imports
import os

import pickle
import random
import subprocess
import inspect

import numpy as np
import pytz

import boto3

import torch
from torch.utils.data import DataLoader

from transformers import BertConfig, BertForMaskedLM

from composer.models.huggingface import HuggingFaceModel
from composer.utils import reproducibility
from composer import Trainer
from composer import Callback, Event, Logger, State

from streaming import StreamingDataset

from omegaconf import DictConfig

from cfgutils import *

def _env_truthy(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default)
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _get_serverless_gpu_distributed():
    """Best-effort import for Databricks Serverless GPU @distributed decorator."""
    try:
        from serverless_gpu.launcher import distributed  # type: ignore
        return distributed
    except Exception:
        try:
            from serverless_gpu import distributed  # type: ignore
            return distributed
        except Exception:
            return None


def _maybe_init_torch_distributed():
    """Optionally init torch.distributed if the launcher only sets env vars.

    Some frameworks (and some example notebooks) explicitly call
    `torch.distributed.init_process_group("nccl")` after `@distributed(...)` has
    provisioned workers + set WORLD_SIZE/RANK/LOCAL_RANK.
    """
    try:
        import torch.distributed as dist  # type: ignore
    except Exception:
        return

    try:
        world_size = int(os.getenv("WORLD_SIZE", "1") or "1")
    except Exception:
        world_size = 1

    if world_size <= 1:
        return

    try:
        local_rank = int(os.getenv("LOCAL_RANK", "0") or "0")
    except Exception:
        local_rank = 0

    # Pin this process to its GPU.
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    except Exception:
        pass

    # Initialize process group if needed.
    try:
        if dist.is_available() and not dist.is_initialized():
            backend = os.getenv("SERVERLESS_GPU_DDP_BACKEND", "nccl")
            dist.init_process_group(backend=backend)
    except Exception as e:
        # Don't hard-fail: Composer/Trainer may initialize the process group itself.
        print(f"WARNING: torch.distributed init_process_group failed/skipped: {e}")


def _get_dist_rank_world_size():
    """Return (rank, world_size) if torch.distributed is initialized else (0, 1)."""
    try:
        import torch.distributed as dist  # type: ignore
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()), int(dist.get_world_size())
    except Exception:
        pass
    return 0, 1


def _streaming_shard_kwargs(rank: int, world_size: int, seed: int):
    """Build StreamingDataset kwargs for rank-aware sharding, if supported by this version."""
    kwargs = {}
    if world_size <= 1:
        return kwargs

    try:
        sig = inspect.signature(StreamingDataset.__init__)  # type: ignore[name-defined]
        params = sig.parameters

        # Common MosaicML StreamingDataset sharding params (vary by version).
        if "num_canonical_nodes" in params:
            kwargs["num_canonical_nodes"] = world_size
        if "canonical_node_rank" in params:
            kwargs["canonical_node_rank"] = rank
        elif "node_rank" in params:
            kwargs["node_rank"] = rank

        # Make shuffle deterministic across ranks if supported.
        if "shuffle_seed" in params:
            kwargs["shuffle_seed"] = seed
    except Exception:
        # If inspection fails, keep kwargs empty and rely on StreamingDataset internal dist detection.
        return {}

    return kwargs


def _as_long_tensor(x):
    """Ensure token ids are int64 tensors (required by HF MLM collator)."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.long)
    return torch.tensor(x, dtype=torch.long)

def _mlm_collate_fn(
    features,
    *,
    vocab_size: int,
    mask_token_id: int,
    pad_token_id: int,
    mlm_probability: float,
):
    """Minimal BERT-style MLM collator (avoids heavy geneformer dependency tree)."""
    # Pad variable-length sequences to max length in batch.
    seqs = [_as_long_tensor(f["input_ids"]).view(-1) for f in features]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        seqs, batch_first=True, padding_value=pad_token_id
    )

    labels = input_ids.clone()

    # Do not mask padding.
    probability_matrix = torch.full(labels.shape, mlm_probability, dtype=torch.float)
    probability_matrix = probability_matrix.masked_fill(input_ids.eq(pad_token_id), 0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100
    # Ignore loss on padding tokens as well.
    labels = labels.masked_fill(input_ids.eq(pad_token_id), -100)

    # 80% -> [MASK]
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8, dtype=torch.float)).bool()
        & masked_indices
    )
    input_ids[indices_replaced] = mask_token_id

    # 10% -> random token
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5, dtype=torch.float)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]

    # 10% -> keep original
    return {"input_ids": input_ids, "labels": labels}




def main(cfg: DictConfig):
    #### Env variables
    #os.environ["NCCL_DEBUG"] = "INFO"

    # Optional: when running under Databricks Serverless GPU @distributed launcher, initialize
    # torch.distributed if it isn't already initialized (mirrors common examples).
    if _env_truthy("USE_SERVERLESS_GPU_DISTRIBUTED", "0") and _env_truthy(
        "SERVERLESS_GPU_MANUAL_INIT_PROCESS_GROUP", "1"
    ):
        _maybe_init_torch_distributed()

    seed_val = cfg.seed_val
    random.seed(seed_val)
    np.random.seed(seed_val)

    working_dir = cfg.working_dir
    data_bucket_name = cfg.get("data_bucket_name", None)
    data_bucket_key = cfg.get("data_bucket_key", None)

    token_dictionary_filename = cfg.get("token_dictionary_filename", "token_dictionary.pkl")
    remote_data_dir = f"s3://{data_bucket_name}/{data_bucket_key}" if data_bucket_name and data_bucket_key else None
    streaming_dataset_location = cfg.streaming_dataset_location

    # batch size for training and eval
    train_batch_size = cfg.train_batch_size  #<< This is per device batch size
    eval_batch_size = cfg.eval_batch_size
    mlm_probability = cfg.mlm_probability

    remote_streaming_dataset_location = (
        f"{remote_data_dir}/{streaming_dataset_location}" if remote_data_dir else None
    )
    local_streaming_dataset_location = f"{cfg.local_data_dir}/{streaming_dataset_location}"
    streaming_dataset_cache_location = f"{working_dir}/streaming/cache"

    if cfg.data_location == "local":
        data_local = True
    else:
        data_local = False

    # output directories

    #############################################
    ### Start processing
    reproducibility.configure_deterministic_mode()
    reproducibility.seed_all(seed_val)

    loggers = [
        build_logger(name, logger_cfg)
        for name, logger_cfg in cfg.get('loggers', {}).items()
    ]

    # Callbacks
    callbacks = [
        build_callback(name, callback_cfg)
        for name, callback_cfg in cfg.get('callbacks', {}).items()
    ]

    # Algorithms
    algorithms = [
        build_algorithm(name, algorithm_cfg)
        for name, algorithm_cfg in cfg.get('algorithms', {}).items()
    ]
    # Read the token dictionary file
    if data_local:
        token_dictionary_path = cfg.get("token_dictionary_path", None)
        if not token_dictionary_path:
            # Common layout for Volumes: <...>/geneformer/data/dataset (local_data_dir)
            # token dictionary at the parent directory: <...>/geneformer/data/token_dictionary.pkl
            token_dictionary_path = os.path.join(os.path.dirname(cfg.local_data_dir), token_dictionary_filename)
        with open(token_dictionary_path, "rb") as f:
            token_dictionary = pickle.load(f)
        print(f"Loaded token dictionary from local path: {token_dictionary_path}")
    else:
        if not data_bucket_name or not data_bucket_key:
            raise ValueError("Remote data_location requires data_bucket_name and data_bucket_key")
        s3 = boto3.resource("s3")
        token_dictionary = pickle.loads(
            s3.Bucket(data_bucket_name).Object(f"{data_bucket_key}/{token_dictionary_filename}").get()["Body"].read()
        )

    ### Load model
    model_config = build_model_config(cfg,token_dictionary)

    print("=============================")
    print(model_config)

    config = BertConfig(**model_config)
    model = BertForMaskedLM(config)
    model.train()
    print(model)

    #Create streaming dataset
    rank, world_size = _get_dist_rank_world_size()
    if world_size > 1:
        print(f"Distributed context detected: rank={rank} world_size={world_size}")
    shard_kwargs = _streaming_shard_kwargs(rank=rank, world_size=world_size, seed=seed_val)
    if shard_kwargs:
        print(f"StreamingDataset sharding kwargs: {shard_kwargs}")

    if data_local:
        streaming_dataset_train = StreamingDataset(
            local=f"{local_streaming_dataset_location}/train",
            batch_size=train_batch_size,
            **shard_kwargs,
        )
        streaming_dataset_eval = StreamingDataset(
            local=f"{local_streaming_dataset_location}/test",
            batch_size=eval_batch_size,
            **shard_kwargs,
        )
    else:
        if remote_streaming_dataset_location is None:
            raise ValueError("Remote data_location requires a valid remote_data_dir")
        streaming_dataset_train = StreamingDataset(
            remote=f"{remote_streaming_dataset_location}/train",
            local=f"{streaming_dataset_cache_location}/train",
            batch_size=train_batch_size,
            **shard_kwargs,
        )
        streaming_dataset_eval = StreamingDataset(
            remote=f"{remote_streaming_dataset_location}/test",
            local=f"{streaming_dataset_cache_location}/test",
            batch_size=eval_batch_size,
            **shard_kwargs,
        )

    #Prepare composer model
    composer_model = HuggingFaceModel(model)

    # Build optimizer
    optimizer = build_optimizer(cfg.optimizer, model)

    # Scheduler
    scheduler = build_scheduler(cfg.scheduler)

    pad_token_id = int(token_dictionary.get("<pad>", 0))
    mask_token_id = token_dictionary.get("<mask>", None)
    if mask_token_id is None:
        raise ValueError('Token dictionary is missing "<mask>" token id required for MLM.')
    mask_token_id = int(mask_token_id)
    vocab_size = len(token_dictionary)

    def collate_fn(features):
        return _mlm_collate_fn(
            features,
            vocab_size=vocab_size,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            mlm_probability=mlm_probability,
        )

    train_dataloader = DataLoader(streaming_dataset_train,
                            shuffle=False, 
                            drop_last=False, 
                            collate_fn=collate_fn,
                            batch_size=train_batch_size,
                            num_workers = 32,
                            pin_memory = True,
                            persistent_workers = True)

    eval_dataloader = DataLoader(streaming_dataset_eval,
                            shuffle=False, 
                            drop_last=False, 
                            collate_fn=collate_fn,
                            batch_size=eval_batch_size,
                            num_workers = 32,
                            pin_memory = True,
                            persistent_workers = True)

    ##############################
    #Following code is to introduce an error after 7 epochs , 
    # to see if we can restart the training from 5th epoch
    #
    #class RaiseErrorOnEpoch7(Callback):
    #    def run_event(self, event: Event, state: State, logger: Logger):
    #        if event == Event.EPOCH_START and state.timestamp.epoch==7:
    #            raise Exception("Rescue me!!!!")
            
    #callbacks.append(RaiseErrorOnEpoch7())
    ##############################

    # Create Trainer Object
    trainer = Trainer(
        #run_name=cfg.run_name,
        model=composer_model, 
        algorithms=algorithms,
        train_dataloader=train_dataloader,    
        eval_dataloader=eval_dataloader,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        optimizers=optimizer,
        schedulers=[scheduler],
        device=cfg.get("device", "gpu"),
        device_train_microbatch_size=cfg.get("device_train_microbatch_size","auto"),
        save_folder=cfg.get("save_folder", None),
        save_interval=cfg.get("save_interval", "5ep"),
        save_overwrite=cfg.get("save_overwrite", False),
        save_num_checkpoints_to_keep=cfg.get("save_num_checkpoints_to_keep",1),
        train_subset_num_batches=cfg.get("train_subset_num_batches", -1),
        eval_subset_num_batches=cfg.get("eval_subset_num_batches", -1),
        autoresume=cfg.get("autoresume", False),
        #Load path required only for manual restarts
        #load_path=cfg.get("load_path", None),
        #load_weights_only=cfg.get("load_weights_only", False),
        python_log_level=cfg.get("python_log_level", None),
        seed=seed_val,        
        fsdp_config = cfg.get("fsdp_config", None),
        loggers=loggers,
        callbacks=callbacks,

    )
    # Start training
    trainer.fit()

    print(trainer.state.train_metrics)
    print(trainer.state.eval_metrics)



    print("*************Done")


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]

    def _run_from_yaml(yaml_path: str, args_list: list[str]):
        with open(yaml_path) as f:
            yaml_cfg = om.load(f)
        cli_cfg = om.from_cli(args_list)
        cfg = om.merge(yaml_cfg, cli_cfg)
        cfg = cast(DictConfig, cfg)  # for type checking
        main(cfg)

    # Optional Databricks Serverless GPU launcher mode:
    # - Enable with USE_SERVERLESS_GPU_DISTRIBUTED=1 (set in workload.yaml env_variables).
    # - Configure with:
    #   - SERVERLESS_GPU_GPUS (default: 1)
    #   - SERVERLESS_GPU_GPU_TYPE (optional, e.g. "h100_80gb")
    #   - SERVERLESS_GPU_REMOTE (default: false)
    if _env_truthy("USE_SERVERLESS_GPU_DISTRIBUTED", "0"):
        distributed = _get_serverless_gpu_distributed()
        if distributed is None:
            raise RuntimeError(
                "USE_SERVERLESS_GPU_DISTRIBUTED=1 but serverless_gpu is not importable. "
                "This mode requires Databricks Serverless GPU runtime (serverless_gpu package)."
            )

        gpus = int(os.getenv("SERVERLESS_GPU_GPUS", "1"))
        gpu_type = os.getenv("SERVERLESS_GPU_GPU_TYPE", "").strip() or None
        remote = _env_truthy("SERVERLESS_GPU_REMOTE", "false")

        kwargs = {"gpus": gpus, "remote": remote}
        if gpu_type is not None:
            kwargs["gpu_type"] = gpu_type

        wrapped = distributed(**kwargs)(_run_from_yaml)
        # Some implementations expose `.distributed(...)`; others are callable directly.
        if hasattr(wrapped, "distributed"):
            wrapped.distributed(yaml_path, args_list)
        else:
            wrapped(yaml_path, args_list)
    else:
        _run_from_yaml(yaml_path, args_list)
