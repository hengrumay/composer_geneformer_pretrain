

##1 Set parameters

import datetime

# imports
import os

import pickle
import random
import subprocess

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

    if data_local:
        streaming_dataset_train = StreamingDataset(local=f"{local_streaming_dataset_location}/train" ,batch_size=train_batch_size)
        streaming_dataset_eval = StreamingDataset(local=f"{local_streaming_dataset_location}/test" ,batch_size=eval_batch_size)        
    else:
        if remote_streaming_dataset_location is None:
            raise ValueError("Remote data_location requires a valid remote_data_dir")
        streaming_dataset_train = StreamingDataset(
            remote=f"{remote_streaming_dataset_location}/train",
            local=f"{streaming_dataset_cache_location}/train",
            batch_size=train_batch_size,
        )
        streaming_dataset_eval = StreamingDataset(
            remote=f"{remote_streaming_dataset_location}/test",
            local=f"{streaming_dataset_cache_location}/test",
            batch_size=eval_batch_size,
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
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    cfg = cast(DictConfig, cfg)  # for type checking
    main(cfg)
