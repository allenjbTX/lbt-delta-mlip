import os, json
import logging
import sys
from mpi4py import MPI
import argparse

import torch
import torch.distributed as dist

import hydragnn
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
from hydragnn.utils.input_config_parsing.config_utils import get_log_name_config
from hydragnn.utils.model import print_model
from hydragnn.utils.datasets.serializeddataset import (
    SerializedWriter,
    SerializedDataset,
)
from hydragnn.preprocess.load_data import split_dataset
from hydragnn.preprocess.single_npz_loader import NpzDataLoader
from hydragnn.preprocess.graph_samples_checks_and_updates import get_radius_graph_config
from hydragnn.train.train_validate_test import resolve_precision
from hydragnn.utils.model.model import load_existing_model

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosDataset
except ImportError:
    pass

import torch
import torch.distributed as dist

random_state = 42
torch.manual_seed(random_state)


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loadexistingsplit",
        action="store_true",
        help="load from existing serialized train/val/test splits",
    )
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only: serialize dataset and exit",
    )
    parser.add_argument(
        "--config",
        help="input JSON config file",
        type=str,
        default="delta_mlip.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="path to a .pk checkpoint file; loads model weights only (optimizer is reset)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--adios",
        help="Adios dataset",
        action="store_const",
        dest="format",
        const="adios",
    )
    group.add_argument(
        "--pickle",
        help="Pickle dataset",
        action="store_const",
        dest="format",
        const="pickle",
    )
    parser.set_defaults(format="pickle")

    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    input_filename = os.path.join(dirpwd, args.config)
    with open(input_filename, "r") as f:
        config = json.load(f)

    hydragnn.utils.print.setup_log(get_log_name_config(config))
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    comm = MPI.COMM_WORLD

    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    precision, param_dtype, _ = resolve_precision(
        config["NeuralNetwork"]["Training"].get("precision", "fp64")
    )
    torch.set_default_dtype(param_dtype)

    datasetname = config["Dataset"]["name"]

    if not args.loadexistingsplit and rank == 0:
        # NpzDataLoader.load_raw_data() writes to $SERIALIZED_DATA_PATH/serialized_dataset/
        os.environ["SERIALIZED_DATA_PATH"] = dirpwd

        loader = NpzDataLoader(config["Dataset"])
        loader.load_raw_data()
        total = loader.dataset_list[0]

        radius_graph = get_radius_graph_config(config["NeuralNetwork"]["Architecture"])
        total = [radius_graph(data) for data in total]

        trainset, valset, testset = split_dataset(
            dataset=total,
            perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
            stratify_splitting=False,
        )
        info(
            "total/train/val/test size: %d %d %d %d"
            % (len(total), len(trainset), len(valset), len(testset))
        )

        basedir = os.path.join(dirpwd, "serialized_dataset")
        SerializedWriter(
            trainset,
            basedir,
            datasetname,
            "trainset",
            minmax_node_feature=loader.minmax_node_feature,
            minmax_graph_feature=loader.minmax_graph_feature,
        )
        SerializedWriter(valset, basedir, datasetname, "valset")
        SerializedWriter(testset, basedir, datasetname, "testset")

    comm.Barrier()
    if args.preonly:
        sys.exit(0)

    timer = Timer("load_data")
    timer.start()

    basedir = os.path.join(dirpwd, "serialized_dataset")
    if args.format == "adios":
        info("Adios load")
        fname = os.path.join(dirpwd, "%s.bp" % datasetname)
        opt = {"preload": True, "shmem": False}
        trainset = AdiosDataset(fname, "trainset", comm, **opt)
        valset = AdiosDataset(fname, "valset", comm, **opt)
        testset = AdiosDataset(fname, "testset", comm, **opt)
    elif args.format == "pickle":
        info("Pickle load")
        trainset = SerializedDataset(basedir, datasetname, "trainset")
        valset = SerializedDataset(basedir, datasetname, "valset")
        testset = SerializedDataset(basedir, datasetname, "testset")
    else:
        raise ValueError("Unknown format: %s" % args.format)

    config["NeuralNetwork"]["Variables_of_interest"][
        "minmax_node_feature"
    ] = trainset.minmax_node_feature
    config["NeuralNetwork"]["Variables_of_interest"][
        "minmax_graph_feature"
    ] = trainset.minmax_graph_feature

    info(
        "trainset/valset/testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    (train_loader, val_loader, test_loader) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )
    timer.stop()

    config = hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )
    config["NeuralNetwork"]["Variables_of_interest"].pop("minmax_node_feature", None)
    config["NeuralNetwork"]["Variables_of_interest"].pop("minmax_graph_feature", None)

    verbosity = config["Verbosity"]["level"]
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    if rank == 0:
        print_model(model)
    comm.Barrier()

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    model, optimizer = hydragnn.utils.distributed.distributed_model_wrapper(
        model, optimizer, verbosity
    )
    if rank == 0:
        print_model(model)

    if args.checkpoint is not None:
        # Load model weights only; optimizer is intentionally reset so the new
        # loss weights (e.g. force_weight) start with unbiased moment estimates.
        ckpt_file = os.path.abspath(args.checkpoint)
        model_name = os.path.basename(os.path.dirname(ckpt_file))
        logs_dir = os.path.dirname(os.path.dirname(ckpt_file))
        load_existing_model(model, model_name, path=logs_dir, optimizer=None)
        if rank == 0:
            logging.info("Loaded checkpoint weights from: %s", ckpt_file)

    log_name = get_log_name_config(config)
    writer = hydragnn.utils.model.get_summary_writer(log_name)

    if dist.is_initialized():
        dist.barrier()

    hydragnn.utils.input_config_parsing.save_config(config, log_name)

    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        verbosity,
        create_plots=True,
        precision=precision,
        compute_grad_energy=config["NeuralNetwork"]["Training"][
            "compute_grad_energy"
        ]
    )

    hydragnn.utils.model.save_model(model, optimizer, log_name)
    hydragnn.utils.profiling_and_tracing.print_timers(verbosity)
    if writer is not None:
        writer.close()

    dist.destroy_process_group()
    sys.exit(0)
