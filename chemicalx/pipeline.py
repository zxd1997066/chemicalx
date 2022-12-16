"""A collection of full training and evaluation pipelines."""

import collections.abc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Type, Union

import pandas as pd
import torch
from class_resolver import FunctionResolver, HintOrType
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from tabulate import tabulate
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.types import Device
from tqdm import trange

from chemicalx.data import DatasetLoader, dataset_resolver
from chemicalx.models import Model, model_resolver
from chemicalx.utils import resolve_device
from chemicalx.version import __version__

__all__ = [
    "Result",
    "pipeline",
]

metric_resolver = FunctionResolver([])
metric_resolver.register(roc_auc_score, synonyms={"roc_auc", "auc_roc", "auroc"})
metric_resolver.register(mean_squared_error, synonyms={"mse"})
metric_resolver.register(mean_absolute_error, synonyms={"mae"})


@dataclass
class Result:
    """A result package."""

    model: Model
    predictions: pd.DataFrame
    losses: List[float]
    train_time: float
    evaluation_time: float
    metrics: Mapping[str, float]

    def summarize(self) -> None:
        """Print results to the console."""
        print(tabulate(sorted(self.metrics.items()), headers=["Metric", "Value"]))

    def save(self, directory: Union[str, Path]) -> None:
        """Save the results to a directory."""
        if isinstance(directory, str):
            directory = Path(directory)
        directory = directory.resolve()
        directory.mkdir(exist_ok=True, parents=True)

        torch.save(self.model, directory.joinpath("model.pkl"))
        directory.joinpath("results.json").write_text(
            json.dumps(
                {
                    "evaluation": self.metrics,
                    "losses": self.losses,
                    "training_time": self.train_time,
                    "evaluation_time": self.evaluation_time,
                    "chemicalx_version": __version__,
                },
                indent=2,
            )
        )


def pipeline(
    *,
    dataset: HintOrType[DatasetLoader],
    model: HintOrType[Model],
    model_kwargs: Optional[Mapping[str, Any]] = None,
    optimizer_cls: Type[Optimizer] = torch.optim.Adam,
    optimizer_kwargs: Optional[Mapping[str, Any]] = None,
    loss_cls: Type[_Loss] = torch.nn.BCELoss,
    loss_kwargs: Optional[Mapping[str, Any]] = None,
    batch_size: int = 512,
    epochs: int,
    context_features: bool,
    drug_features: bool,
    drug_molecules: bool,
    train_size: Optional[float] = None,
    random_state: Optional[int] = None,
    metrics: Optional[Sequence[str]] = None,
    device: Device = None,
    args: Optional[dict] = None,
) -> Result:
    """Run the training and evaluation pipeline.

    :param dataset:
        The dataset can be specified in one of three ways:

        1. The name of the dataset
        2. A subclass of :class:`chemicalx.DatasetLoader`
        3. An instance of a :class:`chemicalx.DatasetLoader`
    :param model:
        The model can be specified in one of three ways:

        1. The name of the model
        2. A subclass of :class:`chemicalx.Model`
        3. An instance of a :class:`chemicalx.Model`
    :param model_kwargs:
        Keyword arguments to pass through to the model constructor. Relevant if passing model by string or class.
    :param optimizer_cls:
        The class for the optimizer to use. Currently defaults to :class:`torch.optim.Adam`.
    :param optimizer_kwargs:
        Keyword arguments to pass through to the optimizer construction.
    :param loss_cls:
        The loss to use. If none given, uses :class:`torch.nn.BCELoss`.
    :param loss_kwargs:
        Keyword arguments to pass through to the loss construction.
    :param batch_size:
        The batch size
    :param epochs:
        The number of epochs to train
    :param context_features:
        Indicator whether the batch should include biological context features.
    :param drug_features:
        Indicator whether the batch should include drug features.
    :param drug_molecules:
        Indicator whether the batch should include drug molecules
    :param train_size:
        The ratio of training triples. Default is 0.8 if None is passed.
    :param random_state:
        The random seed for splitting the triples. Default is 42. Set to none for no fixed seed.
    :param metrics:
        The list of metrics to use.
    :param device:
        The device to use
    :returns:
        A result object with the trained model and evaluation results
    """
    device = resolve_device(device)

    loader = dataset_resolver.make(dataset)
    train_generator, test_generator = loader.get_generators(
        batch_size=batch_size,
        context_features=context_features,
        drug_features=drug_features,
        drug_molecules=drug_molecules,
        train_size=train_size,
        random_state=random_state,
    )

    model = model_resolver.make(model, model_kwargs)
    model = model.to(device)
    # NHWC
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        print("---- Use NHWC model")
    optimizer = optimizer_cls(model.parameters(), **(optimizer_kwargs or {}))

    if not args.evaluate:
        print('Start training...')
        model.train()

        loss = loss_cls(**(loss_kwargs or {}))

        losses = []
        train_start_time = time.time()
        for _epoch in trange(epochs):
            for batch in train_generator:
                batch = batch.to(device)
                optimizer.zero_grad()
                prediction = model(*model.unpack(batch))
                loss_value = loss(prediction, batch.labels)
                losses.append(loss_value.item())
                loss_value.backward()
                optimizer.step()
        train_time = time.time() - train_start_time

    print('Start evaluate...')
    model.eval()
    total_time = 0.0
    total_sample = 0
    evaluation_start_time = time.time()
    predictions = []
    for i, batch in enumerate(test_generator):
        if args.num_iter > 0 and i >= args.num_iter: break
        elapsed = time.time()
        batch = batch.to(device)
        prediction = model(*model.unpack(batch))
        if torch.cuda.is_available(): torch.cuda.synchronize()
        elapsed = time.time() - elapsed
        if args.profile:
            args.p.step()
        print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
        if i >= args.num_warmup:
            total_time += elapsed
            total_sample += batch_size
        if isinstance(prediction, collections.abc.Sequence):
            prediction = prediction[0]
        prediction = prediction.detach().cpu().float().numpy()
        identifiers = batch.identifiers
        identifiers["prediction"] = prediction
        predictions.append(identifiers)

    throughput = total_sample / total_time
    latency = total_time / total_sample * 1000
    print('inference latency: %.3f ms' % latency)
    print('inference Throughput: %f images/s' % throughput)
    evaluation_time = time.time() - evaluation_start_time

    predictions_df = pd.concat(predictions)

    if metrics is None:
        metric_dict = {"roc_auc": roc_auc_score}
    else:
        metric_dict = {name: metric_resolver.lookup(name) for name in metrics}

    if not args.evaluate:
        return Result(
            model=model,
            predictions=predictions_df,
            losses=losses,
            train_time=train_time,
            evaluation_time=evaluation_time,
            metrics={
                name: func(predictions_df["label"], predictions_df["prediction"]) for name, func in metric_dict.items()
            },
        )
    else:
        return None
