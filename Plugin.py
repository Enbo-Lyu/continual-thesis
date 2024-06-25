import random
import torch
import os
from torch import cat, Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset, ConcatDataset, TensorDataset
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import datasets, transforms
import torch.optim.lr_scheduler # ?
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, CenterCrop, RandomHorizontalFlip, Resize
from torchvision.transforms.functional import center_crop
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.utils import save_image
from torchvision.transforms.functional import pil_to_tensor

from typing import Optional, TYPE_CHECKING

from avalanche.benchmarks.utils import concat_classification_datasets
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import (
    ExemplarsBuffer,
    ExperienceBalancedBuffer,
)
# dataset
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.benchmarks.utils.data_loader import GroupBalancedDataLoader, ReplayDataLoader
from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark
from avalanche.benchmarks.generators import filelist_benchmark, dataset_benchmark, \
                                            tensors_benchmark, paths_benchmark

from avalanche.logging import InteractiveLogger, TensorboardLogger, \
    WandBLogger, TextLogger, TensorboardLogger

from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics

from avalanche.training.plugins.checkpoint import CheckpointPlugin, \
    FileSystemCheckpointStorage
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.training import Naive, CWRStar, Replay, GDumb, \
    Cumulative, LwF, GEM, AGEM, EWC, AR1
from avalanche.models import SimpleMLP
from avalanche.training.plugins import ReplayPlugin
from types import SimpleNamespace
from avalanche.training.storage_policy import ParametricBuffer, RandomExemplarsSelectionStrategy
from collections import defaultdict
import random
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Generic,
    Optional,
    List,
    TYPE_CHECKING,
    Set,
    TypeVar,
)

from avalanche.benchmarks.utils import (
    classification_subset,
    AvalancheDataset,
)
from avalanche.models import FeatureExtractorBackbone
# from ..benchmarks.utils.utils import concat_datasets
from avalanche.benchmarks.utils import concat_datasets
from avalanche.training.storage_policy import ReservoirSamplingBuffer, BalancedExemplarsBuffer, ClassBalancedBuffer

from avalanche.training.storage_policy import ParametricBuffer, RandomExemplarsSelectionStrategy, ExemplarsBuffer, ExperienceBalancedBuffer
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins import SupervisedPlugin
from typing import Optional, TYPE_CHECKING

from avalanche.benchmarks.utils import concat_classification_datasets
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate, BaseSGDTemplate
# all imports

def sample_Avadataset(dataset, percentage= 0.1):
    """random sample in certain percentage, percentage should be float"""
    data_len = len(dataset)
    indices = list(range(data_len))
    random.shuffle(indices)
    sample_num = int(data_len * percentage)
    sampled_indices = random.sample(indices, sample_num)

    new_buffer_data = dataset.subset(sampled_indices)

    return new_buffer_data




class ReplayPlugin_method1(SupervisedPlugin):
    """
    method1: fixed replay
    """

    def __init__(
        self,
        mem_size: int = 200,
        batch_size: Optional[int] = None,
        batch_size_mem: Optional[int] = None,
        task_balanced_dataloader: bool = False,
        storage_policy: Optional["ExemplarsBuffer"] = None,
    ):
        super().__init__()
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader

        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == self.mem_size
        else:  # Default
            self.storage_policy = ExperienceBalancedBuffer(
                max_size=self.mem_size, adaptive_size=True
            )

    def before_training_exp(
        self,
        strategy: "SupervisedTemplate",
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        **kwargs
    ):
        """
        Dataloader to build batches containing examples from both memories and
        the training dataset
        """
        buffer_size = len(self.storage_policy.buffer)
        print("buffer size: " + str(buffer_size))
        num_class = len(self.storage_policy.buffer_datasets)
        print("current class number in replay buffer: " + str(num_class))
        
        if len(self.storage_policy.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return

        batch_size = self.batch_size
        if batch_size is None:
            batch_size = strategy.train_mb_size

        batch_size_mem = self.batch_size_mem
        if batch_size_mem is None:
            batch_size_mem = strategy.train_mb_size

        assert strategy.adapted_dataset is not None

        other_dataloader_args = dict()

        if "ffcv_args" in kwargs:
            other_dataloader_args["ffcv_args"] = kwargs["ffcv_args"]

        if "persistent_workers" in kwargs:
            if parse(torch.__version__) >= parse("1.7.0"):
                other_dataloader_args["persistent_workers"] = kwargs[
                    "persistent_workers"
                ]
        strategy.dataloader = ReplayDataLoader(
                strategy.adapted_dataset,
                self.storage_policy.buffer,
                batch_size=strategy.train_mb_size,
                num_workers=num_workers,
                shuffle=shuffle,
            )


    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        self.storage_policy.update(strategy, **kwargs)
        
        
class CustomReplay_method2(SupervisedPlugin):
    """
    method2: resample replay
    """
    def __init__(
        self,
        mem_size: int = 200,
        batch_size: Optional[int] = None,
        batch_size_mem: Optional[int] = None,
        task_balanced_dataloader: bool = False,
        storage_policy: Optional["ExemplarsBuffer"] = None,
        # The policy that controls how to add new exemplars in memory
                        #
    ):
        super().__init__()
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader

        self.storage_policy = storage_policy
        assert storage_policy.max_size == self.mem_size



    def before_training_exp(self,
                            strategy: "SupervisedTemplate",
                            num_workers: int = 0,
                            shuffle: bool = True,
                            **kwargs):
        """ Here we set the dataloader. create batch with examples from both
        training data and external memory"""
        if len(self.storage_policy.buffer) == 0:
            return

        # replay dataloader samples mini-batches from the memory and current
        # data separately and combines them together.
        print("Override the dataloader.")

        buffer_size = len(self.storage_policy.buffer)
        print("buffer size: " + str(buffer_size))
        num_class = len(self.storage_policy.buffer_datasets)
        print("current class number in replay buffer: " + str(num_class))
        # for item in self.storage_policy.buffer:
        # tempory_buffer = AvalancheDataset([])

        tempory_buffer = []

        for i, dataset in enumerate(self.storage_policy.buffer_datasets):
            sub_data = sample_Avadataset(dataset)
            if i == 0:
                tempory_buffer = sub_data
            else:
                tempory_buffer = tempory_buffer.concat(sub_data)


        # buffer = [sample_Avadataset(dataset) for dataset in self.storage_policy.buffer_datasets]
        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            # self.storage_policy.buffer,
            tempory_buffer,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle)
        # for x, y, t in dl:
        #     print(t.tolist())
        #     print(len(t.tolist()))

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        """ We update the buffer after the experience.
            You can use a different callback to update the buffer in a different place
        """
        print("Buffer update.")
        self.storage_policy.update(strategy, **kwargs)

