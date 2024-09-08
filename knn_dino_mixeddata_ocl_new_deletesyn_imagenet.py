# all imports

# buffer
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

# dataset
from avalanche.benchmarks import SplitMNIST, SplitCIFAR100
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.benchmarks.classic import SplitCIFAR10
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
from avalanche.training import Naive, CWRStar, Replay, GDumb, \
    Cumulative, LwF, GEM, AGEM, EWC, AR1
from avalanche.models import SimpleMLP
from avalanche.training.plugins import ReplayPlugin
from types import SimpleNamespace
from avalanche.training.storage_policy import ParametricBuffer, RandomExemplarsSelectionStrategy

import matplotlib.pyplot as plt
import numpy as np
from numpy import inf

# all imports

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


from typing import Iterable, Sequence, Optional, Union, List
from pkg_resources import parse_version

import torch
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from avalanche.benchmarks import CLExperience, CLStream
from avalanche.core import BaseSGDPlugin
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from avalanche.training.plugins.clock import Clock
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates.base import BaseTemplate, ExpSequence
from avalanche.models.utils import avalanche_model_adaptation
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader, \
    collate_from_data_or_kwargs
from avalanche.training.utils import trigger_plugins



# import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os

import os
import torch
from torchvision import transforms
from torchvision.utils import save_image


from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, ToTensor
from torchvision.utils import save_image

# ?

from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.benchmarks.classic import SplitCIFAR10

from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark

from avalanche.benchmarks.generators import filelist_benchmark, dataset_benchmark, \
                                            tensors_benchmark, paths_benchmark
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.logging import InteractiveLogger, TensorboardLogger, \
    WandBLogger, TextLogger

from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics, class_accuracy_metrics

from avalanche.training.plugins.checkpoint import CheckpointPlugin, \
    FileSystemCheckpointStorage
from avalanche.training.determinism.rng_manager import RNGManager

from avalanche.training import Naive, CWRStar, Replay, GDumb, \
    Cumulative, LwF, GEM, AGEM, EWC, AR1

# strategies
from avalanche.models import SimpleMLP
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

# training
from avalanche.training import Naive, CWRStar, Replay, GDumb, \
    Cumulative, LwF, GEM, AGEM, EWC, AR1

# strategies
from avalanche.models import SimpleMLP
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.training.plugins import ReplayPlugin

from types import SimpleNamespace
import shutil
import argparse
################################################################################################



from knn_dino_utils import *
from dino_models import *
from cl_knn_ocl_models_new_deletesyn import *

################################################################################################

# Configuration
CUDA_DEVICE = 1
LOG_FILE = 'logs/test.txt'
FILTER1 = '/storage3/enbo/saved_data/sdxl_llava_i2i_allimage10percentprompt_60real'
FILTER2 =  'saved_data/sdxl_llava_synfromreal_s8g2'
TRAIN_DATA_DESTINATION_FOLDER = '/scratch/local/ssd/enbo/saved_data/synrealallimage10percentprompt_synsyns8g2'
TEST_DATA_DESTINATION_FOLDER = 'saved_data/cifar_test100'


def parse_args():
    parser = argparse.ArgumentParser(description="Run CIFAR100 benchmark with different configurations")
    parser.add_argument('--cuda_device', type=int, default=0, help='CUDA device index')
    parser.add_argument('--knn_k', type=int, default=5, help='k-nearest neighbour')
    parser.add_argument('--log_file', type=str, default='logs/test.txt', help='File for logging')
    parser.add_argument('--syn_index', type=int, default=3, help='the starting index of synthetic class within each experience')
    parser.add_argument('--filter1', type=str, default='/storage3/enbo/saved_data/sdxl_llava_i2i_allimage10percentprompt_60real', help='File 1 for filter')
    parser.add_argument('--filter2', type=str, default='saved_data/sdxl_llava_synfromreal_s8g2', help='File 2 for filter')
    parser.add_argument('--train_destination', type=str, default='/scratch/local/ssd/enbo/saved_data/synrealallimage10percentprompt_synsyns8g2', help='Train data destination')
    parser.add_argument('--test_destination', type=str, default='saved_data/cifar_test100', help='Test data destination')
    return parser.parse_args()


################################################################################################

def setup_device(cuda_device):
    torch.cuda.set_device(cuda_device)
    if torch.cuda.is_available():
        current_gpu = torch.cuda.current_device()
        print(f"Current default GPU index: {current_gpu}")
        print(f"Current default GPU name: {torch.cuda.get_device_name(current_gpu)}")
    else:
        print("No GPUs available.")

def prepare_data_transform():
    return transforms.Compose([
        transforms.Resize((196, 196)),
        transforms.ToTensor(),
    ])


def load_data(transform):
    trainset = torchvision.datasets.CIFAR100(root='data', train=True, download=True, transform=transform)
    name_list = trainset.classes
    benchmark = SplitCIFAR100(n_experiences=20, seed=41)
    orders = benchmark.classes_order
    order_list = [orders[x:x+5] for x in range(0, len(orders), 5)]
    return trainset, name_list, benchmark, order_list


def prepare_class_dictionaries(order_list, name_list, syn_index):
    order_sample = [order[syn_index:] for order in order_list]
    integer_to_name = {i: name for i, name in enumerate(name_list)}
    
    classname_list = []
    label_list = []
    for order_l in order_sample:
        label_list.append(order_l)
        cur_classname = [integer_to_name[i] for i in order_l]
        classname_list.append(cur_classname)
    
    classname_list_sep = [item for sublist in classname_list for item in sublist]
    label_list_sep = [item for sublist in label_list for item in sublist]
    real_list = set(range(100)) - set(label_list_sep)
    
    syn_dict = {class_number: integer_to_name[class_number] for class_number in label_list_sep}
    real_dict = {class_number: integer_to_name[class_number] for class_number in real_list}
    return real_dict, syn_dict
    

def filter_and_combine_files(filter1, filter2, train_destination, real_dict, syn_dict):
    filter_class_txt_files_from2to1(
        filter1, 
        filter2,
        train_destination, 
        real_dict, 
        syn_dict
    )



def prepare_loggers(log_file):
    tb_logger = TensorboardLogger()
    text_logger = TextLogger(open(log_file, 'w'))
    interactive_logger = InteractiveLogger()
    return tb_logger, text_logger, interactive_logger


def prepare_evaluation_plugin(loggers):
    tb_logger, text_logger, interactive_logger = loggers
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        class_accuracy_metrics(minibatch=False, epoch=False, epoch_running=False, experience=False, stream=True),
        loggers=[interactive_logger, text_logger, tb_logger]
    )
    return eval_plugin



def prepare_strategy(eval_plugin, knn_k):
    RNGManager.set_random_seeds(1234)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_plugin = CheckpointPlugin(
        FileSystemCheckpointStorage(
        directory='./checkpoints/task_cifar',
    ),
    map_location=device
)
    
    strategy, initial_exp = checkpoint_plugin.load_checkpoint_if_exists()
    storage_p = Custom_ParametricBuffer(
        max_size=270000,
        groupby='class',
        selection_strategy=RandomExemplarsSelectionStrategy()
    )
    replay_plugin = KNN_storagePlugin_update(mem_size=270000, storage_policy=storage_p)
    dino_model = DINOFeatureExtractor_v2()

    cl_strategy = KNN_DINO_update_ocl_deleteold(
        model=dino_model,
        train_mb_size=512,
        train_epochs=1,
        eval_mb_size=512,
        device=device,
        evaluator=eval_plugin,
        plugins=[replay_plugin],
        k = knn_k
    )
    return cl_strategy

def main():
    setup_device(CUDA_DEVICE)
    
    transform = prepare_data_transform()
    trainset, name_list, benchmark, order_list = load_data(transform)
    
    real_dict, syn_dict = prepare_class_dictionaries(order_list, name_list)
    filter_and_combine_files(real_dict, syn_dict)
    
    if not os.path.exists(TRAIN_DATA_DESTINATION_FOLDER):
        os.makedirs(TRAIN_DATA_DESTINATION_FOLDER)
    
    train_experience_list = generate_experience_lists(order_list, TRAIN_DATA_DESTINATION_FOLDER)
    test_experience_list = generate_experience_lists(order_list, TEST_DATA_DESTINATION_FOLDER)
    
    transform_train = transform_test = transforms.Compose([
        transforms.Resize((196, 196)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4866, 0.4410], std=[0.1941, 0.1917, 0.1957])
    ])
    
    synthesis_cifar_benchmark = filelist_benchmark(
        None,
        train_file_lists=train_experience_list,
        test_file_lists=test_experience_list,
        task_labels=[0] * 20,
        train_transform=transform_train,
        eval_transform=transform_train
    )

    loggers = prepare_loggers()
    eval_plugin = prepare_evaluation_plugin(loggers)
    cl_strategy = prepare_strategy(eval_plugin)

    # Training
    print('Starting experiment...')
    results = []
    for experience in synthesis_cifar_benchmark.train_stream:
        print(experience)
        print(f"Start of experience: {experience.current_experience}")
        print(f"Current Classes: {experience.classes_in_this_experience}")

        res = cl_strategy.train(experience)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        test_re = cl_strategy.eval(synthesis_cifar_benchmark.test_stream)
        results.append(test_re)

def main1():
    args = parse_args()
    
    setup_device(args.cuda_device)
    
    transform = prepare_data_transform()
    trainset, name_list, benchmark, order_list = load_data(transform)
    real_dict, syn_dict = prepare_class_dictionaries(order_list, name_list, args.syn_index)
    filter_and_combine_files(args.filter1, args.filter2, args.train_destination, real_dict, syn_dict)
    
    if not os.path.exists(args.train_destination):
        os.makedirs(args.train_destination)
    
    train_experience_list = generate_experience_lists(order_list, args.train_destination)
    test_experience_list = generate_experience_lists(order_list, args.test_destination)
    
    transform_train = transform_test = transforms.Compose([
        transforms.Resize((196, 196)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4866, 0.4410], std=[0.1941, 0.1917, 0.1957])
    ])
    # Define dataset benchmark
    synthesis_cifar_benchmark = filelist_benchmark(
        None,
        train_file_lists=train_experience_list,
        test_file_lists=test_experience_list,
        task_labels=[0] * 20,
        train_transform=transform_train,
        eval_transform=transform_train
    )

    loggers = prepare_loggers(args.log_file)
    eval_plugin = prepare_evaluation_plugin(loggers)
    cl_strategy = prepare_strategy(eval_plugin, args.knn_k)

    # Training
    print('Starting experiment...')
    results = []
    for experience in synthesis_cifar_benchmark.train_stream:
        print(experience)
        print(f"Start of experience: {experience.current_experience}")
        print(f"Current Classes: {experience.classes_in_this_experience}")

        res = cl_strategy.train(experience)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        test_re = cl_strategy.eval(synthesis_cifar_benchmark.test_stream)
        results.append(test_re)



if __name__ == "__main__":
    main1()







