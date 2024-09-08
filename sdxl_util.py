import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.utils import load_image
import torchvision
import torch
import os
import json
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
import random
import re


def read_json(json_path):
# Step 1: Open the JSON file
    with open(json_path, 'r') as file:
    # Step 2: Load the JSON data
        data = json.load(file)
    return data

def extract_prompts_fromtxts(file_path):
    # for the caption with 'USER' as start, used for prompt before using llava pipeline
    prompts = []  # List to hold the prompts
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() != 'USER':  # Ignore lines that are just 'USER'
                # Split the line at the first colon and strip whitespace from the prompt
                parts = line.split(':', 1)
                if len(parts) > 1:
                    prompt = parts[1].strip()
                    prompts.append(prompt)
    return prompts

def extract_prompts_fromtxts_real2syn(file_path):
    path = []
    prompts = []  # List to hold the prompts
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split(':', 1)
            if len(parts) > 1:
                img_path = parts[0].strip()
                prompt = parts[1].strip()
                path.append(img_path)
                prompts.append(prompt)
    return path, prompts


def split_words(word):
    # Check if the word contains an underscore
    if '_' in word:
        # Split the compound word into individual words
        words = word.split('_')
        # Include the original compound word in the list
        words.append(word)
    else:
        # If it's a single word, just return it as a list
        words = [word]
    return words


def replace_words(text, word, new_word):
    # Extract words to replace from the input word
    words_to_replace = split_words(word)
    
    # Create a regular expression pattern to match the words
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(word) for word in words_to_replace) + r')\b', re.IGNORECASE)
    
    # Replace the matched words with the new word
    new_text = pattern.sub(new_word, text)
    
    return new_text