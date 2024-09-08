import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, StableDiffusionImageVariationPipeline
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
from PIL import Image
import requests
# from transformers import AutoProcessor, LlavaForConditionalGeneration

from sdxl_util import *
from knn_dino_mixeddata_base import *


# # Configuration
# CUDA_DEVICE = 1
# NAME_DICT_PATH = 
# FILTER1 = '/storage3/enbo/saved_data/sdxl_llava_i2i_allimage10percentprompt_60real'
# FILTER2 =  'saved_data/sdxl_llava_synfromreal_s8g2'
# TRAIN_DATA_DESTINATION_FOLDER = '/scratch/local/ssd/enbo/saved_data/synrealallimage10percentprompt_synsyns8g2'
# TEST_DATA_DESTINATION_FOLDER = 'saved_data/cifar_test100'


# def parse_args():
#     parser = argparse.ArgumentParser(description="Run CIFAR100 benchmark with different configurations")
#     parser.add_argument('--cuda_device', type=int, default=0, help='CUDA device index')
#     parser.add_argument('--image_name_dict_path', type=str, default='/scratch/local/ssd/enbo/saved_data/imagenet_label_name.json', help='dict for image label and name')
#     parser.add_argument('--filter1', type=str, default='/storage3/enbo/saved_data/sdxl_llava_i2i_allimage10percentprompt_60real', help='File 1 for filter')
#     parser.add_argument('--filter2', type=str, default='saved_data/sdxl_llava_synfromreal_s8g2', help='File 2 for filter')
#     parser.add_argument('--train_destination', type=str, default='/scratch/local/ssd/enbo/saved_data/synrealallimage10percentprompt_synsyns8g2', help='Train data destination')
#     parser.add_argument('--test_destination', type=str, default='saved_data/cifar_test100', help='Test data destination')
#     return parser.parse_args()



# order_sample = [[36, 0, 54, 5, 20], [22, 45, 13, 83, 19], [26, 73, 16, 62, 33], [34, 98, 24, 74, 53], [10, 94, 51, 4, 32], [38, 81, 50, 40, 41], [30, 89, 69, 64, 21], [84, 14, 88, 49, 68], [6, 80, 57, 65, 46], [9, 91, 48, 72, 31], [76, 7, 47, 8, 1], [61, 75, 63, 18, 86], [59, 70, 43, 85, 95], [27, 93, 35, 25, 82], [44, 56, 67, 66, 37], [60, 11, 2, 78, 52], [97, 39, 55, 3, 99], [29, 71, 23, 28, 90], [87, 15, 92, 17, 77], [12, 42, 96, 79, 58]]



# syn_classes = [5, 20, 83, 19, 62, 33, 74, 53, 4, 32, 40, 41, 64, 21, 49, 68, 65, 46, 72, 31, 8, 1, 18, 86, 85, 95, 25, 82, 66, 37, 78, 52, 3, 99, 28, 90, 17, 77, 79, 58]
# real_classes = list(set([i for i in range(100)]) - set(syn_classes))

# Image2Image
def sdxl_img2img_lessimageprompt_withuserincaption(class_dict, 
                                  image_size, 
                                  prompt_file_dict, 
                                  startimage_file_path,
                                  generator_seed, 
                                  num_image_replay=50,
                                  folder_name="/content/sd_images"):
    
    """
    original create_sdxl_data_fixed_prompts_randommultiple_img2img
    when there are less image and prompt then the number of images required
    """

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for id, class_name in class_dict.items():
        print("Generating images for class " + str(id) + ": " + class_name)
        
        path_file = os.path.join(startimage_file_path, f"class{id}.txt")
        class_file_path = os.path.join(folder_name, f"class{id}.txt")
        with open(path_file, 'r') as file:
            startimage_file_paths = [line.split()[0] for line in file.readlines()]
        existing_images_count = 0

        prompt_file_path = os.path.join(prompt_file_dict, f"class{id}.txt")
        current_prompt_list = extract_prompts_fromtxts(prompt_file_path)
        
        random.seed(42)
        indices1 = [random.randint(0, len(current_prompt_list)-1) for _ in range(num_image_replay)]
        current_prompt_list = [current_prompt_list[item] for item in indices1]

        random.seed(41)
        indices2 = [random.randint(0, len(startimage_file_paths)-1) for _ in range(num_image_replay)]
        current_init_image_list = [startimage_file_paths[item] for item in indices2]

        with open(class_file_path, "a") as file:
            for j, prompt in enumerate(current_prompt_list):
#                 print(prompt)
                
                starting_image_path = current_init_image_list[j]
                init_image = load_image(starting_image_path).resize((512, 512))

                _, ext = os.path.splitext(starting_image_path)
                ext_lower = ext.lower()

                image_name = class_name + str(j) + ext
                image_path = os.path.join(folder_name, image_name)
                new_image = pipe(prompt=prompt,image = init_image, 
                                strength = 0.8, 
                                num_inference_steps=20, 
                                generator = generator_seed, guidance_scale=2).images[0]
                resized_image = new_image.resize(image_size)

                resized_image.save(image_path, format = ext_lower.strip('.'))

                file.write(f"{image_path} {id}\n")

            print(f"Generated images for class {class_name}")


def sdxl_img2img_lessimageprompt(class_dict, 
                                  image_size, 
                                  prompt_file_dict, 
                                  startimage_file_path,
                                  generator_seed, 
                                  num_image_replay=50,
                                  folder_name="/content/sd_images"):
    
    """
    when there is no 'user' in the prompt txt file
    """

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for id, class_name in class_dict.items():
        print("Generating images for class " + str(id) + ": " + class_name)
        
        class_file_path = os.path.join(folder_name, f"class{id}.txt")
        existing_images_count = 0

        prompt_file_path = os.path.join(prompt_file_dict, f"class{id}.txt")
        startimage_file_paths, current_prompt_list = extract_prompts_fromtxts_real2syn(prompt_file_path)
        
        random.seed(42)
        indices1 = [random.randint(0, len(current_prompt_list)-1) for _ in range(num_image_replay)]
        current_prompt_list = [current_prompt_list[item] for item in indices1]

        random.seed(41)
        indices2 = [random.randint(0, len(startimage_file_paths)-1) for _ in range(num_image_replay)]
        current_init_image_list = [startimage_file_paths[item] for item in indices2]

        with open(class_file_path, "a") as file:
            for j, prompt in enumerate(current_prompt_list):
                
                starting_image_path = current_init_image_list[j]
                init_image = load_image(starting_image_path).resize((512, 512))
                
                _, ext = os.path.splitext(starting_image_path)
                ext_lower = ext.lower()

                image_name = class_name + str(j) + ext
                image_path = os.path.join(folder_name, image_name)
                new_image = pipe(prompt=prompt,image = init_image, 
                                strength = 0.8, 
                                num_inference_steps=20, 
                                generator = generator_seed, guidance_scale=2).images[0]
                resized_image = new_image.resize(image_size)

                resized_image.save(image_path, format = ext_lower.strip('.'))

                file.write(f"{image_path} {id}\n")

            print(f"Generated images for class {class_name}")





def sdxl_img2img_matching_image_prompt(class_dict, image_size, prompt_file_dict, startimage_file_path,
                                                          generator_seed, num_image_replay=50, folder_name="/content/sd_images"):
    
    """
    this is specified to matching pair of lavva prompt and images
    """
    # Create the folder if it doesn't exist
    random.seed(42)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    classes_to_process = list(class_dict.items())
    classes_processed = []

    while classes_to_process:
        id, class_name = classes_to_process.pop(0)
        print("Generating images for class " + str(id) + ": " + class_name)

        class_file_path = os.path.join(folder_name, f"class{id}.txt")

        prompt_file_path = os.path.join(prompt_file_dict, f"class{id}.txt")
        current_path_list, current_prompt_list = extract_prompts_fromtxts_real2syn(prompt_file_path)
        
        if len(current_prompt_list) < num_image_replay:
            print(f"Not enough prompts for class {id}. Skipping and will revisit later.")
            classes_to_process.append((id, class_name))
            continue

        # Generate the new images
        with open(class_file_path, "w") as file:
            for j, prompt in enumerate(current_prompt_list):
                # print(prompt)
                
                starting_image_path = current_path_list[j]
                image_name = os.path.basename(starting_image_path)

                init_image = load_image(starting_image_path).resize((512, 512))

            
                image_path = os.path.join(folder_name, image_name)
                new_image = pipe(prompt=prompt,image = init_image, 
                                strength = 0.8, 
                                num_inference_steps=20, 
                                generator = generator_seed, guidance_scale=2).images[0]
                resized_image = new_image.resize(image_size)

                resized_image.save(image_path)

                file.write(f"{image_path} {id}\n")

                print(f"Generated image {image_name} for class {class_name}")
        classes_processed.append((id, class_name))
        
        # Check if all classes are processed and the while loop should be terminated
        if len(classes_processed) == len(class_dict):
            break


def sdxl_img2img_moreimages_lessprompts(class_dict, image_size, prompt_file_dict, startimage_file_path,
                                                          generator_seed, num_image_replay=50, folder_name="/content/sd_images"):
    

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for id, class_name in class_dict.items():
        print("Generating images for class " + str(id) + ": " + class_name)
        
        path_file = os.path.join(startimage_file_path, f"class{id}.txt")
        class_file_path = os.path.join(folder_name, f"class{id}.txt")
        with open(path_file, 'r') as file:
            startimage_file_paths = [line.split()[0] for line in file.readlines()]
            print(len(startimage_file_paths))
        existing_images_count = 0

        prompt_file_path = os.path.join(prompt_file_dict, f"class{id}.txt")
        _, current_prompt_list = extract_prompts_fromtxts_real2syn(prompt_file_path)
        
        random.seed(42)
        indices1 = [random.randint(0, len(current_prompt_list)-1) for _ in range(num_image_replay)]
        current_prompt_list = [current_prompt_list[item] for item in indices1]

        current_init_image_list = startimage_file_paths

        with open(class_file_path, "a") as file:
            for j, prompt in enumerate(current_prompt_list):
                
                starting_image_path = current_init_image_list[j]
                init_image = load_image(starting_image_path).resize((512, 512))

                # image_name = starting_image_path.split('/')[-1]
                image_name = os.path.basename(starting_image_path)
                image_path = os.path.join(folder_name, image_name)
                new_image = pipe(prompt=prompt,image = init_image, 
                                strength = 0.8, 
                                num_inference_steps=20, 
                                generator = generator_seed, guidance_scale=2).images[0]
                resized_image = new_image.resize(image_size)

                resized_image.save(image_path)

                file.write(f"{image_path} {id}\n")

            print(f"Generated images for class {class_name}")
        

        ### text2img ###
def sdxl_text2img_from_all_matching_prompts(class_dict, 
                                               image_size, 
                                               prompt_file_dict, 
                                               integer_to_name,
                                                generator_seed, 
                                               num_image_replay=50,
                                               folder_name="/content/sd_images"):
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    classes_to_process = list(class_dict.items())
    classes_processed = []

    while classes_to_process:
        id, class_name = classes_to_process.pop(0)
        print("Generating images for class " + str(id) + ": " + class_name)
        class_file_path = os.path.join(folder_name, f"class{id}.txt")

        prompt_file_path = os.path.join(prompt_file_dict, f"class{id}.txt")
        current_path_list, current_prompt_list = extract_prompts_fromtxts_real2syn(prompt_file_path)
        
        if len(current_prompt_list) < num_image_replay:
            print(f"Not enough prompts for class {id}. Skipping and will revisit later.")
            classes_to_process.append((id, class_name))
            continue
            
        with open(class_file_path, "w") as file:
            for j, prompt in enumerate(current_prompt_list):

                image_name = os.path.basename(current_path_list[j])
                print(image_name)
                image_path = os.path.join(folder_name, image_name)
                new_image = pipe(prompt=prompt, 
                                num_inference_steps=16, 
                                generator = generator_seed, guidance_scale=1.0).images[0]
                resized_image = new_image.resize(image_size)

                resized_image.save(image_path)

                file.write(f"{image_path} {id}\n")

            print(f"Generated images for class {class_name}")
        classes_processed.append((id, class_name))
        
        # Check if all classes are processed and the while loop should be terminated
        if len(classes_processed) == len(class_dict):
            break

    print("All classes have been processed.")

def sdxl_img2img_matching_image_prompt(class_dict, image_size, prompt_file_dict, startimage_file_path,
                                                          generator_seed, num_image_replay=50, folder_name="/content/sd_images"):
    
    """
    this is specified to matching pair of lavva prompt and images
    """
    # Create the folder if it doesn't exist
    random.seed(42)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    classes_to_process = list(class_dict.items())
    classes_processed = []

    while classes_to_process:
        id, class_name = classes_to_process.pop(0)
        print("Generating images for class " + str(id) + ": " + class_name)

        class_file_path = os.path.join(folder_name, f"class{id}.txt")

        prompt_file_path = os.path.join(prompt_file_dict, f"class{id}.txt")
        current_path_list, current_prompt_list = extract_prompts_fromtxts_real2syn(prompt_file_path)
        
        start_index = 0
        if os.path.exists(class_file_path):
            with open(class_file_path, "r") as file:
                existing_lines = file.readlines()
                start_index = len(existing_lines)
            if start_index >= num_image_replay:
                print(f"Class {id} already has {start_index} images. Skipping.")
                continue
            else:
                print(f"Class {id} already has {start_index} images. Generating the remaining {num_image_replay - start_index} images.")

        # if len(current_prompt_list) < num_image_replay:
        #     print(f"Not enough prompts for class {id}. Skipping and will revisit later.")
        #     classes_to_process.append((id, class_name))
        #     continue

        # Generate the new images
        with open(class_file_path, "a") as file:
            for j in range(start_index, num_image_replay):
                prompt = current_prompt_list[j]
                
                starting_image_path = current_path_list[j]
                image_name = os.path.basename(starting_image_path)

                init_image = load_image(starting_image_path).resize((512, 512))

            
                image_path = os.path.join(folder_name, image_name)
                new_image = pipe(prompt=prompt,image = init_image, 
                                strength = 0.8, 
                                num_inference_steps=35, 
                                generator = generator_seed, guidance_scale=2).images[0]
                resized_image = new_image.resize(image_size)

                resized_image.save(image_path)

                file.write(f"{image_path} {id}\n")

                print(f"Generated image {image_name} for class {class_name}")
        classes_processed.append((id, class_name))
        
        # Check if all classes are processed and the while loop should be terminated
        if len(classes_processed) == len(class_dict):
            break

        #### real to synthetic ###
def sdxl_text2img_real2syn(class_dict, 
                           image_size, 
                           prompt_file_dict, 
                           integer_to_name, 
                           dict_syn_to_real_id,
                            generator_seed, 
                           num_image_replay=50, 
                           folder_name="/content/sd_images"):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for id, class_name in class_dict.items():
        print("Generating images for class " + str(id) + ": " + class_name)

        class_file_path = os.path.join(folder_name, f"class{id}.txt")
        existing_images_count = 0
        
        num_images_to_generate = num_image_replay
        
        real_id = dict_syn_to_real_id[id] # this is the real class id that bert thinks is most similar to the current syn class
        real_class_name = integer_to_name[real_id]
        prompt_file_path = os.path.join(prompt_file_dict, f"class{real_id}.txt")
        current_path_list, current_prompt_list = extract_prompts_fromtxts_real2syn(prompt_file_path)

        # Generate the new images
        with open(class_file_path, "w") as file:
            for j, prompt in enumerate(current_prompt_list):
                # print(prompt)
                syn_prompt = replace_words(prompt, real_class_name, class_name)
                # syn_image_name = class_name + f"{j}.png"
                
                real_image_name = os.path.basename(current_path_list[j])
                syn_image_name = real_image_name.replace(real_class_name, class_name)

                generated_image_path = os.path.join(folder_name, syn_image_name)
                new_image = pipe(prompt=syn_prompt, 
                                num_inference_steps=10, 
                                generator = generator_seed, guidance_scale=1.0).images[0]
                resized_image = new_image.resize(image_size)

                resized_image.save(generated_image_path)

                file.write(f"{generated_image_path} {id}\n")

                print(f"Generated image {syn_image_name} for class {class_name}")

                
                
def sdxl_img2img_matching_image_prompt_real2syn(pipe,
                                                class_dict, 
                                                image_size, 
                                                prompt_file_dict, 
                                                startimage_file_path, 
                                                integer_to_name, 
                                                dict_syn_to_real_id,
                                                generator_seed, 
                                                num_image_replay=50,
                                                folder_name="/content/sd_images"):
    
    """
    this is specified to matching pair of lavva prompt and images
    """

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for id, class_name in class_dict.items():
        # id, class_name are synthetic id and classname
        print("Generating images for class " + str(id) + ": " + class_name)

        class_file_path = os.path.join(folder_name, f"class{id}.txt")
        existing_images_count = 0
        
        num_images_to_generate = num_image_replay
        
        real_id = dict_syn_to_real_id[id] # this is the real class id that bert thinks is most similar to the current syn class
        real_class_name = integer_to_name[real_id]
        prompt_file_path = os.path.join(prompt_file_dict, f"class{real_id}.txt")
        current_path_list, current_prompt_list = extract_prompts_fromtxts_real2syn(prompt_file_path)

        # Generate the new images
        with open(class_file_path, "w") as file:
            for j, prompt in enumerate(current_prompt_list):
                # print(prompt)
                # syn_prompt = prompt.replace(real_class_name, class_name)
                syn_prompt = replace_words(prompt, real_class_name, class_name)
                # syn_image_name = class_name + f"{j}.png"
                
                real_image_name = os.path.basename(current_path_list[j])
                syn_image_name = real_image_name.replace(real_class_name, class_name)

                starting_image_path = current_path_list[j]
                init_image = load_image(starting_image_path).resize((512, 512))

                # generated image is synthetic name
                generated_image_path = os.path.join(folder_name, syn_image_name)
                # int(num_inference_steps * strength)
                new_image = pipe(prompt=syn_prompt,image = init_image, 
                                strength = 0.8, 
                                num_inference_steps=20, 
                                generator = generator_seed, guidance_scale=2).images[0]
                resized_image = new_image.resize(image_size)

                resized_image.save(generated_image_path)

                file.write(f"{generated_image_path} {id}\n")

                print(f"Generated image {syn_image_name} for class {class_name}")
                
 



# def sdxl_text2img_real2syn(class_dict
#                            image_size, 
#                            prompt_file_dict, 
#                            integer_to_name, 
#                            dict_syn_to_real_id,
#                             generator_seed, 
#                            num_image_replay=50, 
#                            folder_name="/content/sd_images"):
#     # Create the folder if it doesn't exist
#     if not os.path.exists(folder_name):
#         os.makedirs(folder_name)

#     for id, class_name in class_dict.items():
#         print("Generating images for class " + str(id) + ": " + class_name)

#         class_file_path = os.path.join(folder_name, f"class{id}.txt")
#         existing_images_count = 0
        
#         num_images_to_generate = num_image_replay
        
#         real_id = dict_syn_to_real_id[id] # this is the real class id that bert thinks is most similar to the current syn class
#         real_class_name = integer_to_name[real_id]
#         prompt_file_path = os.path.join(prompt_file_dict, f"class{real_id}.txt")
#         current_path_list, current_prompt_list = extract_prompts_fromtxts_real2syn(prompt_file_path)

#         # Generate the new images
#         with open(class_file_path, "w") as file:
#             for j, prompt in enumerate(current_prompt_list):
#                 # print(prompt)
#                 syn_prompt = prompt.replace(real_class_name, class_name)
#                 # syn_image_name = class_name + f"{j}.png"
                
#                 real_image_name = os.path.basename(current_path_list[j])
#                 syn_image_name = real_image_name.replace(real_class_name, class_name)

#                 generated_image_path = os.path.join(folder_name, syn_image_name)
#                 new_image = pipe(prompt=syn_prompt, 
#                                 num_inference_steps=10, 
#                                 generator = generator_seed, guidance_scale=1.0).images[0]
#                 resized_image = new_image.resize(image_size)

#                 resized_image.save(generated_image_path)

#                 file.write(f"{generated_image_path} {id}\n")

#                 print(f"Generated image {syn_image_name} for class {class_name}")

                
                
# name_dict = read_json('/scratch/local/ssd/enbo/saved_data/imagenet_label_name.json')
# name_list = [item for key, item in name_dict.items()]        
        
# real_dict, syn_dict = prepare_class_dictionaries(order_sample, name_list)

# generator1 = torch.Generator(device="cuda").manual_seed(42)

# create_sdxl_data_from_own_matching_prompts(real_integer_to_name, image_size = (224,224), 
#                                 prompt_file_dict = '/scratch/local/ssd/enbo/saved_data/llava_imagenet/imagenet_long_60real',
#                                   integer_to_name = integer_to_name,
#                                 generator_seed = generator1, num_image_replay=1300, 
#                                 folder_name='/storage3/enbo/saved_data/imageget_sdxl_llava_t2i_allprompt')


def sdxl_img2img_matching_image_prompt_synreal_synsyn(pipe,
                                                      synreal_class_dict,
                                                      synsyn_class_dict,
                                                      image_size, 
                                                      prompt_file_dict,
                                                      startimage_file_path,
                                                      integer_to_name,
                                                      dict_syn_to_real_id,
                                                      generator_seed,
                                                      num_image_replay=50,
                                                      synreal_folder_name="/content/sd_images",
                                                      synsyn_folder_name="/content/sd_images",
                                                      step_pre_div = 20
                                                     ):
    
    # Create the folder if it doesn't exist
    random.seed(42)
    if not os.path.exists(synreal_folder_name):
        os.makedirs(synreal_folder_name)
    if not os.path.exists(synsyn_folder_name):
        os.makedirs(synsyn_folder_name)
    
    synreal_classes_to_process = list(synreal_class_dict.items())
    synsyn_classes_to_process = list(synsyn_class_dict.items())
    
    all_classes_to_process = synreal_classes_to_process + synsyn_classes_to_process
    
    while all_classes_to_process:
        id, class_name = all_classes_to_process.pop(0)
        print("Generating images for class " + str(id) + ": " + class_name)
        
        ################synreal#####################
        if (id, class_name) in synreal_classes_to_process:
            
            class_file_path = os.path.join(synreal_folder_name, f"class{id}.txt")
            
            prompt_file_path = os.path.join(prompt_file_dict, f"class{id}.txt")
            current_path_list, current_prompt_list = extract_prompts_fromtxts_real2syn(prompt_file_path)
        
            start_index = 0
            if os.path.exists(class_file_path):
                with open(class_file_path, "r") as file:
                    existing_lines = file.readlines()
                    start_index = len(existing_lines)
                if start_index >= num_image_replay:
                    print(f"Class {id} already has {start_index} images. Skipping.")
                    continue
                else:
                    print(f"Class {id} already has {start_index} images. Generating the remaining {num_image_replay - start_index} images.")

            # Generate the new images
            with open(class_file_path, "a") as file_synreal:
                for j in range(start_index, num_image_replay):
                    prompt = current_prompt_list[j]

                    starting_image_path = current_path_list[j]
                    image_name = os.path.basename(starting_image_path)

                    init_image = load_image(starting_image_path).resize((512, 512))

                    image_path = os.path.join(synreal_folder_name, image_name)
                    new_image = pipe(prompt=prompt, image=init_image, 
                                     strength=0.8, 
                                     num_inference_steps=step_pre_div, 
                                     generator=generator_seed, guidance_scale=2).images[0]
                    resized_image = new_image.resize(image_size)

                    resized_image.save(image_path)

                    file_synreal.write(f"{image_path} {id}\n")

                    print(f"Generated image {image_name} for class {class_name}")


        ################synsyn#####################

        elif (id, class_name) in synsyn_classes_to_process:
            class_file_path = os.path.join(synsyn_folder_name, f"class{id}.txt")
            
            real_id = dict_syn_to_real_id[id]  # This is the real class ID that is most similar to the current syn class
            real_class_name = integer_to_name[real_id]
            prompt_file_path = os.path.join(prompt_file_dict, f"class{real_id}.txt")
            current_path_list, current_prompt_list = extract_prompts_fromtxts_real2syn(prompt_file_path)
            
            start_index = 0
            if os.path.exists(class_file_path):
                with open(class_file_path, "r") as file:
                    existing_lines = file.readlines()
                    start_index = len(existing_lines)
                if start_index >= num_image_replay:
                    print(f"Class {id} already has {start_index} images. Skipping.")
                    continue
                else:
                    print(f"Class {id} already has {start_index} images. Generating the remaining {num_image_replay - start_index} images.")

            # Generate the new images
            with open(class_file_path, "a") as file_synsyn:
                for j in range(start_index, num_image_replay):
                    prompt = current_prompt_list[j]
                    syn_prompt = replace_words(prompt, real_class_name, class_name)

                    real_image_name = os.path.basename(current_path_list[j])
                    syn_image_name = real_image_name.replace(real_class_name, class_name)

                    starting_image_path = current_path_list[j]
                    init_image = load_image(starting_image_path).resize((512, 512))

                    generated_image_path = os.path.join(synsyn_folder_name, syn_image_name)
                    new_image = pipe(prompt=syn_prompt, image=init_image, 
                                     strength=0.8, 
                                     num_inference_steps=step_pre_div, 
                                     generator=generator_seed, guidance_scale=2).images[0]
                    resized_image = new_image.resize(image_size)

                    resized_image.save(generated_image_path)

                    file_synsyn.write(f"{generated_image_path} {id}\n")

                    print(f"Generated image {syn_image_name} for class {class_name}")
                    
                    
                    
def sdxl_img2img_moreimage_lessprompt_real2syn(pipe,
                                                class_dict, 
                                                image_size, 
                                                prompt_file_dict, 
                                                startimage_file_path, 
                                                integer_to_name, 
                                                dict_syn_to_real_id,
                                                generator_seed, 
                                                num_image_replay=50,
                                                folder_name="/content/sd_images"):
    
    """
    this is specified to matching pair of lavva prompt and images
    """

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for id, class_name in class_dict.items():
        # id, class_name are synthetic id and classname
        print("Generating images for class " + str(id) + ": " + class_name)

        class_file_path = os.path.join(folder_name, f"class{id}.txt")
        existing_images_count = 0
        
        num_images_to_generate = num_image_replay
        
        real_id = dict_syn_to_real_id[id] # this is the real class id that bert thinks is most similar to the current syn class
        real_class_name = integer_to_name[real_id]
        prompt_file_path = os.path.join(prompt_file_dict, f"class{real_id}.txt")
        current_path_list, current_prompt_list = extract_prompts_fromtxts_real2syn(prompt_file_path)
        
        current_prompt_list = current_prompt_list[:int(num_images_to_generate*0.1)]
        print('num of prompts: ', len(current_prompt_list))
        random.seed(42)
        indices1 = [random.randint(0, len(current_prompt_list)-1) for _ in range(num_image_replay)]
        current_prompt_list = [current_prompt_list[item] for item in indices1]
        
        

        # Generate the new images
        with open(class_file_path, "w") as file:
            for j, prompt in enumerate(current_prompt_list):
                # print(prompt)
                # syn_prompt = prompt.replace(real_class_name, class_name)
                syn_prompt = replace_words(prompt, real_class_name, class_name)
                print(syn_prompt)
                # syn_image_name = class_name + f"{j}.png"
                
                real_image_name = os.path.basename(current_path_list[j])
                syn_image_name = real_image_name.replace(real_class_name, class_name)

                starting_image_path = current_path_list[j]
                init_image = load_image(starting_image_path).resize((512, 512))

                # generated image is synthetic name
                generated_image_path = os.path.join(folder_name, syn_image_name)
                # int(num_inference_steps * strength)
                new_image = pipe(prompt=syn_prompt,image = init_image, 
                                strength = 0.8, 
                                num_inference_steps=25, 
                                generator = generator_seed, guidance_scale=2).images[0]
                resized_image = new_image.resize(image_size)

                resized_image.save(generated_image_path)

                file.write(f"{generated_image_path} {id}\n")

                print(f"Generated image {syn_image_name} for class {class_name}")
                                    
                    
def sdxl_img2img_lessimage_lessprompt_real2syn(pipe,
                                                class_dict, 
                                                image_size, 
                                                prompt_file_dict, 
                                                startimage_file_path, 
                                                integer_to_name, 
                                                dict_syn_to_real_id,
                                                generator_seed, 
                                                num_image_replay=50,
                                                folder_name="/content/sd_images"):
    
    """
    this is specified to matching pair of lavva prompt and images
    """

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for id, class_name in class_dict.items():
        # id, class_name are synthetic id and classname
        print("Generating images for class " + str(id) + ": " + class_name)

        class_file_path = os.path.join(folder_name, f"class{id}.txt")
        existing_images_count = 0
        
        num_images_to_generate = num_image_replay
        
        real_id = dict_syn_to_real_id[id] # this is the real class id that bert thinks is most similar to the current syn class
        real_class_name = integer_to_name[real_id]
        prompt_file_path = os.path.join(prompt_file_dict, f"class{real_id}.txt")
        current_path_list, current_prompt_list = extract_prompts_fromtxts_real2syn(prompt_file_path)
        
        current_path_list = current_path_list[:int(num_images_to_generate*0.1)]
        current_prompt_list = current_prompt_list[:int(num_images_to_generate*0.1)]
        print('num of prompts: ', len(current_prompt_list))
        random.seed(42)
        indices1 = [random.randint(0, len(current_prompt_list)-1) for _ in range(num_image_replay)]
        current_prompt_list = [current_prompt_list[item] for item in indices1]
        
        random.seed(41)
        indices2 = [random.randint(0, len(startimage_file_paths)-1) for _ in range(num_image_replay)]
        current_path_list = [current_path_list[item] for item in indices2]
        
        

        # Generate the new images
        with open(class_file_path, "w") as file:
            for j, prompt in enumerate(current_prompt_list):
                # print(prompt)
                # syn_prompt = prompt.replace(real_class_name, class_name)
                syn_prompt = replace_words(prompt, real_class_name, class_name)
                print(syn_prompt)
                # syn_image_name = class_name + f"{j}.png"
                
                real_image_name = os.path.basename(current_path_list[j])
#                 syn_image_name = real_image_name.replace(real_class_name, class_name)
                extension = os.path.splitext(real_image_name)
                syn_image_name = class_name + str(j) + extension

                starting_image_path = current_path_list[j]
                init_image = load_image(starting_image_path).resize((512, 512))

                # generated image is synthetic name
                generated_image_path = os.path.join(folder_name, syn_image_name)
                # int(num_inference_steps * strength)
                new_image = pipe(prompt=syn_prompt,image = init_image, 
                                strength = 0.8, 
                                num_inference_steps=25, 
                                generator = generator_seed, guidance_scale=2).images[0]
                resized_image = new_image.resize(image_size)

                resized_image.save(generated_image_path)

                file.write(f"{generated_image_path} {id}\n")

                print(f"Generated image {syn_image_name} for class {class_name}")
                                    
                                        
def sdxl_img2img_lessimage_lessprompt_real2syn_lessgeneration(pipe,
                                                class_dict, 
                                                image_size, 
                                                prompt_file_dict, 
                                                startimage_file_path, 
                                                integer_to_name, 
                                                dict_syn_to_real_id,
                                                generator_seed, 
                                                num_image_replay=50,
                                                folder_name="/content/sd_images"):
    
    """
    this is specified to matching pair of lavva prompt and images
    """

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for id, class_name in class_dict.items():
        # id, class_name are synthetic id and classname
        print("Generating images for class " + str(id) + ": " + class_name)

        class_file_path = os.path.join(folder_name, f"class{id}.txt")
        existing_images_count = 0
        
        num_images_to_generate = num_image_replay
        
        real_id = dict_syn_to_real_id[id] # this is the real class id that bert thinks is most similar to the current syn class
        real_class_name = integer_to_name[real_id]
        prompt_file_path = os.path.join(prompt_file_dict, f"class{real_id}.txt")
        current_path_list, current_prompt_list = extract_prompts_fromtxts_real2syn(prompt_file_path)
        
        current_path_list = current_path_list[:int(num_images_to_generate)]
        current_prompt_list = current_prompt_list[:int(num_images_to_generate)]
        
        

        # Generate the new images
        with open(class_file_path, "w") as file:
            for j, prompt in enumerate(current_prompt_list):
                # print(prompt)
                # syn_prompt = prompt.replace(real_class_name, class_name)
                syn_prompt = replace_words(prompt, real_class_name, class_name)
                print(syn_prompt)
                # syn_image_name = class_name + f"{j}.png"
                
                real_image_name = os.path.basename(current_path_list[j])
                syn_image_name = real_image_name.replace(real_class_name, class_name)
#                 extension = os.path.splitext(real_image_name)
#                 syn_image_name = class_name + str(j) + extension

                starting_image_path = current_path_list[j]
                init_image = load_image(starting_image_path).resize((512, 512))

                # generated image is synthetic name
                generated_image_path = os.path.join(folder_name, syn_image_name)
                # int(num_inference_steps * strength)
                new_image = pipe(prompt=syn_prompt,image = init_image, 
                                strength = 0.8, 
                                num_inference_steps=25, 
                                generator = generator_seed, guidance_scale=2).images[0]
                resized_image = new_image.resize(image_size)

                resized_image.save(generated_image_path)

                file.write(f"{generated_image_path} {id}\n")

                print(f"Generated image {syn_image_name} for class {class_name}")
                                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    