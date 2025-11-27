import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import linecache
from collections import defaultdict
import json
from torchvision import transforms
from models import VLChatProcessor
from torch.utils.data import Dataset, DataLoader
import datasets

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class TextToImageDataset(Dataset):
    def __init__(
        self,
        model_path,
        data_path,
    ):
        self.gen_transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        t2i_metadata = datasets.load_from_disk(data_path)
        t2i_metadata = t2i_metadata['train'] if isinstance(t2i_metadata, datasets.DatasetDict) else t2i_metadata
        t2i = t2i_metadata
        print(len(t2i))

        self.dataset = t2i

    def __getitem__(self, idx):
        curdata = self.dataset[idx]

        conversations = curdata['conversations']
        for conversation in conversations:
            if conversation['from']=='human':
                prompt = conversation['value']
                break
        
        image_path = curdata['image']

        image = Image.open(image_path).convert('RGB')
        image = self.gen_transform(image)
        conversation = [
            {
                "role": "<|User|>",
                "content":prompt,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + self.vl_chat_processor.image_start_tag
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').squeeze(0)
        if random.random() < 0.1:
            input_ids[1:-1] = self.vl_chat_processor.pad_id
            judge = 0
        else:
            judge = 1
        
        front=[0, len(input_ids)]
        end=[len(input_ids), len(input_ids)+576]

        return {"input_ids": input_ids, "image": image, "task_type": 0, "front": front, "end": end, "use_attn": judge}
    
    def __len__(self):
        return len(self.dataset)

class ImageToTextDataset(Dataset):
    def __init__(
        self,
        model_path,
        data_path,
    ):
        self.gen_transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        i2t_metadata = datasets.load_from_disk(data_path)
        i2t_metadata = i2t_metadata['train'] if isinstance(i2t_metadata, datasets.DatasetDict) else i2t_metadata

        print(len(i2t_metadata))
        self.dataset = i2t_metadata

    def __getitem__(self, idx):
        curdata = self.dataset[idx]

        texts = curdata['conversations']
        image_path = curdata['image']

        image = Image.open(image_path).convert('RGB')
        image = self.vl_chat_processor.image_processor([image])['pixel_values'].squeeze(0)
        all_input_ids,all_labels = [],[]
        end = []
        length = 0
        for id, text in enumerate(texts):
            if id%2==0:
                conversation = [
                    {
                        "role": "<|User|>",
                        "content": "",
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
                if id==0:
                    conversation[0]['content'] = "<image_placeholder>\n" + text['value']
                else:
                    conversation[0]['content'] = text['value']
            if id%2==1:
                sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                    conversations=conversation,
                    sft_format=self.vl_chat_processor.sft_format,
                    system_prompt="",
                )
                if id==1:
                    input_ids1 = self.tokenizer.encode(sft_format, return_tensors='pt').squeeze(0)
                    input_ids1 = torch.LongTensor(input_ids1)
                    image_token_mask: torch.BoolTensor = input_ids1 == self.vl_chat_processor.image_id
                    index = image_token_mask.nonzero()[0]
                    labels1 = torch.ones(len(input_ids1)+1+self.vl_chat_processor.num_image_tokens, dtype=input_ids1.dtype) * -100

                    all_input_ids.append(input_ids1[:index])
                    all_input_ids.append(self.vl_chat_processor.image_start_id * torch.ones((1), dtype=torch.long))
                    all_input_ids.append(self.vl_chat_processor.image_id * torch.ones((self.vl_chat_processor.num_image_tokens,), dtype=torch.long))
                    all_input_ids.append(self.vl_chat_processor.image_end_id * torch.ones((1), dtype=torch.long))
                    all_input_ids.append(input_ids1[index+1:])

                    all_labels.append(labels1)

                    front = [index, index + self.vl_chat_processor.num_image_tokens]
                    length = length + len(input_ids1) + 1 + self.vl_chat_processor.num_image_tokens
                else:
                    prompt = sft_format
                    input_ids1 = self.tokenizer.encode(prompt, return_tensors='pt').squeeze(0)
                    labels1 = torch.ones(input_ids1.shape, dtype=input_ids1.dtype) * -100
                    all_input_ids.append(input_ids1)
                    all_labels.append(labels1)
                    length = length + len(input_ids1) 
                answer = text['value']
                if id==len(texts)-1:
                    answer += self.tokenizer.eos_token
                temp_end = [length, -1]
                labels = self.tokenizer.encode(answer, add_special_tokens=False, return_tensors='pt').squeeze(0)
                length = length + len(labels) 
                temp_end[-1] = length

                end.append(temp_end)
                all_input_ids.append(labels)
                all_labels.append(labels)

        input_ids = torch.cat(all_input_ids,dim=0)
        labels = torch.cat(all_labels, dim=0)
        return {"input_ids":input_ids, "image":image, "labels": labels, "task_type":1, "front": front, "end": end, "use_attn": 1}

    def __len__(self):
        return len(self.dataset)

def my_collate_fn(batch):
    return batch

def TextToImageDataloader(cfg, tasks=[0,1]):
    probs = []
    dataloaders = []

    dataset1 = TextToImageDataset(
        model_path=cfg.model.processor_path,
        data_path=cfg.dataloader.gen_data_path,
    )
    sampler1 = torch.utils.data.distributed.DistributedSampler(dataset1)
    loader1 = DataLoader(
        dataset1,
        batch_size=cfg.dataloader.train.task1.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True,
        sampler=sampler1,
        prefetch_factor=cfg.dataloader.prefetch_factor,
        drop_last=True,
        collate_fn=my_collate_fn
    )
    dataloaders.append(loader1)
    probs.append(cfg.dataloader.train.task1.sample_ratio)

    dataset2 = ImageToTextDataset(
        model_path=cfg.dataloader.processor_path,
        data_path=cfg.dataloader.und_data_path,
    )
    sampler2 = torch.utils.data.distributed.DistributedSampler(dataset2)
    loader2 = DataLoader(
        dataset2,
        batch_size=cfg.dataloader.train.task2.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True,
        sampler=sampler2,
        prefetch_factor=cfg.dataloader.prefetch_factor,
        drop_last=True,
        collate_fn=my_collate_fn
    )
    dataloaders.append(loader2)
    probs.append(cfg.dataloader.train.task2.sample_ratio)

    probs = [p / sum(probs) for p in probs]
    if len(dataloaders)==1:
        return dataloaders[0], probs[0]

    return dataloaders, probs
