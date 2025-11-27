# Copyright (c) 2023 OpenGVLab
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under MIT, with the full license text
# available at https://github.com/OpenGVLab/InternVL/blob/main/LICENSE.
#
# This modified file is released under the same license.

import argparse
import json
import os
import random
import torch
from vlm.utils import process_conversation
from PIL import Image
from tqdm import tqdm
from PIL import Image
import sys
from transformers import AutoModelForCausalLM
root = "root/AIA"
sys.path.append(root)
from models import MultiModalityCausalLM, VLChatProcessor

ds_collections = {
    'mmvet': {
        'root': 'data_root/und_eval/mm-vet/mm-vet/images',
        'question': 'data_root/und_eval/mm-vet/llava-mm-vet.jsonl',
        'metric': None,
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    }
}


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, root, data, prompt):
        self.root = root
        self.data = open(data).readlines()
        self.prompt = prompt
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = json.loads(self.data[idx].strip())
        image, question, question_id, annotation = data['image'], data[
            'text'], data['question_id'], data.get('answer', None)

        image = os.path.join(self.root, image)
        image = Image.open(image).convert('RGB')
        images = [image]
        
        question = question + ' ' + self.prompt

        images, conversation = process_conversation(images, question)

        return question_id, question, images, conversation, annotation


def evaluate_chat_model():
    random.seed(args.seed)
    prompt = ''

    for ds_name in args.datasets:
        dataset = VQADataset(
            root=ds_collections[ds_name]['root'],
            data=ds_collections[ds_name]['question'],
            prompt=prompt,
        )

        outputs = {}
        for _, (question_id, question, images, convers, annotations) in tqdm(enumerate(dataset)):
            images = images[0]
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{convers}",
                    "images": [f'img_path'],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

            # load images and prepare for inputs
            pil_images = [images]
            prepare_inputs = vl_chat_processor(
                conversations=conversation, images=pil_images, force_batchify=True
            ).to(vl_gpt.device)

            # # run image encoder to get the image embeddings
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

            results = vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
            )
            pred = tokenizer.decode(results[0].cpu().tolist(), skip_special_tokens=True)
            outputs[f'v1_{question_id}'] = pred

        print(f'Evaluating {ds_name} ...')
        results_file = os.path.join(args.out_dir, 'results.json')
        json.dump(outputs, open(results_file, 'w'))
        print('Results saved to {}'.format(results_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='mmvet')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model-path', type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    model_path = "deepseek-ai/Janus-Pro-7B"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    finetune_ckpt = torch.load(args.model_path)
    vl_gpt.load_state_dict(finetune_ckpt)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    total_params = sum(p.numel() for p in vl_gpt.parameters()) / 1e9
    print(f'[test] total_params: {total_params}B')

    evaluate_chat_model()
