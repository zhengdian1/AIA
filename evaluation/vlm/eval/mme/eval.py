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
import os
import re
import torch
from tqdm import tqdm
import sys
from transformers import AutoModelForCausalLM
root = "root/AIA"
sys.path.append(root)
from models import MultiModalityCausalLM, VLChatProcessor
from utils.io import load_pil_images

def post_processing(response):
    response = response.replace('\n', '').replace('不是', 'No').replace('是', 'Yes').replace('否', 'No')
    response = response.lower().replace('true', 'yes').replace('false', 'no')
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    response = re.sub(pattern, '', response)
    return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./Your_Results')
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--model-path', type=str, default=None)
    args = parser.parse_args()

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

    os.makedirs(args.out_dir, exist_ok=True)
    prompt = 'Answer the question using a single word or phrase.'
    for filename in os.listdir(args.root):
        fin = open(os.path.join(args.root, filename), 'r', encoding='utf-8')
        fout = open(os.path.join(args.out_dir, filename), 'w', encoding='utf-8')
        lines = fin.readlines()
        filename = filename.replace('.txt', '')
        for line in tqdm(lines):
            img, question, gt = line.strip().split('\t')
            question = question + ' ' + prompt
            img_path = os.path.join('data_root/und_eval/mme/MME_Benchmark_release_version', filename, img)
            if not os.path.exists(img_path):
                img_path = os.path.join('data_root/und_eval/mme/MME_Benchmark_release_version', filename, "images", img)
            if not os.path.exists(img_path):
                continue
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{question}",
                    "images": [f'{img_path}'],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

            # load images and prepare for inputs
            pil_images = load_pil_images(conversation)
            prepare_inputs = vl_chat_processor(
                conversations=conversation, images=pil_images, force_batchify=True
            ).to(vl_gpt.device)

            # # run image encoder to get the image embeddings
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

            outputs = vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
            )
            response = tokenizer.decode(outputs[0][0].cpu().tolist(), skip_special_tokens=True)

            print(img, question, gt, response, sep='\t', file=fout)
        fin.close()
        fout.close()

    os.system(f"python -m vlm.eval.mme.calculation --out-dir {args.out_dir}")
