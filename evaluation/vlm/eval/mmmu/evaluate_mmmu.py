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
import itertools
import json
import os
import sys
import random
import torch
import numpy as np
from .data_utils import CAT_SHORT2LONG, process_single_sample
from datasets import concatenate_datasets, load_dataset
from vlm.utils import process_conversation
from tqdm import tqdm
from transformers import AutoModelForCausalLM
root = "root/AIA"
sys.path.append(root)
from models import MultiModalityCausalLM, VLChatProcessor


ds_collections = {
    'MMMU_validation': {
        'root': 'data_root/und_eval/MMMU/MMMU',
        'max_new_tokens': 10,
        'min_new_tokens': 1,
        'split': 'validation'
    },
    'MMMU_test': {
        'root': 'data_root/und_eval/MMMU/MMMU',
        'max_new_tokens': 10,
        'min_new_tokens': 1,
        'split': 'test'
    },
    'MMMU_dev': {
        'root': 'data_root/und_eval/MMMU/MMMU',
        'max_new_tokens': 10,
        'min_new_tokens': 1,
        'split': 'dev'
    },
}

def get_top_text_token_count(attn_weights, text_end_pos, topk):
    attn_weights = attn_weights[0, :, 0, :]  # [num_heads, total_tokens]
    top_indices = torch.topk(attn_weights, k=topk, dim=-1).indices  # [num_heads, topk]
    text_token_counts = []
    for head_idx in range(top_indices.shape[0]):
        text_count = (top_indices[head_idx] < text_end_pos).sum().item()
        text_token_counts.append(text_count)
    return sum(text_token_counts) / len(text_token_counts)  # 返回平均值

def collate_fn(batches):
    questions = [_['question'] for _ in batches]
    images = [_['images'] for _ in batches]
    conversation = [_['conversation'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    data_ids = [_['data_id'] for _ in batches]
    options = [_['option'] for _ in batches]
    return questions, images, conversation, answers, data_ids, options

class MMMUDataset(torch.utils.data.Dataset):

    def __init__(self, root, split, prompt):
        # run for each subject
        sub_dataset_list = []
        for subject in tqdm(CAT_SHORT2LONG.values()):
            sub_dataset = load_dataset(root, subject, split=split, cache_dir=os.path.join(os.getcwd(), 'eval/vlm/data/MMMU/'))
            sub_dataset_list.append(sub_dataset)

        # merge all dataset
        self.data = concatenate_datasets(sub_dataset_list)
        self.prompt = prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = process_single_sample(self.data[idx])
        data_id = data['id']
        question = data['question'].strip()
        pil_images = data['image']
        question_type = data['question_type']

        choices = eval(data['options'])
        answer = data['answer'] if 'answer' in data else None

        choice_list = []
        options = {}
        multiple_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
        for i, c in enumerate(choices):
            choice_list.append('{}. {}'.format(multiple_choices[i], c.strip()))
            options[multiple_choices[i]] = c.strip()
        choice_txt = '\n'.join(choice_list)
        images = []
        for idx, pil_image in enumerate(pil_images):
            if pil_image is not None:
                # if idx == 0:
                #     pil_image = pil_image.resize((pil_image.width * 2, pil_image.height * 2), Image.BILINEAR)
                images.append(pil_image)

        if len(choice_txt) > 0:
            question += '\n' + choice_txt
        question += '\n' + self.prompt[question_type]
        question = question.strip()

        # NOTE: Do not add <image> since <image 1> has been added
        # question = "<image>" * len(images) + "\n" + question

        images, conversation = process_conversation(images, question)

        return {
            'question': question,
            'images': images,
            'conversation': conversation,
            'answer': answer,
            'option': options,
            'data_id': data_id
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def post_process(pred, option):
    pred = pred.strip()
    option_candidate = list(option.keys())
    if len(option_candidate)==0:
        return pred
    if len(pred) == 1:
        return pred
    elif len(pred) == 0:
        pred = option_candidate[1]
    elif len(pred) != 1 and pred[0] in option_candidate:
        return pred[0]
    elif len(pred) != 1 and pred[0] not in option_candidate:
        for k, v in option.items():
            if v in pred:
                return k
        return option_candidate[0]

    return pred

def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_chat_model():
    prompt = {
        'multiple-choice': "Answer with the option's letter from the given choices directly.",
        'open': 'Answer the question using a single word or phrase.'
    }
    set_seed(args.seed)

    for ds_name in args.datasets:
        dataset = MMMUDataset(
            root=ds_collections[ds_name]['root'],
            split=ds_collections[ds_name]['split'],
            prompt=prompt,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        outputs = []
        for _, (questions, images, conversation, answers, data_ids, options) in tqdm(enumerate(dataloader)):
            images = images[0][0].convert('RGB')
            qs = conversation[0]

            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{qs}",
                    "images": [f'temp.png'],
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
            pred = tokenizer.decode(results[0].cpu().tolist(), skip_special_tokens=True).strip()
            # print(pred)
            pred = post_process(pred, options[0])
            preds = [pred]

            for question, pred, answer, data_id in zip(questions, preds, answers, data_ids):
                outputs.append({
                    'question': question,
                    'answer': pred,
                    'gt_answers': answer,
                    'data_id': data_id
                })

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            output_path = os.path.join(args.out_dir, "prediction.json")
            outputs = {}
            for item in merged_outputs:
                outputs[item['data_id']] = item['answer']
            with open(output_path, 'w') as f:
                json.dump(outputs, f, indent=4)
            print('Results saved to {}'.format(output_path))
            if ds_collections[ds_name]['split'] == 'validation':
                print('Evaluating ...')
                cmd = f'python -m vlm.eval.mmmu.main_eval_only ' \
                      f'--output_path {output_path} ' \
                      f'--answer_path vlm/eval/mmmu/answer_dict_val.json ' \
                      f'--out-dir {args.out_dir}'
                print(cmd)
                os.system(cmd)
            output_path = os.path.join(args.out_dir, "results.jsonl")
            writer = open(output_path, 'w')
            for item in merged_outputs:
                writer.write(json.dumps(item) + '\n')
            writer.close()
            print('Results saved to {}'.format(output_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='MMMU_validation')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model-path', type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

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
