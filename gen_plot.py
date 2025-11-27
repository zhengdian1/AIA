# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt
from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, required=True)
parser.add_argument('--prompt', type=str, required=True)
args = parser.parse_args()

# specify the path to the model
model_path = "deepseek-ai/Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path, attn_implementation="eager")
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, attn_implementation="eager"
)
finetune_ckpt = torch.load(args.ckpt_path)
vl_gpt.load_state_dict(finetune_ckpt)
vl_gpt.set_attn_implementation("eager")
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

conversation = [
    {
        "role": "User",
        "content": args.prompt,
    },
    {"role": "Assistant", "content": ""},
]

sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation,
    sft_format=vl_chat_processor.sft_format,
    system_prompt="",
)
prompt = sft_format + vl_chat_processor.image_start_tag

def plot_meanstep(data_list):
    data_matrix = []
    for i, tensor_data in enumerate(data_list):
        if isinstance(tensor_data, torch.Tensor):
            values = data_list[i].float().detach().cpu().numpy()
        else:
            values = data_list[i]
        data_matrix.append(values)
    data_matrix = np.array(data_matrix)  # 形状: (49, 28)
    mean_across_timesteps = np.mean(data_matrix, axis=0)  # 形状: (28,)
    fig, ax1 = plt.subplots(figsize=(10, 8))
    layers = range(1, 31)  # 28层,从1开始标记
    color1 = '#1f77b4'  # 蓝色
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Magnitude Value', color=color1)
    ax1.plot(layers, mean_across_timesteps, linewidth=2, color=color1, label='Average Value')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 1)  # 设置左侧y轴范围为0-1
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper right')
    plt.title('Average Values Across Timesteps for Each Layer')
    plt.tight_layout()
    plt.savefig('gen_mag1.png', dpi=300, bbox_inches='tight')
    plt.close()

@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 1,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
    input_length = inputs_embeds.shape[1]
    all_magnitudes = []
    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, output_attentions=True, return_dict_in_generate=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state
        if i > 0:
            attentions = outputs.attentions
            magnitudes = []
            for attention in attentions:
                attention = attention[0, :, 0, :]
                magnitude = attention[:,:input_length].sum(dim=-1).mean()
                magnitudes.append(magnitude)
            magnitudes = torch.tensor(magnitudes)
            all_magnitudes.append(magnitudes)
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)
    plot_meanstep(all_magnitudes)

generate(
    vl_gpt,
    vl_chat_processor,
    prompt,
)
