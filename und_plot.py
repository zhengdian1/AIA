import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, required=True)
parser.add_argument('--prompt', type=str, required=True)
parser.add_argument('--image_path', type=str, required=True)
args = parser.parse_args()

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
    layers = range(1, 31)  # 28层，从1开始标记
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
    plt.savefig('und_mag1.png', dpi=300, bbox_inches='tight')
    plt.close()

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
        "role": "<|User|>",
        "content": f"<image_placeholder>\n{args.prompt}",
        "images": [args.image_path],
    },
    {"role": "<|Assistant|>", "content": ""},
]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation, images=pil_images, force_batchify=True
).to(vl_gpt.device)
start = torch.where(prepare_inputs['input_ids'][0]==torch.tensor(100016))[0].item() # 100016, 100593
end = torch.where(prepare_inputs['input_ids'][0]==torch.tensor(100593))[0].item() # 100016, 100593

# # run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# # run the model to get the response
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    output_attentions=True,
    return_dict_in_generate=True,
    do_sample=False,
    use_cache=True,
)
result = outputs[0][0]
img_end = inputs_embeds.shape[1]
topk = 3
full_attention = outputs.attentions
all_magnitudes = []
for token_idx in range(1, len(full_attention)):
    magnitudes = []
    for layer_idx in range(len(full_attention[0])):
        attn = full_attention[token_idx][layer_idx]
        text_attention = attn[0, :, 0, start:end]
        magnitude = text_attention.sum(dim=-1).mean().item()
        magnitudes.append(magnitude)
    magnitudes = torch.tensor(magnitudes)
    all_magnitudes.append(magnitudes)
    
plot_meanstep(all_magnitudes)

answer = tokenizer.decode(outputs[0][0].cpu().tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)
