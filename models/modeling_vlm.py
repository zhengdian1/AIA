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
from attrdict import AttrDict
from einops import rearrange
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedModel,
)
from transformers.configuration_utils import PretrainedConfig
import torch.nn as nn
from .clip_encoder import CLIPVisionTower
from .projector import MlpProjector

class vision_head(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.output_mlp_projector = torch.nn.Linear(
            params.n_embed, params.image_token_embed
        )
        self.vision_activation = torch.nn.GELU()
        self.vision_head = torch.nn.Linear(
            params.image_token_embed, params.image_token_size
        )

    def forward(self, x):
        x = self.output_mlp_projector(x)
        x = self.vision_activation(x)
        x = self.vision_head(x)
        return x


def model_name_to_cls(cls_name):
    if "MlpProjector" in cls_name:
        cls = MlpProjector

    elif "CLIPVisionTower" in cls_name:
        cls = CLIPVisionTower

    elif "VQ" in cls_name:
        from .vq_model import VQ_models

        cls = VQ_models[cls_name]
    elif "vision_head" in cls_name:
        cls = vision_head
    else:
        raise ValueError(f"class_name {cls_name} is invalid.")

    return cls


class VisionConfig(PretrainedConfig):
    model_type = "vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class AlignerConfig(PretrainedConfig):
    model_type = "aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenVisionConfig(PretrainedConfig):
    model_type = "gen_vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenAlignerConfig(PretrainedConfig):
    model_type = "gen_aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenHeadConfig(PretrainedConfig):
    model_type = "gen_head"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class MultiModalityConfig(PretrainedConfig):
    model_type = "multi_modality"
    vision_config: VisionConfig
    aligner_config: AlignerConfig

    gen_vision_config: GenVisionConfig
    gen_aligner_config: GenAlignerConfig
    gen_head_config: GenHeadConfig

    language_config: LlamaConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vision_config = kwargs.get("vision_config", {})
        self.vision_config = VisionConfig(**vision_config)

        aligner_config = kwargs.get("aligner_config", {})
        self.aligner_config = AlignerConfig(**aligner_config)

        gen_vision_config = kwargs.get("gen_vision_config", {})
        self.gen_vision_config = GenVisionConfig(**gen_vision_config)

        gen_aligner_config = kwargs.get("gen_aligner_config", {})
        self.gen_aligner_config = GenAlignerConfig(**gen_aligner_config)

        gen_head_config = kwargs.get("gen_head_config", {})
        self.gen_head_config = GenHeadConfig(**gen_head_config)

        language_config = kwargs.get("language_config", {})
        if isinstance(language_config, LlamaConfig):
            self.language_config = language_config
        else:
            self.language_config = LlamaConfig(**language_config)


class MultiModalityPreTrainedModel(PreTrainedModel):
    config_class = MultiModalityConfig
    base_model_prefix = "multi_modality"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"


class MultiModalityCausalLM(MultiModalityPreTrainedModel):
    def __init__(self, config: MultiModalityConfig):
        super().__init__(config)

        vision_config = config.vision_config
        vision_cls = model_name_to_cls(vision_config.cls)
        self.vision_model = vision_cls(**vision_config.params)

        aligner_config = config.aligner_config
        aligner_cls = model_name_to_cls(aligner_config.cls)
        self.aligner = aligner_cls(aligner_config.params)

        gen_vision_config = config.gen_vision_config
        gen_vision_cls = model_name_to_cls(gen_vision_config.cls)
        self.gen_vision_model = gen_vision_cls()

        gen_aligner_config = config.gen_aligner_config
        gen_aligner_cls = model_name_to_cls(gen_aligner_config.cls)
        self.gen_aligner = gen_aligner_cls(gen_aligner_config.params)

        gen_head_config = config.gen_head_config
        gen_head_cls = model_name_to_cls(gen_head_config.cls)
        self.gen_head = gen_head_cls(gen_head_config.params)

        self.gen_embed = torch.nn.Embedding(
            gen_vision_config.params.image_token_size, gen_vision_config.params.n_embed
        )
        self.loss_fct = torch.nn.CrossEntropyLoss()

        language_config = config.language_config
        language_config._attn_implementation = 'eager'
        # language_config._attn_implementation = 'flash_attention_2'
        self.language_model = LlamaForCausalLM(language_config)
       

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        images_seq_mask: torch.LongTensor,
        images_emb_mask: torch.LongTensor,
        **kwargs,
    ):
        """

        Args:
            input_ids (torch.LongTensor): [b, T]
            pixel_values (torch.FloatTensor):   [b, n_images, 3, h, w]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_emb_mask (torch.BoolTensor): [b, n_images, n_image_tokens]

            assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """

        bs, n = pixel_values.shape[0:2]
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        # [b x n, T2, D]
        images_embeds = self.aligner(self.vision_model(images))

        # [b x n, T2, D] -> [b, n x T2, D]
        images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
        # [b, n, T2] -> [b, n x T2]
        images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")

        # [b, T, D]
        input_ids[input_ids < 0] = 0  # ignore the image embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # replace with the image embeddings
        inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]

        return inputs_embeds

    def forward(self, input_ids, attention_mask, labels=None, image1=None, image_seq_mask=None, image2=None, task_type=0, front=None, end=None, use_attn=False):
        if task_type == 0:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            image_embeds, labels = self.prepare_embedding(image1)
            input_embeds = torch.cat((input_embeds, image_embeds), dim=1)
            B, L = image_embeds.shape[0], image_embeds.shape[1]

            attention_mask = torch.cat((attention_mask, torch.ones((B, L)).long().to(attention_mask.device)), dim=1)
            label_len = labels.shape[-1]
            outputs = self.language_model.model(inputs_embeds=input_embeds, 
                                                    attention_mask=attention_mask, output_attentions=True, return_dict_in_generate=True)
            last_hidden_state = outputs.last_hidden_state
            full_attention = outputs.attentions

            image_logits = self.gen_head(last_hidden_state)
            visual_vocab_size = image_logits.shape[-1]
            shift_logits = image_logits[..., -1-label_len:-1, :].contiguous()
            loss_ntp = self.loss_fct(shift_logits.view(-1, visual_vocab_size), labels.view(-1))

            full_attn_loss = 0
            for layer_idx in range(len(full_attention)):
                batch_attn = full_attention[layer_idx]
                for batch_idx in range(len(batch_attn)):
                    attn = batch_attn[batch_idx]
                    s_use_attn = use_attn[batch_idx]
                    s_front = front[batch_idx]
                    s_end = end[batch_idx]

                    used = attn[:,s_end[0]:s_end[1], s_front[0]:s_front[1]]
                    attn_loss_ori = used.float().sum(dim=-1).mean(dim=1).mean(dim=-1)
                    if layer_idx<10:
                        criterion = nn.HuberLoss(delta=0.2)
                        boundary = 0.4
                    elif layer_idx>=10 and layer_idx<25:
                        criterion = nn.HuberLoss(delta=0.1)
                        boundary = 0.4
                    else:
                        criterion = nn.HuberLoss(delta=0.05)
                        boundary = 0.2
                    target = torch.full_like(attn_loss_ori, boundary)
                    attn_loss = criterion(attn_loss_ori, target)
                    if s_use_attn:
                        full_attn_loss = full_attn_loss + attn_loss
                    else:
                        full_attn_loss = full_attn_loss + attn_loss * 0
            loss_attn = full_attn_loss / (len(full_attention) * len(full_attention[2]))
            loss = loss_ntp + 5 * loss_attn

        elif task_type==1:
            image_embeds, _ = self.prepare_embedding(image1, gen_image=False)
            input_ids[input_ids < 0] = 0
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            for i in range(input_embeds.shape[0]):
                input_embeds[i][image_seq_mask[i]] = image_embeds[i]
            label_len = labels.shape[-1]
            outputs = self.language_model(inputs_embeds=input_embeds, 
                                              attention_mask=attention_mask, output_attentions=True, return_dict_in_generate=True)
            text_logits = outputs.logits
            full_attention = outputs.attentions

            text_vocab_size = text_logits.shape[-1]
            shift_logits = text_logits[..., -label_len:-1, :].contiguous()
            shift_labels = labels[...,1:].contiguous()
            loss_ntp = self.loss_fct(shift_logits.view(-1,text_vocab_size), shift_labels.view(-1))

            criterion = nn.HuberLoss(delta=0.05)
            full_attn_loss = 0
            for layer_idx in range(len(full_attention)):
                batch_attn = full_attention[layer_idx]
                for batch_idx in range(len(batch_attn)):
                    attn = batch_attn[batch_idx]
                    s_use_attn = use_attn[batch_idx]
                    s_front = front[batch_idx]
                    s_end = end[batch_idx]

                    us_list = []
                    if s_use_attn:
                        for en in s_end:
                            us_list.append(attn[:,en[0]:en[1], s_front[0]:s_front[1]])
                        used = torch.cat((us_list),dim=1)
                        attn_loss_ori = used.float().sum(dim=-1).mean(dim=1).mean(dim=-1)
                        if layer_idx<10:
                            boundary = 0.1
                        elif layer_idx>=10 and layer_idx<=20:
                            boundary= 0.15
                        elif layer_idx>20 and layer_idx<=29:
                            boundary = 0.3
                        else:
                            boundary=0.2
                        target = torch.full_like(attn_loss_ori, boundary)
                        attn_loss = criterion(attn_loss_ori, target)
                        full_attn_loss = full_attn_loss + attn_loss
                    else:
                        used = attn[:,:, s_front[0]:s_front[1]]
                        attn_loss = used.float().sum(dim=-1).mean(dim=1).mean(dim=-1)
                        full_attn_loss = full_attn_loss + attn_loss * 0
            loss_attn = full_attn_loss / (len(full_attention) * len(full_attention[2]))
            loss = loss_ntp + 10 * loss_attn
        else:
            raise NotImplementedError
            
        return loss, loss_attn

    def prepare_embedding(
        self,
        pixel_values: torch.FloatTensor,
        gen_image=True,
        **kwargs,
    ):
        if gen_image:
            _, _, all_image_ids = self.gen_vision_model.encode(pixel_values)
            image_ids = all_image_ids[2]
            images_embeds = self.gen_aligner(self.gen_embed(image_ids))
        else:
            image_ids = None
            images_embeds = self.aligner(self.vision_model(pixel_values))

        return images_embeds, image_ids

    def prepare_gen_img_embeds(self, image_ids: torch.LongTensor):
        return self.gen_aligner(self.gen_embed(image_ids))


AutoConfig.register("vision", VisionConfig)
AutoConfig.register("aligner", AlignerConfig)
AutoConfig.register("gen_vision", GenVisionConfig)
AutoConfig.register("gen_aligner", GenAlignerConfig)
AutoConfig.register("gen_head", GenHeadConfig)
AutoConfig.register("multi_modality", MultiModalityConfig)
AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)
