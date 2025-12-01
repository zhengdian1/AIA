![aia_logo](asset/log.jpg)

**How to Reach Us:**
- Code Issues: Please open an [issue](https://github.com/zhengdian1/AIA/issues) in our GitHub repository for any problems or bugs.
- General Inquiries: Contact Dian Zheng at zhengd35 [at] mail2 [at] sysu [at] edu [at] cn. 

[![AIA Report (Arxiv)](https://img.shields.io/badge/AIA%20Report-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2511.22663)
[![Project Page](https://img.shields.io/badge/AIA-Website-green?logo=googlechrome&logoColor=green)](https://zhengdian1.github.io/AIA-project/)
![Visitors](https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2Fzhengdian1%2FAIA&label=Visitors&icon=people&color=%23FFA500)

This repository contains the implementation of the following paper.

> **Architecture Decoupling Is Not All You Need For Unified Multimodal Model**<br>
> [Dian Zheng](https://zhengdian1.github.io/), [Manyuan Zhang](https://manyuan97.github.io), [Hongyu Li](https://scholar.google.com/citations?hl=zh-CN&user=PccL82sAAAAJ), [Kai Zou](https://github.com/Jacky-hate), [Hongbo Liu](https://github.com/Alexios-hub), [Ziyu Guo](https://ziyuguo99.github.io/), [Kaituo Feng](https://tulerfeng.github.io/),  [Yexin Liu](https://scholar.google.com/citations?user=Y8zBpcoAAAAJ&hl=zh-CN), [Ying Luo](https://scholar.google.com/citations?user=-VlvW5IAAAAJ&hl=en), [Yan Feng](https://scholar.google.com/citations?user=m4f3F4cAAAAJ&hl=en), [Peng Pei](https://openreview.net/profile?id=~Peng_Pei1),[Xunliang Cai](https://openreview.net/profile?id=~Xunliang_Cai1), [Hongsheng Li](https://www.ee.cuhk.edu.hk/~hsli/)<sup>+</sup><br>

## :fire: Updates
- [11/2025] :fire: **We released the training, inference, evaluation code, checkpoint of AIA!** :fire: 

## :mega: Overview
![overall](asset/motivation.jpg)
Overview of AIA. While the community has primarily focused on data quality, data mixture ratios, and architectural decoupling strategies for unified multimodal models, we are the first to analyze the underlying mechanisms and explore the way to narrow the gap between purly architecture and BAGEL like ones. We discover that architectural decoupling does not fundamentally resolve the conflicts between generation and understanding tasks, but rather drives the multimodal interaction patterns closer to those of single-task models. Based on this insight, we propose Attention Interaction Alignment (AIA), a method that explicitly constrains interaction patterns during training without requiring architectural decoupling. Our approach achieves performance improvements on both Emu3 and Janus-Pro, demonstrating its effectiveness in alleviating task conflicts.


## :hammer: Supervised Fine-Tuning

**0. Environment Preparation**

First please clone our repo and prepare the python environment. We recommend using Python>=3.10. 
```bash
git clone https://github.com/zhengdian1/AIA.git
cd AIA

conda create -n janus-pro-aia python=3.11
conda activate janus-pro-aia
pip install -r requirements.txt
```

**1. Training Configuration**

Before starting the training, you need to prepare a configuration file in advance. We provide an example for reference: `configs/t2i_generation.yml`. This YAML configuration file defines the training settings for SFT. It includes sections for general training setup, optimization strategies, model paths, and data loading. 

To run the training code, you need to specify the following parameters:

- `output_path`: Path to save model checkpoints and outputs.
- `pre_path`: Path to training resume.
- `log_path`: Path to store training logs.
- `model_path`: Path to the pretrained model.
- `processor_path`: Path to the processor.
- `und_data_path`: Path to understanding training data.
- `gen_data_path`: Path to generation training data.

**2. Prepare Training Data**

We provide an example data sample to clarify the required format for training data.

Specifically, for text-to-image, each data sample should follow the format below:

```json
{
  "conversations": [{"from": "human", "value": "a photo of a cat"}, {"from": "gpt", "value": "<image>"}], 
  "image": "path"
}
```

For image understanding, each data sample should follow the format below:

```json
{
  "image": "path", 
  "conversations": [{"from": "human", "value": "Is there a cat in the image? Please answer yes or no."}, {"from": "gpt", "value": "yes"}]
}
```

**3. Training**

Next, we provide two types of training scripts, you can choose the one suitable for your situation.

If you train on a single node, use the scropt below:

```bash
python launch.py --args_yml_fn configs/t2i_generation.yml
```

If you train on multi nodes, use the scropt below:

```bash
bash run.sh
```

## :surfer: Inference && Evaluation

**Inference**

First, downloading our [checkpoint (coming soon)](https://github.com/zhengdian1/AIA/tree/main)

Then, we provide inference codes for result or our proposed cross-modal interaction pattern plot.

If you want to output the result, use the scropt below:

```bash
python generation_inference.py --ckpt_path your_path --prompt 'A cute cat.'

python interactivechat.py --ckpt_path your_path --prompt "Describe the image in detail" --image_path /path/to/image.jpg
```

If you want to see the cross-modal interaction pattern plot, use the scropt below:

```bash
python gen_plot.py --ckpt_path your_path --prompt 'A cute cat.'

python und_plot.py --ckpt_path your_path --prompt "Describe the image in detail" --image_path /path/to/image.jpg
```

**Evaluation**

We provide the evaluation code on widely used visual understanding and generation benchmarks below.

For visual understanding evaluation (MMMU, MME, MMVP, MMBench, POPE, MMVet), please firstly downloading the corresponding dataset from [data](https://internvl.readthedocs.io/en/latest/get_started/eval_data_preparation.html) and then following the scripts below:
```bash
cd evaluation
bash scripts/eval/evaluate.sh
```
Note that for MMBench and MMVet, you need to submit the result to the official website: [MMBench](https://mmbench.opencompass.org.cn/mmbench-submission), [MMVet](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator)

For visual generation evaluation (DPG, GenEval), please refer to [UlmEvalKit](https://github.com/ULMEvalKit/ULMEvalKit), which will be more efficient. 
Additionaly, we use the long text prompt for GenEval from [BAGEL](https://github.com/bytedance-seed/BAGEL)

<a name="citation_and_acknowledgement"></a>
## :black_nib: Citation

   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
    @article{zheng2025architecture,
      title={Architecture Decoupling Is Not All You Need For Unified Multimodal Model},
      author={Zheng, Dian and Zhang, Manyuan and Li, Hongyu and Zou, Kai and Liu, Hongbo and Guo, Ziyu and Feng, Kaituo and Liu, Yexin and Luo, Ying and Feng, Yan and Pei, Peng and Cai, Xunliang and Li, Hongsheng},
      journal={arXiv preprint arXiv:2503.21755},
      year={2025}
    }
   ```

## :hearts: Acknowledgement

#### :hugs: Open-Sourced Repositories
This project wouldn't be possible without the following open-sourced repositories:
[Janus-Pro](https://github.com/deepseek-ai/Janus), [Janus-Pro-R1](https://github.com/wendell0218/Janus-Pro-R1), [BAGEL](https://github.com/bytedance-seed/BAGEL), [Emu3](https://github.com/baaivision/Emu3).

## Related Links

<!-- We are putting together [Awesome-Evaluation-of-Visual-Generation](https://github.com/ziqihuangg/Awesome-Evaluation-of-Visual-Generation), which collects works for evaluating visual generation. -->

Our related projects: [Uni-MMMU](https://vchitect.github.io/Uni-MMMU-Project/), [VBench-2.0](https://github.com/Vchitect/VBench/tree/master/VBench-2.0)

```bibtex
@article{zou2025uni,
    title={Uni-MMMU: A Massive Multi-discipline Multimodal Unified Benchmark},
    author={Zou, Kai and Huang, Ziqi and Dong, Yuhao and Tian, Shulin and Zheng, Dian and Liu, Hongbo and He, Jingwen and Liu, Bin and Qiao, Yu and Liu, Ziwei},
    journal={arXiv preprint arXiv:2510.13759},
    year={2025}
}

@article{zheng2025vbench2,
    title={{VBench-2.0}: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness},
    author={Zheng, Dian and Huang, Ziqi and Liu, Hongbo and Zou, Kai and He, Yinan and Zhang, Fan and Zhang, Yuanhan and He, Jingwen and Zheng, Wei-Shi and Qiao, Yu and Liu, Ziwei},
    journal={arXiv preprint arXiv:2503.21755},
    year={2025}
}
```