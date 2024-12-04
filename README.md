# Large-Multimodal-Model-HighResolution

[NVLM-D 论文地址](https://arxiv.org/abs/2409.11402)

NVLM-D最近看论文觉得效果还不错的样子，但官方没有训练代码，就自己看着写了一下训练的代码，使用trainer进行复现。

支持高分辨动态切图，能有效提高细颗粒度图片的识别，如OCR任务。

本次方案以Qwen为语言模型基座，视觉头模型为InternViT-6B-448px-V1-5


![模型主要架构图](https://github.com/X-jun-0130/NVLM-D_Code-Reproduction/blob/main/VisualModel/NVLM-D.png)



- **图片数据处理成token的逻辑**
```
IMG_START_TOKEN = '<|vision_start|>'
IMG_CONTEXT_TOKEN = '<|vision_pad|>'
IMG_END_TOKEN = '<|vision_end|>'

num_image_token = int((force_image_size // patch_size) ** 2 * (down_sample_ratio ** 2))

def convert_image_token(image):
    if dynamic_image_size:
        image = Image.open(image).convert('RGB')
        num_tile = len(dynamic_preprocess(image))
        tile_pos_identifiers = [f"<tile_{i}>" for i in range(1, num_tile)] + ["<tile_global_thumbnail>"]
        image_tokens = ''
        for tile_pos_identifier in tile_pos_identifiers:
            image_tokens += tile_pos_identifier + IMG_CONTEXT_TOKEN * num_image_token
        image_tokens = IMG_START_TOKEN + image_tokens + IMG_END_TOKEN
    else:
        image_tokens = IMG_CONTEXT_TOKEN * num_image_token
        image_tokens = IMG_START_TOKEN + image_tokens + IMG_END_TOKEN
    return image_tokens
```

### Lora训练

增加lora训练过程，VisualModel_SFT_Pretrain_Lora.py，这部分的代码可以进行第一阶段或者第二阶段的lora训练，能有效减少显存。

- 模型lora微调后如何进行线性层和lora权重提取单独保存，查看save_model.py
- 如何将所有权重进行合并，得到lora微调后的最终模型，查看merge_lora.py

