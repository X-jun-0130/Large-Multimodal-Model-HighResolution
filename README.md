# Large-Multimodal-Model-HighResolution

[NVLM-D 论文地址](https://arxiv.org/abs/2409.11402)

NVLM-D最近看论文觉得效果还不错的样子，但官方没有训练代码，就自己看着写了一下训练的代码，使用trainer进行复现。

支持高分辨动态切图，能有效提高细颗粒度图片的识别，如OCR任务。

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
