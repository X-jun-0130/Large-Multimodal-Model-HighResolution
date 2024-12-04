from transformers import AutoModel,AutoTokenizer
import torch
import os
from data.image_dynamic_preprocess import load_image, build_transform
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

model_name = "/VL-Finetune/WiNGPT-VL-32B-100steps/"
model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
generation_config = dict(max_new_tokens=1024, do_sample=False)

question = '<image>\n介绍这幅图'

file = "./4047013cec75f683148babff0554d413.jpeg"
'''
no-dynamic
'''
# img = Image.open(file).convert('RGB')
# transform = build_transform(448)
# pixel_values = [transform(img)]
# pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()

'''
with-dynamic
'''
pixel_values = load_image(file, max_num=6).to(torch.bfloat16).cuda()

response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')



