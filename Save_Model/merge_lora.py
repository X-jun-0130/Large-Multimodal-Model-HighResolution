import json
import os
import json
import random
from transformers import AutoModel
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


original_model_path = '/VL-Pretrain/WiNGPT-VL-32B-10000/'
lora_pth = '/Train_Result/VLM-SFT-lora/'
mlp_pth = '/Train_Result/VLM-SFT-lora/non_lora_trainables.bin'
output_dir = '/VL-Finetune/sft-100steps'


# #加载预训练后的模型
original_model = AutoModel.from_pretrained(original_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)

from peft import PeftModel
print('Loading LoRA weights...')
model = PeftModel.from_pretrained(original_model.language_model, lora_pth)
print('Merging LoRA weights...')
model = model.merge_and_unload()
print("success")

#将model的权重替换original_model.language_model的权重
merged_state_dict = model.state_dict()
original_model.language_model.load_state_dict(merged_state_dict)

## 加载线性层权重
state_dict = torch.load(mlp_pth, map_location='cpu')
original_model.mlp1.load_state_dict(state_dict)

#保存合并后的模型
torch.save(original_model.state_dict(), os.path.join(output_dir,'pytorch_model.bin'))



