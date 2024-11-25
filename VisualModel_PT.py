
# deepspeed --master_addr 172.xxxx.94 --master_port 5050 --include localhost:0,1,2,3,4,5,6,7  /VisualModel_PT.py
import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer,  AutoConfig
from transformers import AutoModelForCausalLM
#from liger_kernel.transformers import AutoLigerKernelForCausalLM as AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence
from VisualModel.model_winvlm import WiNVLM_Model
from VisualModel.configuration_intern_vit import InternVisionConfig
from VisualModel.modeling_intern_vit import InternVisionModel
from VisualModel.configuration_winvlm import WiNVLM_Config
from image_dynamic_preprocess import dynamic_preprocess, build_transform
from PIL import Image


import warnings
warnings.filterwarnings("ignore")
torch.manual_seed(42)

IMG_START_TOKEN = '<|vision_start|>'
IMG_CONTEXT_TOKEN = '<|vision_pad|>'
IMG_END_TOKEN = '<|vision_end|>'

model_args = {
            "lm_model_name_or_path":"/Qwen2.5-3B-Instruct/",
            "vision_path":"/InternViT-6B-448px-V1-5/",
            "use_cache":False,
            "use_flash_attention_2":True,
            "drop_path_rate":0.0,
            "vision_select_layer":-1,
            "ps_version":'v2',
            "freeze_vm":True,
            "freeze_llm":True,
            "grad_checkpoint":True,
            'train_type':'PT'
            }


data_args ={
            "force_image_size":448,
            "down_sample_ratio":0.5,
            "pad2square":False,
            "conv_style":'chatml',
            "meta_path":None,
            "use_data_resampling":False,
            "dynamic_image_size":True,
            "use_thumbnail":True,
            "min_dynamic_patch":1,
            "max_dynamic_patch":6,
            "normalize_type":'imagenet'
            }

IGNORE_INDEX = -100

tokenizer = AutoTokenizer.from_pretrained(model_args['lm_model_name_or_path'])

START_IDS = [151644, 77091, 198]
END_IDS = tokenizer.convert_tokens_to_ids('<|im_end|>')
padding_id = tokenizer.pad_token_id


print([tokenizer.decode(START_IDS)])
print(tokenizer.decode(END_IDS))
print(tokenizer.decode(padding_id))

def tokenize(text):
    image_list = text['image_list']
    conversation = text['text']
    
    inputs_with_mask = tokenizer(conversation)
    inputs = inputs_with_mask['input_ids']
    labels = [-100] * len(inputs)
    
    for i in range(len(labels)):
        if inputs[i - len(START_IDS): i]  == START_IDS:
            j = inputs.index(END_IDS, i)
            for k in range(i,j+1):
                labels[k] = inputs[k]

    return dict(
        pixel_values=image_list,
        input_ids=inputs,
        attention_mask=inputs_with_mask['attention_mask'],
        labels=labels,
        )
         
'''
data_process
data_type:
{'text':'data_text','image_list':[]}
'''
instruction_dataset = load_dataset("json", data_files="/visual-pt.json", split="train", cache_dir="/workspace/cache_dir/")
tokenized_dataset = instruction_dataset.map(tokenize, remove_columns=instruction_dataset.column_names, num_proc=32, keep_in_memory=False)
# 对数据集进行shuffle
tokenized_dataset = tokenized_dataset.shuffle()

print(len(tokenized_dataset))
# #1569080
'''
model training
'''
training_args = TrainingArguments(
    output_dir='./Train_Result/VLM-PT-32B',            # output directory
    max_steps = 10000,
    per_device_train_batch_size=8,         # batch size per device during training
    warmup_ratio=0.05,                        # number of warmup steps for learning rate scheduler
    lr_scheduler_type ='warmup_stable_decay',
    lr_scheduler_kwargs ={'num_stable_steps':8000,'num_decay_steps':1500,'min_lr_ratio':0.01},
    weight_decay=0.05,  
    logging_steps=50,
    save_strategy='steps',
    save_steps = 5000,
    learning_rate=3e-5,
    gradient_accumulation_steps=2,
    bf16=True,
    deepspeed='./ds_config_sft.json'
    )

vision_config = InternVisionConfig.from_pretrained(model_args['vision_path'])
vision_config.drop_path_rate = model_args['drop_path_rate']
vision_model = InternVisionModel.from_pretrained(model_args['vision_path'], torch_dtype=torch.bfloat16, config=vision_config)

print('loaded_visionmodel')

llm_config = AutoConfig.from_pretrained(model_args['lm_model_name_or_path'])
llm = AutoModelForCausalLM.from_pretrained(model_args['lm_model_name_or_path'], attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
     
WiNVLM_Model_config = WiNVLM_Config(
            vision_config.to_dict(), llm_config.to_dict(), downsample_ratio=data_args['down_sample_ratio'],
            pad2square=data_args['pad2square'], template=data_args['conv_style'],
            select_layer=model_args['vision_select_layer'], dynamic_image_size=data_args['dynamic_image_size'],
            use_thumbnail=data_args['use_thumbnail'], ps_version=model_args['ps_version'],
            min_dynamic_patch=data_args['min_dynamic_patch'], max_dynamic_patch=data_args['max_dynamic_patch'])

WiNVLM_Model_config.force_image_size = data_args['force_image_size']

model = WiNVLM_Model(WiNVLM_Model_config, vision_model, llm)
model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

model.language_model.config.use_cache = False
model.vision_model.gradient_checkpointing = True
model.vision_model.encoder.gradient_checkpointing = True

if model_args['grad_checkpoint']:
        model.language_model._set_gradient_checkpointing()
print('model_loaded')

def _freeze_params(module):
    for param in module.parameters():
        param.requires_grad = False

if model_args['freeze_vm']:
    # model.vision_model = model.vision_model.eval()
    _freeze_params(model.vision_model)

if model_args['freeze_llm']:
    model.language_model = model.language_model.eval()
    _freeze_params(model.language_model)


def image_stack(batch):
    img_tensor_list = []
    for k in batch:
        for img_file in k['pixel_values']:
            img = Image.open(img_file).convert('RGB')
            if data_args['dynamic_image_size']:
                image = dynamic_preprocess(img)
                img_tensor_list += image
            else:
                img_tensor_list.append(img)
    transform = build_transform(data_args['force_image_size'])
    pixel_values = [transform(image) for image in img_tensor_list]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def the_collate_fn(batch): #batch内参数需要与字典返回参数命名一致，否则会丢失数据
    input_ids = pad_sequence([torch.tensor(f['input_ids']) for f in batch], padding_value=padding_id, batch_first=True)
    pixel_values = image_stack(batch)
    attention_mask = pad_sequence([torch.tensor(f['attention_mask']) for f in batch], padding_value=0, batch_first=True)
    labels = pad_sequence([torch.tensor(f['labels'])  for f in batch], padding_value=IGNORE_INDEX, batch_first=True)
    return {'pixel_values':pixel_values, 'input_ids':input_ids, 'attention_mask':attention_mask, 'labels':labels}

class Mytrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels = inputs["labels"])
        loss, logits = outputs[:2]
        return (loss, logits) if return_outputs else loss

trainer = Mytrainer(model=model, 
                    args=training_args, 
                    train_dataset=tokenized_dataset,
                    data_collator=the_collate_fn,
                    )

trainer.train()