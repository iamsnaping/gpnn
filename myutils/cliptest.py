import torch
# import clip
from PIL import Image
from peft import get_peft_model, LoraConfig
from transformers import CLIPProcessor, CLIPModel

# device='cuda:3'
# print(clip.available_models())
# model,preprocess=clip.load('ViT-B/16',device=device,jit=True)
# mode_path='/home/wu_tian_ci/GAFL/clip_base/clip_base_16.bin'

# img=preprocess(Image.open('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/action_genome/frames/0A8CF.mp4/000083.png')).unsqueeze(0).to(device)
# img2=preprocess(Image.open('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/action_genome/frames/0A8ZT.mp4/000042.png')).unsqueeze(0).to(device)
# img3=torch.concat([img,img2],dim=0)


# lora_config = LoraConfig(
#     r=8,  # 低秩矩阵的秩，可以根据需求调整
#     lora_alpha=16,  # 控制 LoRA 增量的强度，通常设置为较小的值
#     target_modules=["attn.proj_q", "attn.proj_k", "attn.proj_v"],  # 定义你要修改的层，通常是注意力层
#     lora_dropout=0.1,  # LoRA 的 dropout 比率
# )

# # 将 LoRA 应用到 CLIP 模型
# model = get_peft_model(model, lora_config)
# image_feature=model.encode_image(img3)
# print(img3.shape)
# print(image_feature.shape)
# # with torch.no_grad():  
# #     for i in range(1000):
# #         iimg_test=torch.zeros(20,3,224,224).to(device)                                                                                
# #         image_feature=model.encode_image(iimg_test)
# #         print(image_feature.shape)

base_path='/home/wu_tian_ci/GAFL/clip_base/clip-vit-base-patch16'

processor = CLIPProcessor.from_pretrained(base_path,local_files_only=True,)

images=[Image.open('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/action_genome/frames/0A8CF.mp4/000083.png'),
        Image.open('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/action_genome/frames/0A8ZT.mp4/000042.png')]
model=CLIPModel.from_pretrained(base_path,local_files_only=True)
device='cuda:0'
model.to(device)
inputs = processor(images=images, return_tensors="pt", padding=True)
inputs={k:v.to(device) for k,v in inputs.items()}
inputs.pop('input_ids', None)
inputs.pop('attention_mask', None)

# image_value=inputs['pixel_values'].to(device)

outputs=model.get_image_features(**inputs)
print(outputs.shape)