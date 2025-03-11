# import torch

# # 加载两个.pt格式的权重文件
# weight_file_1 = 'classss.pt'
# # weight_file_2 = 'path/to/weight2.pt'

# weights_1 = torch.load(weight_file_1, map_location=torch.device('cpu'))

# # 获取每个权重文件中的model部分的参数键
# model_keys_1 = set(weights_1['model'].keys())

# # 打印每个权重文件中的model部分的参数键
# print("Keys in weight file 1 (model):")
# for key in model_keys_1:
#     print(key)

import torch

# # 加载权重文件
# weight_file = 'save/WeaklyCon/path_models/SimCLR_path_resnet_lr_0.125_decay_0.0001_bsz_128_temp_0.1_trial_0_cosine/ckpt_epoch_1.pt'
# weights = torch.load(weight_file, map_location=torch.device('cpu'))

# # 获取模型对象的状态字典
# model_state_dict = weights['model'].state_dict()

# # 打印权重文件中模型部分的所有键
# print("Keys in the original weight file (model):")
# for key in model_state_dict.keys():
#     print(key)

# 加载权重文件
weight_file = 'save_yolov8m/Simclr/path_models/SimCLR_path_resnet_lr_0.125_decay_0.0001_bsz_128_temp_0.1_trial_0_cosine/ckpt_epoch_221.pt'
weights = torch.load(weight_file, map_location=torch.device('cpu'))

# 打印权重文件中model部分的所有键
print("Keys in the original weight file (model):")
for key in weights['model'].keys():
    print(key)

# 函数：将键中的 encoder.module. 改为 model.
def update_keys(state_dict, old_prefix, new_prefix):
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key
        if key.startswith(old_prefix):
            new_key = new_prefix + key[len(old_prefix):]  # 替换前缀
        new_state_dict[new_key] = state_dict[key]
    return new_state_dict

# 更新键
old_prefix = 'encoder.module.layers'
new_prefix = 'model'
weights['model'] = update_keys(weights['model'], old_prefix, new_prefix)

# old_prefix = 'encoder.module.'
# new_prefix = 'model.'
# weights['model'] = update_keys(weights['model'], old_prefix, new_prefix)

# old_prefix = 'encoder.module.layers2.0'
# new_prefix = 'model.5'
# weights['model'] = update_keys(weights['model'], old_prefix, new_prefix)

# old_prefix = 'encoder.module.layers2.1'
# new_prefix = 'model.6'
# weights['model'] = update_keys(weights['model'], old_prefix, new_prefix)

# old_prefix = 'encoder.module.layers3.0'
# new_prefix = 'model.7'
# weights['model'] = update_keys(weights['model'], old_prefix, new_prefix)

# old_prefix = 'encoder.module.layers3.1'
# new_prefix = 'model.8'
# weights['model'] = update_keys(weights['model'], old_prefix, new_prefix)

# old_prefix = 'encoder.module.layers4.0'
# new_prefix = 'model.9'
# weights['model'] = update_keys(weights['model'], old_prefix, new_prefix)

# old_prefix = 'encoder.module.p3_for_upsample'
# new_prefix = 'model.12'
# weights['model'] = update_keys(weights['model'], old_prefix, new_prefix)

# old_prefix = 'encoder.module.p4_for_upsample'
# new_prefix = 'model.15'
# weights['model'] = update_keys(weights['model'], old_prefix, new_prefix)

# old_prefix = 'encoder.module.p4_for_downsample'
# new_prefix = 'model.18'
# weights['model'] = update_keys(weights['model'], old_prefix, new_prefix)

# old_prefix = 'encoder.module.p5_for_downsample'
# new_prefix = 'model.21'
# weights['model'] = update_keys(weights['model'], old_prefix, new_prefix)

# old_prefix = 'encoder.module.p4_down'
# new_prefix = 'model.16'
# weights['model'] = update_keys(weights['model'], old_prefix, new_prefix)

# old_prefix = 'encoder.module.p5_down'
# new_prefix = 'model.19'
# weights['model'] = update_keys(weights['model'], old_prefix, new_prefix)

# 打印更新后的权重文件中model部分的所有键
print("\nKeys in the updated weight file (model):")
for key in weights['model'].keys():
    print(key)

# 保存新的权重文件
new_weight_file = 'simclr_yolo8m.pt'
torch.save(weights, new_weight_file)

print("\nUpdated weight file saved as:", new_weight_file)