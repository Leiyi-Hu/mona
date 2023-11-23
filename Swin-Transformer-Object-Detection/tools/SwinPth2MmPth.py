import torch
import os

# root = '/pretrained_model/'
root = '/pretrained_model/converted_pre_models'

for _, dirs, files in os.walk(root):
    for file in files:
        new_dict = dict()
        state_dict = torch.load(os.path.join(root, file))
        for k in state_dict['model']:
            new_dict['backbone.'+k] = state_dict['model'][k]
        # torch.save(new_dict, os.path.join(root, 'converted_pre_models', file))  # 只保存模型参数
        print(1)
print(1)