import os.path as osp
import torch
import torch.nn as nn

import knockoff.models.cifar
import knockoff.models.imagenet
import knockoff.models.mnist


import os.path as osp
import torch
import torch.nn as nn
import torchvision.models as torch_models
import sys

sys.path.append("/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/knockoff/adversary")
import torchvision

import knockoff.models.cifar as cifar_models
import knockoff.models.imagenet as imagenet_models
import knockoff.models.mnist as mnist_models


# def get_net(modelname, modeltype, pretrained=None, **kwargs):
#     assert modeltype in ('mnist', 'cifar', 'imagenet'), "Unsupported model type"
#     if pretrained:
#         return get_pretrainednet(modelname, modeltype, pretrained, **kwargs)
#     else:
#         model_class = eval(f'knockoff.models.{modeltype}.{modelname}')
#         return model_class(**kwargs)
import requests
import os

def download_pretrained_model(url, output_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Model downloaded successfully and saved to {output_path}")
    else:
        raise Exception(f"Failed to download model. Status code: {response.status_code}")

# 下载预训练模型文件
url = "https://huggingface.co/amehta633/cifar-10-vgg-pretrained/resolve/main/final_model.pth"
output_path = "final_model.pth"

if not os.path.exists(output_path):
    download_pretrained_model(url, output_path)
else:
    print(f"Model already exists at {output_path}")
def get_net(modelname, modeltype, pretrained=None, **kwargs):
    assert modeltype in ('mnist', 'cifar', 'imagenet'), "Unsupported model type"
    if pretrained:
        return get_pretrainednet(modelname, modeltype, pretrained, **kwargs)
    else:
        model_class = eval(f'knockoff.models.{modeltype}.{modelname}')
        return model_class(**kwargs)

# def get_pretrainednet(modelname, modeltype, pretrained='cifar', num_classes=1000, **kwargs):
#     model_class = eval(f'knockoff.models.{modeltype}.{modelname}')
#
#     if pretrained.lower() == 'imagenet':
#         # Load model pretrained on ImageNet
#         model = model_class(pretrained=True, **kwargs)
#     elif pretrained.lower() == 'cifar':
#         # Assuming no specific CIFAR pretrained model, load default configuration
#         model = model_class(pretrained=False, **kwargs)  # Load without pretrained weights
#         # Optional: Customize the model for CIFAR if necessary
#         # e.g., adjust layers, optimizers, etc., specific to CIFAR
#     elif osp.exists(pretrained):
#         # Load model from a provided path
#         model = model_class(**kwargs)
#         checkpoint = torch.load(pretrained)
#         model.load_state_dict(checkpoint['state_dict'])
#     else:
#         raise ValueError(f'Pretrained model path or keyword "{pretrained}" does not exist')
#
#     if num_classes != 1000:  # Assume 1000 is the default for ImageNet
#         in_features = model.last_linear.in_features
#         model.last_linear = nn.Linear(in_features, num_classes)
#
#     return model

# def get_pretrainednet(modelname, modeltype, pretrained, **kwargs):
#     model_class = eval(f'knockoff.models.{modeltype}.{modelname}')
#     model = model_class(pretrained=pretrained, **kwargs)
#
#     # 尝试访问最后一个全连接层
#     try:
#         last_linear = model.classifier[-1]
#         if isinstance(last_linear, nn.Linear):
#             in_features = last_linear.in_features
#             num_classes = kwargs.get('num_classes', 10)  # 默认类别数为1000
#             last_linear = nn.Linear(in_features, num_classes)
#             model.classifier[-1] = last_linear
#     except AttributeError as e:
#         raise AttributeError(f"模型 '{modelname}' 中没有找到预期的全连接层。确保模型定义包含名为 'classifier' 的 Sequential 容器，并且其中包含全连接层。") from e
#
#     return model

# def get_pretrainednet(modelname, modeltype, pretrained='imagenet', num_classes=10, **kwargs):
#     if pretrained.lower() in ['imagenet', 'cifar']:
#         # Load ImageNet pretrained model
#         model = torch_models.vgg19(pretrained=True, **kwargs)
#
#         # Modify the classifier to match the number of classes in CIFAR-10
#         in_features = model.classifier[-1].in_features
#         model.classifier[-1] = nn.Linear(in_features, num_classes)
#     elif osp.exists(pretrained):
#         # Load model from a provided path
#         model_class = eval(f'knockoff.models.{modeltype}.{modelname}')
#         model = model_class(**kwargs)
#         checkpoint = torch.load(pretrained)
#         model.load_state_dict(checkpoint['state_dict'])
#     else:
#         raise ValueError(f'Pretrained model path or keyword "{pretrained}" does not exist')
#
#     if isinstance(model.classifier, nn.Sequential):
#         last_linear = model.classifier[-1]
#         if isinstance(last_linear, nn.Linear):
#             in_features = last_linear.in_features
#             model.classifier[-1] = nn.Linear(in_features, num_classes)
#     elif isinstance(model.classifier, nn.Linear):
#         in_features = model.classifier.in_features
#         model.classifier = nn.Linear(in_features, num_classes)
#     else:
#         raise TypeError("Unexpected type for classifier.")
#
#     return model


def get_pretrainednet(modelname, modeltype, pretrained='cifar', num_classes=10, **kwargs):
    if pretrained.lower() == 'imagenet':
        # Load ImageNet pretrained model
        model = torch_models.vgg19(pretrained=True, **kwargs)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif pretrained.lower() == 'cifar':
        # Load CIFAR-10 pretrained model from Hugging Face
        model = torch_models.vgg19(pretrained=False, **kwargs)
        checkpoint_path = '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/knockoff/adversary/final_model.pth'

        if osp.exists(checkpoint_path):
            try:
                model.load_state_dict(torch.load(checkpoint_path))
            except RuntimeError as e:
                print(f"RuntimeError: {e}, trying torch.jit.load()")
                model = torch.jit.load(checkpoint_path)
        else:
            raise ValueError(f'CIFAR-10 pretrained model path "{checkpoint_path}" does not exist')

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif osp.exists(pretrained):
        model_class = eval(f'knockoff.models.{modeltype}.{modelname}')
        model = model_class(**kwargs)
        checkpoint = torch.load(pretrained)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError(f'Pretrained model path or keyword "{pretrained}" does not exist')

    return model


# def get_pretrainednet(modelname, modeltype, pretrained='cifar', num_classes=10, **kwargs):
#     model_class = eval(f'knockoff.models.{modeltype}.{modelname}')
#     #model = model_class(pretrained=True, **kwargs)
#
#     if pretrained.lower() == 'imagenet':
#             # Load model pretrained on ImageNet
#         model = model_class(pretrained=True, **kwargs)
#     elif pretrained.lower() == 'cifar':
#             # Assuming no specific CIFAR pretrained model, load default configuration
#         model = model_class(pretrained=True, **kwargs)  # Load without pretrained weights
#             # Optional: Customize the model for CIFAR if necessary
#             # e.g., adjust layers, optimizers, etc., specific to CIFAR
#     elif osp.exists(pretrained):
#             # Load model from a provided path
#         model = model_class(**kwargs)
#         checkpoint = torch.load(pretrained)
#         model.load_state_dict(checkpoint['state_dict'])
#     else:
#         raise ValueError(f'Pretrained model path or keyword "{pretrained}" does not exist')
#     # # Load model pretrained on ImageNet
#     #         model = model_class(pretrained=True, **kwargs)
#     #     elif pretrained.lower() == 'cifar':
#     #         # Assuming no specific CIFAR pretrained model, load default configuration
#     #         model = model_class(pretrained=False, **kwargs)  # Load without pretrained weights
#     #         # Optional: Customize the model for CIFAR if necessary
#     #         # e.g., adjust layers, optimizers, etc., specific to CIFAR
#
#     if isinstance(model.classifier, nn.Sequential):
#         last_linear = model.classifier[-1]
#         if isinstance(last_linear, nn.Linear):
#             in_features = last_linear.in_features
#             num_classes = kwargs.get('num_classes', 10)
#             model.classifier[-1] = nn.Linear(in_features, num_classes)
#     elif isinstance(model.classifier, nn.Linear):
#         in_features = model.classifier.in_features
#         num_classes = kwargs.get('num_classes', 10)
#         model.classifier = nn.Linear(in_features, num_classes)
#     else:
#         raise TypeError("Unexpected type for classifier.")
#
#     return model


# def get_pretrainednet(modelname, modeltype, pretrained, **kwargs):
#     model_class = eval(f'knockoff.models.{modeltype}.{modelname}')
#     model = model_class(pretrained=pretrained, **kwargs)
#
#     if isinstance(model.classifier, nn.Sequential):
#         last_linear = model.classifier[-1]
#         if isinstance(last_linear, nn.Linear):
#             in_features = last_linear.in_features
#             num_classes = kwargs.get('num_classes', 1000)
#             model.classifier[-1] = nn.Linear(in_features, num_classes)
#     elif isinstance(model.classifier, nn.Linear):
#         in_features = model.classifier.in_features
#         num_classes = kwargs.get('num_classes', 1000)
#         model.classifier = nn.Linear(in_features, num_classes)
#     else:
#         raise TypeError("Unexpected type for classifier.")
#
#     return model


def copy_weights_(src_state_dict, dst_state_dict):
    n_params = len(src_state_dict)
    n_success, n_skipped, n_shape_mismatch = 0, 0, 0

    for src_param_name, src_param in src_state_dict.items():
        dst_param = dst_state_dict.get(src_param_name)
        if dst_param:
            if dst_param.data.shape == src_param.data.shape:
                dst_param.data.copy_(src_param.data)
                n_success += 1
            else:
                print(f'Mismatch: {src_param_name} ({dst_param.data.shape} != {src_param.data.shape})')
                n_shape_mismatch += 1
        else:
            n_skipped += 1

    print(
        f'=> # Success param blocks loaded = {n_success}/{n_params}, # Skipped = {n_skipped}, # Shape-mismatch = {n_shape_mismatch}')




#
# def get_net(modelname, modeltype, pretrained=None, **kwargs):
#     assert modeltype in ('mnist', 'cifar', 'imagenet')
#     # print('[DEBUG] pretrained={}\tnum_classes={}'.format(pretrained, kwargs['num_classes']))
#     if pretrained and pretrained is not None:
#         return get_pretrainednet(modelname, modeltype, pretrained, **kwargs)
#     else:
#         try:
#             # This should have ideally worked:
#             model = eval('knockoff.models.{}.{}'.format(modeltype, modelname))(**kwargs)
#         except AssertionError:
#             # But, there's a bug in pretrained models which ignores the num_classes attribute.
#             # So, temporarily load the model and replace the last linear layer
#             model = eval('knockoff.models.{}.{}'.format(modeltype, modelname))()
#             if 'num_classes' in kwargs:
#                 num_classes = kwargs['num_classes']
#                 in_feat = model.last_linear.in_features
#                 model.last_linear = nn.Linear(in_feat, num_classes)
#         return model
#
#
# def get_pretrainednet(modelname, modeltype, pretrained='imagenet', num_classes=1000, **kwargs):
#     if pretrained == 'imagenet':
#         return get_imagenet_pretrainednet(modelname, num_classes, **kwargs)
#     elif osp.exists(pretrained):
#         try:
#             # This should have ideally worked:
#             model = eval('knockoff.models.{}.{}'.format(modeltype, modelname))(num_classes=num_classes, **kwargs)
#         except AssertionError:
#             # print('[DEBUG] pretrained={}\tnum_classes={}'.format(pretrained, num_classes))
#             # But, there's a bug in pretrained models which ignores the num_classes attribute.
#             # So, temporarily load the model and replace the last linear layer
#             model = eval('knockoff.models.{}.{}'.format(modeltype, modelname))()
#             in_feat = model.last_linear.in_features
#             model.last_linear = nn.Linear(in_feat, num_classes)
#         checkpoint = torch.load(pretrained)
#         pretrained_state_dict = checkpoint.get('state_dict', checkpoint)
#         copy_weights_(pretrained_state_dict, model.state_dict())
#         return model
#     else:
#         raise ValueError('Currently only supported for imagenet or existing pretrained models')
#
#
# def get_imagenet_pretrainednet(modelname, num_classes=1000, **kwargs):
#     valid_models = knockoff.models.imagenet.__dict__.keys()
#     assert modelname in valid_models, 'Model not recognized, Supported models = {}'.format(valid_models)
#     model = knockoff.models.imagenet.__dict__[modelname](pretrained='imagenet')
#     if num_classes != 1000:
#         # Replace last linear layer
#         in_features = model.last_linear.in_features
#         out_features = num_classes
#         model.last_linear = nn.Linear(in_features, out_features, bias=True)
#     return model
#
#
# def copy_weights_(src_state_dict, dst_state_dict):
#     n_params = len(src_state_dict)
#     n_success, n_skipped, n_shape_mismatch = 0, 0, 0
#
#     for i, (src_param_name, src_param) in enumerate(src_state_dict.items()):
#         if src_param_name in dst_state_dict:
#             dst_param = dst_state_dict[src_param_name]
#             if dst_param.data.shape == src_param.data.shape:
#                 dst_param.data.copy_(src_param.data)
#                 n_success += 1
#             else:
#                 print('Mismatch: {} ({} != {})'.format(src_param_name, dst_param.data.shape, src_param.data.shape))
#                 n_shape_mismatch += 1
#         else:
#             n_skipped += 1
#     print('=> # Success param blocks loaded = {}/{}, '
#           '# Skipped = {}, # Shape-mismatch = {}'.format(n_success, n_params, n_skipped, n_shape_mismatch))
