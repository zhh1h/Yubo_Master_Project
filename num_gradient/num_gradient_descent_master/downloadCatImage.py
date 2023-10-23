import os
import requests


def download_dataset(url, save_folder, file_name):
    """
    下载数据集并保存到指定文件夹。

    :param url: 数据集的URL
    :param save_folder: 保存数据集的文件夹路径
    :param file_name: 保存的文件名
    """
    # 创建文件夹（如果不存在）
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 获取数据集内容
    response = requests.get(url)
    response.raise_for_status()  # 如果请求失败，抛出异常

    # 保存数据集到文件
    with open(os.path.join(save_folder, file_name), 'wb') as f:
        f.write(response.content)
    print(f"下载完成并保存到: {os.path.join(save_folder, file_name)}")

# 使用示例：
download_dataset('https://storage.googleapis.com/kaggle-data-sets/878523/1495782/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20231023%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20231023T034039Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=5bb94e3833a2ea6dfc767c69da2cd33780b6d90e9551294836aabccb13d902d639cc020fa7f3902c7071edd426545d4edd8e0fb318514e173c26dae899d39a6a4225db72559eade43b429ca74a24c2d51957476caa9bd252bb186cb1558c97ddb08c91363a33ac49be6e2313445d83baad9f5ea4ebc9bf443a0bea26e894c65f8fe2b5b165a54636ff88d8aa5273a220efd3fa5e0806f9595c7e263ac7b47d0e76f2f1f5d9a42702b9739643565be31e3fbdb66e55396c92ebf4812fc17e083c1b97d5980bdffcd7fe3d01804aac6045742c7307b880c7dc6908ee6b8732d2925999098512fdaf0d4f401f46db5d5080e666e97716b1b5fec1b32ee5105a6238', '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/caltech256AimImage/truck', 'truckorCar.zip')
