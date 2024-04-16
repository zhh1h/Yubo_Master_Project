import requests

url = 'https://www.kaggle.com/datasets/tongpython/cat-and-dog/download?datasetVersionNumber=1'
# 指定路径，例如 '/path/to/your/folder/filename.extension'
save_path = '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/caltech256AimImage/dog.zip'

response = requests.get(url, allow_redirects=True)
with open(save_path, 'wb') as f:
    f.write(response.content)
