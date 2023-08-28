# -*- coding: utf-8 -*-
#Functions for the implementation of Adversarial Attacks on Machine Learning Image Classifiers.
# Tiago Alves <tiago@ime.uerj.br>

import numpy as np
#from numdifftools import conda as ndGradient
import random
import scipy.stats as stats
import torch
import torch.nn.functional as F
from torch.optim import SGD


exp_lambda = 30.0

def gray_to_rgb(height, width, image_gray):
        image_rgb = np.zeros((height, width,3))
        #image_rgb = np.array([])
        for row in range(0,height):
            for col in range(0, width):
                for c in range(3):
                    image_rgb[row][col][c] = image_gray[row*width+col]

        return image_rgb.astype('uint8')


def f_noise(f, x):
    # Adds short tailed distributed noise to function result
    u = random.uniform(0, 1)
    # u = random.randint(2,10)
    noise = stats.expon.rvs(scale=1 / exp_lambda)  # exponential
    # noise = np.exp(-1*u) #exponential
    noise = u > 0.5 and noise or -1 * noise
    dist = f(x) + noise
    ##print "Noise %f"%noise
    return dist

def num_grad(f, x):
    delta = 20
    grad = np.zeros_like(x)  # Match the shape of x
    perturbed_x = np.copy(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        original_value = x[ix]

        # Perturb the current element of x
        perturbed_x[ix] = original_value + delta
        grad[ix] = (f(perturbed_x) - f(x)) / delta

        # Reset the perturbed value
        perturbed_x[ix] = original_value

        it.iternext()

    return grad

def grad_ascent(f, x):
        #print x
    conf = f(x)
    print("Conf is {}".format(conf))
    while conf < 0.8:
        # grad = nd.Gradient(f)(x)
        grad = num_grad(f,x)
        print(grad)
        x += grad

        conf = f(x)
        print("Conf {}".format(conf))
        #count += 1

        return x






def num_ascent(f, x):

    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    conf = f(x)
    print("Conf is {}".format(conf))
    count = 0
    #while conf < 0.4:
    grad = num_grad(f,x)
    #grad = ndGradient(f)(x)
    print(grad)
    #x += grad
    x += 0.8*grad
    conf = f(x)
    print("Conf {}".format(conf))
    return  x



def num_ascent_g(f, x):
    x = torch.tensor(x, requires_grad=True)  # 将输入转换为 PyTorch 张量，并启用梯度跟踪
    #a = x
    optimizer = SGD([x], lr=0.1)  # 定义随机梯度上升优化器，学习率设为0.1
    while True:
        x = x.detach().numpy()
        conf = f(x)
        if conf >=0.4:
            break
        x = torch.from_numpy(x)
        optimizer.zero_grad()  # 清除梯度
    #conf = torch.tensor(conf)
        conf = torch.tensor(conf, requires_grad=True)
        conf.backward()  # 反向传播计算梯度
    #grad = conf.backward()
        optimizer.step()  # 更新输入 x

    x = x.detach().numpy()  # 将张量 x 转换回 NumPy 数组
    conf = f(x)  # 计算优化后输入的置信度
    print("Conf {}".format(conf))

    return x  # 返回优化后的输入 x



def ppgd(f,x):
    epsilons = [ .3, .35, .4, .45, .5,]
    conf = f(x)
    print("Conf is {}".format(conf))
    count = 0
    #step = 2
    #grad = num_grad(f, x)
    for eps in epsilons:

            alpha = eps
            grad = num_grad(f, x)
            sign_data_grad = torch.sign(torch.from_numpy(grad))
            x = torch.from_numpy(x)
            x = x + alpha * sign_data_grad
            x = x.detach().numpy()
            conf = f(x)
            print("Conf {}".format(conf))

                #                                                                                                                                                                                                                                                      x = x.numpy()
    conf = f(x)
    #print("Conf {}".format(conf))





    return x



def pgd_d(f, x, epsilons=0.4, alpha=0.01, num_steps=5):
    x = torch.from_numpy(x)
    conf = f(x)
    x_original = x.clone()  # keep a copy of the original image
    for i in range(num_steps):
        grad = num_grad(f, x.numpy())  # calculate gradient using the function 'f' and numpy version of 'x'
        sign_data_grad = torch.from_numpy(np.sign(grad).astype(np.float64))  # convert sign_data_grad to a tensor with same dtype as x
        x = x + alpha * sign_data_grad

        # Clip the perturbation to make sure the perturbed image is in the epsilon-ball
        perturbation = torch.clamp(x - x_original, -epsilons, epsilons)
        x = torch.clamp(x_original + perturbation, 0, 1)  # ensure the image is valid (0<=x<=1)

        conf = f(x.detach().numpy())
        print(f"Step: {i}, Confidence: {conf}")  # print the confidence at each step

    conf = f(x.detach().numpy())

    print(f"After {num_steps} steps, final Confidence: {conf}")  # print the final confidence after num_steps iterations
    return x.detach().numpy() # return numpy version of 'x'


def pgd_t(model, x, target_label, epsilons=[0.3, 0.4, 0.5, 0.6, 0.7]):
    # Ensure computation is performed on the same device as the input model
    device = next(model.parameters()).device

    # Convert the image to PyTorch tensor and send to device
    x = torch.from_numpy(x).float().to(device)
    x = x.unsqueeze(0)  # add batch dimension

    # Convert the target label to PyTorch tensor and send to device
    target_label = torch.tensor([target_label], dtype=torch.long, device=device)

    # Set requires_grad attribute of tensor. Important for Attack
    x.requires_grad = True

    for epsilon in epsilons:
        # Forward pass
        outputs = model(x)
        loss = F.cross_entropy(outputs, target_label)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = x.grad.data

        # Perform PGD
        x = x + epsilon * torch.sign(data_grad)
        x = x.detach().requires_grad_(True)

    # Return the perturbed image
    return x.detach().cpu().numpy().squeeze(0)


def pppgd(f, x, num_steps=100, initial_alpha=0.5, momentum=0.9):
    conf = f(x)
    print("Initial confidence is {}".format(conf))
    alpha = initial_alpha
    grad = num_grad(f, x)
    sign_data_grad = torch.sign(torch.from_numpy(grad))
    update = torch.zeros_like(sign_data_grad)


    for i in range(num_steps):
        x = torch.from_numpy(x)
        update = momentum * update + alpha * sign_data_grad
        x = x + update
        x = x.detach().numpy()
        conf = f(x)
        print("Step {}, confidence {}".format(i + 1, conf))
        alpha *= 0.99  # learning rate decay
        if conf >= 0.5:
            break


    conf = f(x)
    print("Final confidence is {}".format(conf))

    return x


def pgd(model, x, target_label_index, epsilons, alpha=0.01, num_steps=100, early_stopping=False):
    x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    target_label_index = torch.tensor(target_label_index)

    momentum = 0
    mu = 0.9  # momentum factor

    for i in range(num_steps):
        x.requires_grad_()

        output = model(x.unsqueeze(0))
        loss = -output[0, target_label_index]  # maximize the target class score
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad = x.grad
            grad_sign = grad.sign()
            grad_with_momentum = mu * momentum + grad_sign
            x = x + alpha * grad_with_momentum
            x = x.detach()

        # Early stopping
        if early_stopping:
            output = model(x.unsqueeze(0))
            _, predicted = output.max(1)
            if predicted.cpu() == target_label_index:
                break

    return x.detach().cpu().numpy()



def check_classification(f, img, label):
    image_rgb = np.reshape(img, (3, 32, 32))
    image_rgb = image_rgb.transpose(1,2,0)
    #from PIL import Image
    #print image_rgb.shape
    #img = Image.fromarray(image_rgb, 'RGB')
    #img.show()

    image_rgb = transform_fn(nd.array(image_rgb))
    #print "%s %s"%(label, f(image_rgb))
    return label == f(image_rgb)


def progressbar(n, total):
    p = int(float(n)/total*100)
    #print "%d %d" %(p, n)
    bar = '█'*p + ' '*(100-p)
    sys.stderr.write('\r[%s] %s%s ...\r' % (bar, p, '%'))
    if n == total:
        sys.stderr.write('\n')

def test_accuracy(f):
    import cPickle as pickle
    fp = open("cifar-10-batches-py/data_batch_%d" %batch_number, 'rb')
    batch_dict = pickle.load(fp)
    count = 0
    i = 1
    k = len(batch_dict["labels"])
    for img, label in map(None, batch_dict["data"], batch_dict["labels"]):
        if check_classification(f, img, class_names[label]):
            count += 1
        if count % 10 == 0:
            #print "%d %d" %(count, len(batch_dict["data"]))
            progressbar(i, k)
        i += 1
    return float(count)/len(batch_dict["data"])

def rgb_to_gray(pixels):
    #img_array = 0.2125 * pixels[:,:,0] + 0.7154 * pixels[:,:,1] + 0.0721 * pixels[:,:,2]
    r, g, b = pixels[:,:,0] , pixels[:,:,1] , pixels[:,:,2]
    img_array = 0.2125 * r + 0.7154 * g + 0.0721 * b

    height, width = img_array.shape
    print("Shape {}".format(img_array.shape))
    return (height, width, img_array.reshape(width*height).astype(int))

