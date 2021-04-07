# coding=utf-8
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np #导入数值计算拓展模块
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

import tqdm
def point_cloud_generate(limit,ps):
    x_lim = np.linspace(-limit, limit, ps)
    y_lim = np.linspace(-limit, limit, ps)
    z_lim = np.linspace(-limit, limit, ps)
    X_points = []  # 用来存放绘图点X坐标
    Y_points = []  # 用来存放绘图点Y坐标
    Z_points = []  # 用来存放绘图点Z坐标
    for x in x_lim:
        for y in y_lim:
            for z in z_lim:
                if (x ** 2 + (9 / 4) * y ** 2 + z ** 2 - 1) ** 3 - (9 / 80) * y ** 2 * z ** 3 - x ** 2 * z ** 3 <= 0:
                    X_points.append(x)
                    Y_points.append(y)
                    Z_points.append(z)
    print('d p g')
    points_l = len(X_points)
    points = np.ndarray((points_l, 3))
    for ind,i in enumerate(X_points) :
        j = Y_points[ind]
        k = Z_points[ind]
        points[ind] = [i, j ,k]
    print(points)
    np.save('heart_points', points)

def writer_init(embedding_files, image=None, visual_name='embedding_discipline'):
    writer = SummaryWriter('runs/' + visual_name)
    embedding = np.load(embedding_files)
    if image is not None :
        image_label = np.repeat(image, embedding.shape[0]).reshape((-1,image.shape[0],image.shape[1],image.shape[2]))
    else :
        image_label = None
    print(image_label.shape)
    writer.add_embedding(
        embedding,
        label_img = torch.from_numpy(image_label)
    )

def read_img(f) :
    im = Image.open(f)
    im.show()
    img = np.array(im)  # image类 转 numpy

    return img

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print("Hi, {0}".format(name))  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img = read_img('img/img.png')
    writer_init('heart_points.npy', img)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
