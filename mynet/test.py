import numpy as np
import cv2
from rknn.api import RKNN
import torchvision.models as models
import torch
import os

mainfn='model_best'
def show_outputs(output):
    output_sorted = sorted(output, reverse=True)
    top5_str = '\n-----TOP 5-----\n'
    for i in range(5):
        value = output_sorted[i]
        index = np.where(output == value)
        for j in range(len(index)):
            if (i + j) >= 5:
                break
            if value > 0:
                topi = '{}: {}\n'.format(index[j], value)
            else:
                topi = '-1: 0.0\n'
            top5_str += topi
    print(top5_str)


def show_perfs(perfs):
    perfs = 'perfs: {}\n'.format(perfs)
    print(perfs)


def softmax(x):
    return np.exp(x)/sum(np.exp(x))


if __name__ == '__main__':
    modelfn = f'./{mainfn}.pth.tar'
    input_size_list = [[1, 3, 224, 224]]

    # Create RKNN object
    rknn = RKNN()

    # pre-process config
    print('--> config model')
    rknn.config(mean_values=[123.675, 116.28, 103.53], std_values=[58.395, 58.395, 58.395])
    print('done')

    # Load pytorch model
    print('--> Loading model')
    ret = rknn.load_pytorch(model=modelfn, input_size_list=input_size_list)
    if ret != 0:
        print('Load pytorch model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    if ret != 0:
        print('Build pytorch failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(f'./{mainfn}.rknn')
    if ret != 0:
        print(f'Export {mainfn}.rknn failed!')
        exit(ret)
    print('done')

    ret = rknn.load_rknn(f'./{mainfn}.rknn')

    # Set inputs
    img = cv2.imread('./space_shuttle_224.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    np.save(f'./pytorch_{mainfn}.npy', outputs[0])

    show_outputs(softmax(np.array(outputs[0][0])))
    print('done')

    rknn.release()
