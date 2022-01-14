import numpy as np
import cv2
from rknn.api import RKNN
import os
import sys
import torch
from torchvision import transforms as ttf
from PIL import Image
import onnxruntime

import logging
logging.basicConfig(level=logging.DEBUG, \
                    format='%(asctime)s-%(levelname)s: %(message)s')

RGB_MEAN = [123.675, 116.28, 103.53]
RGB_STD = [58.395, 58.395, 58.395]
RGB_MEAN_P = (np.array(RGB_MEAN) / 255.0).tolist() # (0.485, 0.456, 0.406)
RGB_STD_P = (np.array(RGB_STD) / 255.0).tolist() # (0.229, 0.229, 0.229)

def preprocess_pipeline(image):
    """preprocess_pipeline processes the image data.
    
    args:
        image: a numpy array in shape of (224, 224, 3)

    returns:
        image: the processed image numpy array
    """
    assert np.array_equal(image.shape, (224, 224, 3)), \
        "The image's shape requires (224, 224, 3)."

    image = image / 255.0
    logging.debug("Image max: {:.4f}, min: {:.4f}".format(np.max(image), np.min(image)))
    p = ttf.Compose([ttf.Normalize(RGB_MEAN_P, RGB_STD_P)])
    image = image.transpose((2, 0, 1))  # (3, 224, 224)
    image = torch.Tensor(image)
    image = p(image)
    image = torch.unsqueeze(image, axis=0)
    image = image.detach().numpy()
    image = image.transpose((0, 2, 3, 1))  # (1, 224, 224, 3)

    # logging.debug("Image max: {:.4f}, min: {:.4f}".format(np.max(image), np.min(image)))
    # # image = ((image) - RGB_MEAN) / RGB_STD
    # image = ((image - RGB_MEAN) / RGB_STD)
    # image = np.expand_dims(image, axis=0)
    
    image = image.astype(np.float32)
    assert np.array_equal(image.shape, (1, 224, 224, 3)), \
        "The returned image's shape must be (1, 224, 224, 3) Now is {}.".format(image.shape)

    logging.debug("Image type: {}".format(image.dtype))      
    logging.debug("  - Image max: {:.4f}, min: {:.4f}".format(np.max(image), np.min(image)))
    logging.debug("  - Image shape: {}".format(image.shape))
    return image

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

    if len(sys.argv) < 1:
        logging.warning("Usage: python3 test.py model_path")
        sys.exit(1)
    modelfn = sys.argv[1]

    assert os.path.exists(modelfn), \
        "The model file at {} was not found.".format(modelfn)

    input_size_list = [[1, 3, 224, 224]]
    mainfn = os.path.basename(modelfn).split('.')[0]

    # Create RKNN object
    rknn = RKNN(verbose=True, verbose_file='./{}.log'.format(mainfn))

    # pre-process config
    print('--> config model')
    rknn.config(mean_values=RGB_MEAN, std_values=RGB_STD, quant_img_RGB2BGR=False)
    print('done')

    # Load pytorch model
    if modelfn[-4:] == "onnx": 
        logging.info('--> Loading model via rknn.load_onnx')
        ret = rknn.load_onnx(model=modelfn)

        # onnxruntime
        session = onnxruntime.InferenceSession(modelfn, None)
        inputName = session.get_inputs()[0].name
        outputName = session.get_outputs()[0].name
        logging.debug('Input Name: {}, Output Name: {}'.format(inputName, outputName))
    else:
        # print('--> Loading model with torch.load first')
        # model1 = torch.load(modelfn, map_location=torch.device('cpu'))
        # print('model1 is {}'.format(type(model1)))

        # Load pytorch model
        logging.info('--> Loading model via rknn.load_pytorch')
        ret = rknn.load_pytorch(model=modelfn, input_size_list=input_size_list)

    if ret != 0:
        logging.error('Load pytorch model failed!')
        sys.exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    quantized = False
    logging.debug("The quantization type is {}.".format(quantized))
    ret = rknn.build(do_quantization=quantized, dataset='./dataset.txt')
    if ret != 0:
        logging.error('Build pytorch failed!')
        sys.exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(f'./{mainfn}.rknn')
    if ret != 0:
        print(f'Export {mainfn}.rknn failed!')
        sys.exit(ret)
    print('done')

    ret = rknn.load_rknn(f'./{mainfn}.rknn')

    # Set inputs
    img_paths = []
    with open("dataset.txt", "r") as fin:
        for line in fin:
            file_name = line.strip()
            if os.path.exists(file_name):
                img_paths.append(file_name)
            else:
                logging.error("Image {} can't be found.".format(file_name))
    # downsize
    img_paths = img_paths[:3]  

    for img_path in img_paths:
        print("\n--------------------------\n")
        _img = cv2.imread(img_path)
        _img = cv2.resize(_img, (224, 224))    
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)

        # preprocess pipeline
        img = preprocess_pipeline(_img)

        # init runtime environment
        print('--> Init runtime environment')
        ret = rknn.init_runtime()
        if ret != 0:
            print('Init runtime environment failed')
            sys.exit(ret)
        print('done')

        # Inference
        print('--> Running model')
        try:
            outputs = rknn.inference(inputs=[img])
            np.save(f'./pytorch_{mainfn}.npy', outputs[0])

            print("Image Path: {}".format(img_path))
            # show_outputs(softmax(np.array(outputs[0][0])))
            class_idx = np.where(np.array(outputs[0]) > 0.5)[1].tolist()
            if len(class_idx) < 1:
                print("Max Prob. : {:.4f}".format(np.max(outputs[0][0])))
            for idx in class_idx:
                print("Index {} : {:.4f}".format(idx, outputs[0][0][idx]))
            
            img_ = img.transpose((0, 3, 1, 2))
            print("Image shape for onnxruntime: {}".format(img_.shape))
            onnx_outputs = session.run(None, {inputName: img_.astype(np.float32)})[0]
            inferRes = np.where(onnx_outputs > 0.5)
            results = list(zip(inferRes[1], onnx_outputs[inferRes]))
            print("(Index: Prob.): {}".format(results))
        except Exception as err:
            print("Failed in infering the image. ({})".format(err))
            sys.exit(1)

        print("\n--------------------------\n")

    print('done')

    rknn.release()
    sys.exit(0)
