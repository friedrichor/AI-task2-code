import os
import cv2
import json
import argparse

import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter

import params
from data_process import crop_image_from_gray


def main(args):
    print(args)
    device = params.device
    print(f"using {device} device.")

    num_classes = params.num_classes
    img_size = params.img_size
    data_transform = transforms.Compose([
         # transforms.Resize((int(img_size * 1.143), int(img_size * 1.143))),
         # transforms.CenterCrop((img_size, img_size)),
         transforms.Resize((img_size, img_size)),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # read class_indict
    json_path = args.path_json
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = params.model.to(device)
    # load model weights
    if args.model_name == '':
        model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        weights = os.path.join(params.path_weights, args.model_name + '.pth')  # 模型保存路径
        model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    test_path = args.path_test
    num_acc = 0
    num_all = 0
    label_true = []
    label_predict = []
    # 数据预处理
    for cls in os.listdir(test_path):
        print('cls =', cls)
        path_cls = os.path.join(test_path, cls)
        for patient in os.listdir(path_cls):
            print('patient =', patient)
            path_patient = os.path.join(path_cls, patient)
            num_all += len(os.listdir(path_patient))
            label_predict_tmp = []
            for img_path in os.listdir(path_patient):
                label_true.append(class_indict[cls])
                img_path = os.path.join(path_patient, img_path)
                assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
                img = Image.open(img_path).convert('RGB')

                # # 数据预处理
                # img_pro = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                # img_pro = crop_image_from_gray(img_pro)
                # img_pro = cv2.resize(img_pro, (224, 224))
                # # 恢复PIL格式
                # img = Image.fromarray(cv2.cvtColor(img_pro, cv2.COLOR_BGR2RGB))

                # [N, C, H, W]
                img = data_transform(img)
                # expand batch dimension
                img = torch.unsqueeze(img, dim=0)

                with torch.no_grad():
                    # predict class
                    output = torch.squeeze(model(img.to(device))).cpu()
                    predict = torch.softmax(output, dim=0)
                    predict_cls = torch.argmax(predict).numpy()  # choice = [0,1,2]
                    print(predict_cls)
                    label_predict_tmp.append(int(predict_cls))

                    # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cls)],
                    #                                              predict[predict_cls].numpy())
                    # print(print_res)
                    # for i in range(len(predict)):
                    #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)], predict[i].numpy()))

                    if predict_cls == class_indict[cls]:
                        num_acc += 1
            # print('label_predict_tmp =', label_predict_tmp)
            # number = Counter(label_predict_tmp)
            # result = number.most_common()
            # num_others = sum(i[1] for i in result[1:])
            # if result[0][1] > num_others:
            #     label_predict = label_predict + [result[0][0]] * len(label_predict_tmp)
            # else:
            #     label_predict = label_predict + label_predict_tmp
            label_predict = label_predict + label_predict_tmp

    print('label_predict =', label_predict)
    print('num of test datasets =', num_all)
    print('acc =', num_acc / num_all)
    category_show(label_true, label_predict)


# 展示各类的准确率、召回率、f1-score，及混淆矩阵可视化
def category_show(y_true, y_predict):
    target_names = ['Image_DCM', 'Image_HCM', 'Image_NOR']
    print(classification_report(y_true, y_predict, target_names=target_names))
    cm = confusion_matrix(y_true, y_predict)
    cm_display = ConfusionMatrixDisplay(cm).plot()


def parse_opt():
    parser = argparse.ArgumentParser()
    # 模型参数
    parser.add_argument('--model-name', type=str, default='')
    parser.add_argument('--weights', nargs='+', type=str, default=params.weights, help='model weights path')
    # 测试集路径
    parser.add_argument('--path_test', type=str, default=params.path_test, help='test datasets path')
    parser.add_argument('--path_json', type=str, default=params.path_json, help='class_indice.json path')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
