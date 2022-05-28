import os
import shutil


def data_merge(data_ori: str, data_pro: str, mode: str):
    for cls in os.listdir(data_ori):
        Image_path_ori = os.path.join(data_ori, cls, 'png', mode)
        Image_path_pro = os.path.join(data_pro, cls)
        if os.path.exists(Image_path_pro):
            shutil.rmtree(Image_path_pro)
        os.makedirs(Image_path_pro)

        for patient in os.listdir(Image_path_ori):
            patient_path = os.path.join(Image_path_ori, patient)
            for img in os.listdir(patient_path):
                img_path_ori = os.path.join(patient_path, img)
                img_path_pro = os.path.join(Image_path_pro, str(patient) + '_' + img)
                shutil.copy(img_path_ori, img_path_pro)


if __name__ == '__main__':
    data_ori = '../Heart Data'
    Image_pro = 'Image_classify'
    Label_pro = 'Label_classify'
    # data_merge(data_ori, Image_pro, 'Image')
    data_merge(data_ori, Label_pro, 'Label')