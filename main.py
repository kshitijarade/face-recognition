import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# print(os.getcwd())
img_path = 'C://Users//kshit//AppData//Roaming//Python//Python311//site-packages//insightface//data//images'
file_names = []

for imgFile in os.listdir(img_path):
    tmp = imgFile.split('.', 1)
    file_names.append(tmp[0])

# print(len(file_names))

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)

for template_face in ['tg1', 'ts', 'tt', 'ab']:
    img = ins_get_image(template_face)
    faces = app.get(img)

    #img2 = ins_get_image('ua')
    #ufaces = app.get(img2)

    source_face = faces[0]
    # bbox = source_face['bbox']
    # bbox = [int(b) for b in bbox]
    # plt.imshow(img[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])
    # plt.show()

    # usource_face = ufaces[0]
    # bbox = usource_face['bbox']
    # bbox = [int(b) for b in bbox]
    # plt.imshow(img2[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])
    # plt.show()

    # swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)

    for i, target_file in enumerate(file_names):
        if target_file in ['ua', 'tg', 'tg1', 'tg2', 'ts', 'tt', 'ab', 'pk', 'ss']:
            print(f'template file {target_file}')
        else:
            target_image = ins_get_image(target_file)
            target_face = app.get(target_image)
            target_image = swapper.get(target_image, target_face[0], source_face, paste_back=True)
            # target_image = swapper.get(target_image, target_face[1], usource_face, paste_back=True)
            # plt.imshow(target_image[:, :, ::-1])
            # plt.show()
            cv2.imwrite(f"{os.getcwd()}/output/{template_face}/{i}.jpg", target_image)

    print(f'Completed for {template_face}')

print('All tasks complete')
