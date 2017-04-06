# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import cv2

def gen_text_from_image(in_path, out_path):
    img = cv2.imread(in_path)
    if img.shape != (224, 224, 3):
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)

    out = open(out_path, "w")
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                out.write(str(img[i, j, k]) + " ")
        out.write("\n")
    out.close()


if __name__ == '__main__':
    print('Convert image...')
    in_path = "../input/cat_224.png"
    out_path = "../input/cat_224.txt"
    gen_text_from_image(in_path, out_path)
