# -*- coding: utf-8 -*-
# Usage: python image_to_text_converter.py <path to input image (.jpg, .png)> <path to output image in text format (.txt)>
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import cv2
import sys

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
    if len(sys.argv) != 3:
        print('Usage: python image_to_text_converter.py <path to input image (.jpg, .png)> <path to output image in text format (.txt)>')
    else:
        in_path = sys.argv[1]
        out_path = sys.argv[2]
        gen_text_from_image(in_path, out_path)
        print('Complete!')
