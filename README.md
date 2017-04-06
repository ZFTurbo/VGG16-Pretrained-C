# VGG16-Pretrained-C
Pretrained VGG16 neural net in C language

- This is the VGG16 pretrained convolutional neural net written in pure C. It has no external dependenices.
- Initital weights obtained based on Keras VGG16 model pretrained on ImageNet from official repository: 
https://github.com/fchollet/keras/blob/44bf298ec3236f4a7281be04716a58163469b4de/keras/applications/vgg16.py
- There are some useful python scripts:
1) image_to_text_converter.py - convert any image in format needed by ZFC_VGG16_CPU
2) keras_weights_converter.py - convert any VGG16 Keras weights to format needed by ZFC_VGG16_CPU
3) values_checker.py - program to check if results are the same in Keras and in C program

# Installation

gcc -O3 -fopenmp -lm ZFC_VGG16_CPU.c -o ZFC_VGG16_CPU.exe

# Usage

ZFC_VGG16_CPU.exe <weights_path> <file_with_list_of_images> <output_file> <output_convolution_features_(optional)>

Example: ZFC_VGG16_CPU.exe "weights.txt" "image_list.txt" "results.txt" 1

# Downloads

Weights (~800MB Compressed 7z): 
https://mega.nz/#!LIhjXRhQ!scgNodAkfwWIUZdTcRfmKNHjtUfUb2KiIvfvXdIe-vc

# Advantages

1) No external dependencies
2) Cross platform
3) Can be easily optimized for given weights with a little loss in accuracy. For example: using fixed point instead of float, using all available AVX instructions, optimized multiprocessor support, etc.
4) Faster if you need only features from last convolution layer
5) You can use it for infer operation with weights obtained with transfer learning

# Disadvantages

1) Slow weights reading (it should be generated in some binary file)
2) Use own TEXT format for images and weights
3) Not optimized

# Image format

- Text file
- BGR order
- Exactly 3x224x224 integers in txt file, separated by spaces.
- See image_to_text_converter.py for details.

# Weights format

- Text file
- Weights + Bias by levels on independent lines
- Must contain exactly 138357544 floating point numbers
- Can be generated from any VGG16 Keras weights with Python script: keras_weights_converter.py

# Timings

- Reading weights (all layers): 80 seconds
- Reading weights (convolution only): 8 seconds
- Processing one image (all layers, single core): 22 seconds
- Processing one image (all layers, 7 cores): 6 seconds
- Processing one image (convolution only, single core): 20 seconds
- Processing one image (convolution only, 7 cores): 5 seconds
