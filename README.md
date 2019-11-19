# RetinaFace

## Environment
- Caffe
- TensorRT-6.0.1.5
- Cuda 9.0

## Demo
```
$ mkdir build
$ cd build/
$ cmake ../
$ make
$ ./retinaface img_path
```
you need to modify dependency path in CmakeList file.

## Speed

test hardware：1080Ti

![](./data/res.jpg)