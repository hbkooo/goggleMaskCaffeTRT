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
$ ./goggleMask [classify|detect] [path/image.jpg | path/to/filelist.txt]
```
you need to modify dependency path in CmakeList file.

## Speed

test hardware：1080Ti

- TensorRT: detect_time : 12.2, classify_time : 1.1
- caffe : detect_time : 38, classify_time : 6

![](./result/result.jpg)

## Package

编译成功后在 build 目录下会生成打包的 libgogglemask.so 动态库，只需要使用 goggleMask 目录中的 [goggleMaskAPI.h](goggleMask/goggleMaskAPI.h) 头文件即可，在外部项目中使用。

## Note

当第一次运行的时候，可能会出现模型转换错误的问题，不过没有问题，再次运行即可。
