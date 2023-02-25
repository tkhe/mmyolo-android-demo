# mmyolo-android-demo

ncnn android demo for [MMYOLO](https://github.com/open-mmlab/mmyolo)

## TODO

- [ ] PPYOLOE
- [x] RTMDET
- [x] YOLOv5
- [ ] YOLOv6
- [ ] YOLOv7
- [ ] YOLOv8
- [ ] YOLOX

## How to Build and Run

### step1

https://github.com/Tencent/ncnn/releases

* Download ncnn-YYYYMMDD-android-vulkan.zip or build ncnn for android yourself
* Extract ncnn-YYYYMMDD-android-vulkan.zip into **app/src/main/jni** and change the **ncnn_DIR** path to yours in **app/src/main/jni/CMakeLists.txt**

### step2

https://github.com/nihui/opencv-mobile

* Download opencv-mobile-XYZ-android.zip
* Extract opencv-mobile-XYZ-android.zip into **app/src/main/jni** and change the **OpenCV_DIR** path to yours in **app/src/main/jni/CMakeLists.txt**

### step3

* Open this project with Android Studio, build it and enjoy!

## Notes
* Android ndk camera is used for best efficiency
* Crash may happen on very old devices for lacking HAL3 camera interface
* All models are manually modified to accept dynamic input shape
* Most small models run slower on GPU than on CPU, this is common
* FPS may be lower in dark environment because of longer camera exposure time

## Acknowledgement

This repo depends on
* [ncnn](https://github.com/Tencent/ncnn)
* [opencv-mobile](https://github.com/nihui/opencv-mobile)

All models are exported from
* [mmyolo](https://github.com/open-mmlab/mmyolo)

This repo borrows lots of code from [ncnn-android-nanodet](https://github.com/nihui/ncnn-android-nanodet).

## License

This project is released under the [GPL 3.0 license](LICENSE).
