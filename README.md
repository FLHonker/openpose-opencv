# 基于opencv-DNN和caffemodel的人体姿态估计

## Python3版

使用方法，命令行进入代码所在目录执行：
```python
python openpose.py --model pose.caffemodel --proto pose.prototxt --dataset MPI
```
--model 参数和 --proto 参数分别是预先训练好的人体姿态模型和配置文件， --dataset 选择使用`COCO` or `MPI`数据集。因为模型文件很大，并不包括在 OpenCV 代码库中，可以在 [Openpose 项目](https://github.com/CMU-Perceptual-Computing-Lab/openpose)找到下载地址。

另外可以通过 --input 参数指定识别的图片或视频地址，默认则使用摄像头实时采集。

## C++版

Frank使用C++ opencv肝了一周，相比python版每张图片的处理提速0.3秒，使用Frank给提供的`build.sh`脚本直接在Linux下编译生成`cv_pose.out`，运行即可。提供图片处理的接口可直接调用。


## 参考文章：

* [**AI来尬舞！用深度学习制作自定义《堡垒之夜》舞蹈](https://zhuanlan.zhihu.com/p/52304809?utm_source=qq&utm_medium=social&utm_oi=861536217358020608)

* [OpenPose：实时多人2D姿态估计](https://zhuanlan.zhihu.com/p/37526892?utm_source=qq&utm_medium=social&utm_oi=861536217358020608)

* [用 Python 实现抖音尬舞机](https://zhuanlan.zhihu.com/p/47536632?utm_source=qq&utm_medium=social&utm_oi=861536217358020608)

* [动作识别初体验](https://zhuanlan.zhihu.com/p/40574587?utm_source=qq&utm_medium=social&utm_oi=861536217358020608)

* [Image-to-Image Translation with Conditional Adversarial Nets](https://phillipi.github.io/pix2pix/)

* [**CVPR2017-Realtime Multi-Person Pose Estimation](https://arxiv.org/abs/1611.08050)

* [【超越CycleGAN】这个人体动态迁移技术让白痴变舞王](http://www.sohu.com/a/249987154_473283)

* [-Everybody Dance Now](https://arxiv.org/pdf/1808.07371.pdf)


## github

* [**keras_Realtime_Multi-Person_Pose_Estimation](https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation)

* [**pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

* [*openpose:opencv+caffemodel](https://github.com/FLHonker/openpose-opencv)

* [*Realtime_Multi-Person_Pose_Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)

* [pose-tensorflow](https://github.com/eldar/pose-tensorflow)

* [GokuMohandas/practicalAI](https://github.com/GokuMohandas/practicalAI)

* [chenyilun95/tf-cpn](https://github.com/chenyilun95/tf-cpn)

* [jsn5/dancenet](https://github.com/jsn5/dancenet)

* [-yenchenlin/pix2pix-tensorflow](https://github.com/yenchenlin/pix2pix-tensorflow)

* [-affinelayer/pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow)

* [-pytorch-EverybodyDanceNow](https://github.com/nyoki-mtl/pytorch-EverybodyDanceNow)

