# 用 Python 实现抖音尬舞机

使用方法，命令行进入代码所在目录执行：
```python
python openpose.py --model pose.caffemodel --proto pose.prototxt --dataset MPI
```
--model 参数和 --proto 参数分别是预先训练好的人体姿态模型和配置文件。因为模型文件很大，并不包括在 OpenCV 代码库中，可以在 Openpose 项目（https://github.com/CMU-Perceptual-Computing-Lab/openpose ）找到下载地址。

另外可以通过 --input 参数指定识别的图片或视频地址，默认则使用摄像头实时采集。