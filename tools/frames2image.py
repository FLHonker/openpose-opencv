### 用于pix2pix训练集制作的脚本
# 输入两个视频，逐帧读取，一对一拼接在一起，输出拼接好的图片集。

import cv2 as cv
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser(
    description='This is a script to make a dataset for pix2pix with 2 videos.'
)
parser.add_argument('--v1', required=True, help='Path to video1, generate the 1st image.')
parser.add_argument('--v2', required=True, help='Path to video2, generate the 2nd image.')
parser.add_argument('--outpath', default='frames2image', help='path to save the images.')
parser.add_argument('--prefix', default=''. help='prefix of image name.')
parser.add_argument('--count', type=int, default=-1, help='the number of images to gen.')
parser.add_argument('--width', type=int, default=450, help='the width of each frame to resize.')
parser.add_argument('--height', type=int, default=420, help='the height of each frame to resize.')

args = parser.parse_args()

outpath = args.outpath
count = args.count
target_w = args.width
target_h = args.height
cap1 = cv.VideoCapture(args.v1)
cap2 = cv.VideoCapture(args.v2)
# 视频尺寸
w1 = cap1.get(cv.CAP_PROP_FRAME_WIDTH)
h1 = cap1.get(cv.CAP_PROP_FRAME_HEIGHT)
w2 = cap2.get(cv.CAP_PROP_FRAME_WIDTH)
h2 = cap2.get(cv.CAP_PROP_FRAME_HEIGHT)
if w1 != w2 or h1 != h2:
    print('The shapes of the 2 videos are different!')
    print('But we can continue...')
# 获取两个视频的帧数
frameNum1 = int(cap1.get(cv.CAP_PROP_FRAME_COUNT))
frameNum2 = int(cap2.get(cv.CAP_PROP_FRAME_COUNT))
# 最大可生成图片数
max_count = frameNum1 if frameNum1 < frameNum2 else frameNum2
# 应生成图片数
image_count = max_count if count < 0 or count > max_count else count
# 计时开始
start_time = time.time()

# 逐帧读取与拼接图像
for i in range(image_count):
    _, frame1 = cap1.read()
    _, frame2 = cap2.read()
    # resize
    frame1 = cv.resize(frame1, (target_w, target_h), interpolation=cv.INTER_CUBIC)
    frame2 = cv.resize(frame2, (target_w, target_h), interpolation=cv.INTER_CUBIC)
    # 使用numpy将两张图片拼接
    image = np.concatenate([frame1, frame2], axis=1)
    cv.imwrite('%s/%s_%d.jpg'%(outpath, args.prefix, i), image)
    if i > 0 and i % 20 == 0:
        end_time = time.time()
        print('已生成{}/{}张图片, 用时{:.2f}s'.format(i, image_count, end_time - start_time))
end_time = time.time()
print('\n共生成{}张图片, 用时{:.2f}s'.format(image_count, end_time - start_time))

cap1.release()
cap2.release()
