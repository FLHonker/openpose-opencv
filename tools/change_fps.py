# 修改视频的FPS
import cv2 as cv
import argparse
import time

parser = argparse.ArgumentParser(
    description='This is a script to change the FPS of a video.')
parser.add_argument('--input', required=True, help='Path to video to modify.')
parser.add_argument('--fps', type=float, default=20.0, help='The target FPS you need.')
parser.add_argument('--output', default='output_20fps.avi', help='Path to output video.')

args = parser.parse_args()

cap = cv.VideoCapture(args.input)
# vedio writer
# fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
fourcc = cv.VideoWriter_fourcc(* 'XVID')
# 保存size必须和输出size设定为一致，否则无法写入保存文件
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
size = (w, h)
out = cv.VideoWriter(args.output, fourcc, args.fps, size)

print('开始转换...')
start_time = time.time()
i = 0
while True:
    hasFrame, frame = cap.read()
    if not hasFrame:
        break
    out.write(frame)
    i += 1

end_time = time.time()
cap.release()
out.release()
print('共读写{}帧，转换时间为{:.2f}s'.format(i, end_time - start_time))

