# openpose-python提供的外部调用接口，支持单张图片的pose估计。
import cv2 as cv
import numpy as np


class OpenPose():

    def __init__(self, proto='./pose_deploy_linevec.prototxt', \
                 model='./pose/coco/pose_iter_440000.caffemodel', \
                 dataset='COCO', width=368, height=368, threshold=0.1):

        if dataset == 'COCO':
            self.BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                                "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

            self.POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                                ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                                ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                                ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                                ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
        else:
            assert(dataset == 'MPI')
            self.BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                                "Background": 15 }

            self.POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]

        # visualize
        self.colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
                       [0, 255, 0], \
                       [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
                       [85, 0, 255], \
                       [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

        ### configs
        # load caffemodel
        self.net = cv.dnn.readNetFromCaffe(proto, model)
        self.threshold = threshold
        self.scale_size = (width, height)

    ### 单张图片进行人体姿态估计
    # @input: cv image
    # @return cv image
    def openpose_image(self, img): 
        w = img.shape[1]
        h = img.shape[0]
        # 创建空的pose绘制图
        poseFrame = np.zeros((h, w, 3), np.uint8)
        poseFrame.fill(0)

        inp = cv.dnn.blobFromImage(img, 1.0 / 255, self.scale_size,
                                (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(inp)
        out = self.net.forward()

        # assert(len(BODY_PARTS) == out.shape[1])

        # keypoints of body
        points = []
        for i in range(len(self.BODY_PARTS)):
            # Slice heatmap of corresponging body's part. - 4D
            heatMap = out[0, i,:,:]
            
            # Originally, we try to find all the local maximums. To simplify a sample,
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv.minMaxLoc(heatMap)
            print(point)
            x = (w * point[0]) / out.shape[3]
            y = (h * point[1]) / out.shape[2]

            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > self.threshold else None)
        # *************** points ***************
        # print(points)

        for i, pair in enumerate(self.POSE_PAIRS):
            partFrom = pair[0]
            partTo = pair[1]
            # assert(partFrom in BODY_PARTS)
            # assert(partTo in BODY_PARTS)

            idFrom = self.BODY_PARTS[partFrom]
            idTo = self.BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv.line(poseFrame, points[idFrom], points[idTo], self.colors[i], 3)
                cv.ellipse(poseFrame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(poseFrame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
        # return `pose stick figure`
        return poseFrame

# test
if __name__ == '__main__':
    poseDetector = OpenPose()
    img = cv.imread('pbug.png')
    pose = poseDetector.openpose_image(img)
    cv.imshow('pose', pose)
    cv.waitKey(0)