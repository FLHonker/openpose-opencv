// C++ source file
/***********************************************
# Copyright (c) 2018, Wuhan & Guangzhou
# All rights reserved.
#
# @Filename: cv_pose.cpp
# @Version：V1.0
# @Author: Frank Liu - frankliu624@gmail.com
# @Description: Real-time 2D human pose estimation（实时2D人体姿态估计）
# @Create Time: 2018-12-19 22:02:18
# @Last Modified: 2018-12-27 18:59:25
***********************************************/
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

using namespace std;
using namespace cv;

// caffemodel模型文件
const string modelTxt = "../pose_deploy_linevec.prototxt";
const string modelBin = "../pose/coco/pose_iter_440000.caffemodel";

// 数据结构大小
const int n_BODY_PARTS = 19;
const int rows_POSE_PAIRS = 17, cols_POSE_PAIRS = 2;

// body's parts, 17 + 1(bkg)
enum BODY_PARTS
{
	Nose = 0,
	Neck = 1,
	RShoulder = 2,
	RElbow = 3,
	RWrist = 4,
	LShoulder = 5,
	LElbow = 6,
	LWrist = 7,
	RHip = 8,
	RKnee = 9,
	RAnkle = 10,
	LHip = 11,
	LKnee = 12,
	LAnkle = 13,
	REye = 14,
	LEye = 15,
	REar = 16,
	LEar = 17,
	Background = 18
};

// pose pairs, 17 x 2
const int POSE_PAIRS[rows_POSE_PAIRS][cols_POSE_PAIRS] = {
	{Neck, RShoulder}, {Neck, LShoulder}, {RShoulder, RElbow},
    {RElbow, RWrist}, {LShoulder, LElbow}, {LElbow, LWrist}, 
	{Neck, RHip}, {RHip, RKnee}, {RKnee, RAnkle}, {Neck, LHip}, 
	{LHip, LKnee}, {LKnee, LAnkle}, {Neck, Nose}, {Nose, REye},
	{REye, REar}, {Nose, LEye}, {LEye, LEar}
};

// visualize
const Scalar colors[] = {
	Scalar(255, 0, 0), Scalar(255, 85, 0), Scalar(255, 170, 0), Scalar(255, 255, 0), 
	Scalar(170, 255, 0), Scalar(85, 255, 0), Scalar(0, 255, 0), Scalar(0, 255, 85), 
	Scalar(0, 255, 170), Scalar(0, 255, 255), Scalar(0, 170, 255), Scalar(0, 85, 255), 
	Scalar(0, 0, 255), Scalar(85, 0, 255), Scalar(170, 0, 255), Scalar(255, 0, 255), 
	Scalar(255, 0, 170), Scalar(255, 0, 85)
};

/**
 * @brief 
 *
 * @param net
 * @param src
 * @param threshold
 * @param size
 *
 * @return 
 */
Mat openpose_image(dnn::Net net, Mat src, float threshold=0.1, Size size=Size(368,368))
{
	if(net.empty() || src.empty())
		return Mat();
	
	// 神经网络forward的输出的4D结果降维到3D所用newsize
	const int newsz[] = {57, 46, 46};	// 倒着存放，对应三维Mat的高、宽、长(z,x,y)

	// 读取图片并将其转换成openpose Net可以读取的blob
	Mat blob = dnn::blobFromImage(src, 1.0 / 255, size, Scalar(0, 0, 0), false, false);
	net.setInput(blob);
	Mat prob_4D = net.forward();	// 前向传播计算结果: 4-dims
	// cout << prob_4D.size << endl;	// 1 x 57 x 46 x 46
	Mat prob_3D = prob_4D.reshape(1, 3, newsz);	// cn = 1, newdims = 3
	// cout << prob_3D.size << endl;		// 57 x 46 x 46
	vector<Mat> probs_2D(newsz[0]);		// length = 57
	// 这一块工作就是将3D的Mat矩阵切片为57个二维Mat，效率有待改进！
	for(int z = 0; z < newsz[0]; z++)
	{
		Mat &prob = probs_2D[z];
		prob.create(Size(newsz[1], newsz[2]), CV_32F);	// 46 x 46
		
		for (int y = 0; y < newsz[1]; y++)
		{
			float *rowdata = prob_3D.ptr<float>(z,y);	// 行数据
			
			for (int x = 0; x < newsz[2]; x++)
			{
				prob.at<float>(y, x) = *rowdata++;
			}
		}
		// 至此，一个二维的prob构造完成
		// cout << prob << endl;
	}
	/* 开始绘制结果图 */
	Mat poseFrame(src.rows, src.cols, CV_8UC3, Scalar(0,0,0));	// 绘图矩阵
	vector<Point> points; // 关键点集
	// detect body's part
	for (int j = 0; j < n_BODY_PARTS; ++j)
	{
		Point point;			// 关键点
		double confidence;		// 预测概率
		// Originally, we try to find all the local maximums. To simplify a sample,
		// we just find a global one. However only a single pose at the same time
		// could be detected this way.
		minMaxLoc(probs_2D[j], 0, &confidence, 0, &point);
		// cout << point << endl;
		// Add a point if it's confidence is higher than threshold.
		if(confidence > threshold)
		{
			int x = (src.cols * point.x) / 46;
			int y = (src.rows * point.y) / 46;
			points.push_back(Point(x, y));
		}else{
			points.push_back(Point(-1,-1));
		}
	}

	// plot lines to link each part
	for (int k = 0; k < rows_POSE_PAIRS; ++k)
	{
		int pairFrom = POSE_PAIRS[k][0];	// 起点
		int pairTo = POSE_PAIRS[k][1];		// 终点
		if(points[pairFrom].x > 0 && points[pairTo].x > 0)
		{
			line(poseFrame, points[pairFrom], points[pairTo], colors[k], 3);
			cv::circle(poseFrame, points[pairFrom], 3, (0, 0, 255), 2);
			cv::circle(poseFrame, points[pairTo], 3, (0, 0, 255), 2);
		}
	}
	// imshow("openpose", poseFrame);
	return poseFrame;
}

/**
 * @brief 
 *
 * @param useCam
 * @param inVideo
 * @param outVideo
 * @param frameNum
 * @param fps
 * @param threshold
 *
 * @return 
 */
int openpose_video(bool useCam, string inVideo, string outVideo, int frameNum=-1, float fps=20.0, float threshold=0.1)
{
	// 加载视频, 是否使用摄像头(0 or 1)
	VideoCapture cap;
	if(useCam)
		cap.open(0);
	else if(!inVideo.empty())
		cap.open(inVideo);
	else{
		cout << "can't load video from camera or file!" << endl;
		return -1;
	}
	if (!cap.isOpened())
	{
		cout << "open video failed!" << endl;
		return -1;
	}
	
	// 使用cv的DNN模块加载caffe网络模型,读取.protxt文件和.caffemodel文件
    dnn::Net net = dnn::readNetFromCaffe(modelTxt, modelBin);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);
	// 检查网络是否读取成功
	if (net.empty())
	{
		std::cerr << "Can't load network by using the following files:"  << std::endl;
		std::cerr << "prototxt:   " << modelTxt << std::endl;
		std::cerr << "caffemodel: " << modelBin << std::endl;
		return -1;
	}else{
		std::cout << "pose.caffemodel was loaded successfully!" << std::endl;
	}
	
	// 视频总帧数
	int frameCount = cap.get(cv::CAP_PROP_FRAME_COUNT);
	// 需要处理的帧数
	if (frameCount < frameNum || frameNum < 0)
		frameNum = frameCount;

	//获得帧率
	double rate = cap.get(cv::CAP_PROP_FPS);
	cout << "The FPS of source video is: " << rate << endl;
	// 保存size必须和输出size设定为一致，否则无法写入保存文件
	int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
	int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
	Size size(w, h);
	// vedio writer
	VideoWriter poseout(outVideo, VideoWriter::fourcc('X','V','I','D'), fps, size);
	Mat frame; 	// 视频帧

	cout << "Starting..." << endl;
	cout << "共" << frameNum << "帧图像， 预计耗时" << frameNum * 4.8 << "s." << endl;
	
	//记录起始时间，其中getTickCount()函数返回CPU 自某个事件（如启动电脑）以来走过的时钟周期数
	double start_time = static_cast<double>(getTickCount());

	// 逐帧读取图像进行人体姿态估计
	int i = 1;
	for (; i <= frameNum; ++i)
	{
		cap >> frame;
		Mat poseFrame = openpose_image(net, frame, threshold);
		imshow("openpose", poseFrame);
		poseout.write(poseFrame);
		waitKey(10);

		// 结束计时
		if (i % 20 == 0)
		{
			double end_time = static_cast<double>(getTickCount());
			double spend_time = (end_time - start_time) / getTickFrequency();
			cout << "已处理" << i << "/" << frameNum << "帧图像， 用时" << spend_time << "s, 平均每帧用时" << spend_time / i << endl;
		}
	}

	// 结束计时
	double end_time = static_cast<double>(getTickCount());
	double spend_time = (end_time - start_time) / getTickFrequency();
	cout << "共处理[" << --i << "]帧图像， 用时" << spend_time << "s, 平均每帧用时" << spend_time / i << "s." << endl;

	cap.release();
	poseout.release();
	destroyAllWindows();
	net.~Net();	// 销毁
	return i;
}

// Test video
int main()
{
	string inVideo = "../mv_20s.mp4.avi";
	string outVideo = "../cv_outpose.avi";
	openpose_video(false, inVideo, outVideo, 40, 20);
	
	return 0;
}
