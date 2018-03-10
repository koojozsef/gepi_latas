#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/**
* @function main
*/
int main(int argc, char** argv)
{
	Mat src, dst;

	/// Load image
	src = imread(argv[1], 1);

	if (!src.data)
	{
		return -1;
	}

	///second feladat
	//blur the image with 2D convolution in several steps (4-5).
	
	Mat kernel;
	kernel = Mat::ones(3, 3, CV_32F) / (float)(3 * 3);
	
	//1-st step
	Mat blur1;
	namedWindow("1", CV_WINDOW_AUTOSIZE);
	filter2D(src, blur1, -1, kernel);
	imshow("1", blur1);

	//2-nd step
	Mat blur2;
	namedWindow("2", CV_WINDOW_AUTOSIZE);
	filter2D(blur1, blur2, -1, kernel);
	imshow("2", blur2);

	//3-rd step
	Mat blur3;
	namedWindow("3", CV_WINDOW_AUTOSIZE);
	filter2D(blur2, blur3, -1, kernel);
	imshow("3", blur3);

	//4-th step
	Mat blur4;
	namedWindow("4", CV_WINDOW_AUTOSIZE);
	filter2D(blur3, blur4, -1, kernel);
	imshow("4", blur4);

	///// Separate the image in 3 places ( B, G and R )
	//vector<Mat> bgr_planes;
	//split(src, bgr_planes);

	///// Establish the number of bins
	//int histSize = 256;

	///// Set the ranges ( for B,G,R) )

	//float range[] = { 0, 256 };
	//const float* histRange = { range };

	//bool uniform = true; bool accumulate = false;

	//Mat b_hist, g_hist, r_hist;

	///// Compute the histograms:
	//calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	//calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	//calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	//// Draw the histograms for B, G and R
	//int hist_w = 512; int hist_h = 400;
	//int bin_w = cvRound((double)hist_w / histSize);

	//Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	///// Normalize the result to [ 0, histImage.rows ]
	//normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	//normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	//normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	///// Draw for each channel
	//for (int i = 1; i < histSize; i++)
	//{
	//	line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
	//		Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
	//		Scalar(255, 0, 0), 2, 8, 0);
	//	line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
	//		Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
	//		Scalar(0, 255, 0), 2, 8, 0);
	//	line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
	//		Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
	//		Scalar(0, 0, 255), 2, 8, 0);
	//}

	///// Display
	//namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
	//imshow("calcHist Demo", histImage);

	//namedWindow("original", CV_WINDOW_AUTOSIZE);
	//imshow("original", src);


	///// 1. problem
	/////		a) bitwise_not
	//Mat bitimage;
	//bitwise_not(src, bitimage);

	//namedWindow("bitwise", CV_WINDOW_AUTOSIZE);
	//imshow("bitwise", bitimage);

	/////		b) subtract
	//Mat subimage;
	//subtract(255, src, subimage);

	/////		c) invert on pixels
	//Mat invimage;
	//invimage = src;
	//for (int i = 0; i < invimage.rows; ++i)
	//{
	//	for (int j = 0; j < invimage.cols; ++j)
	//	{
	//		invimage.data[i, j, 2] = 0;
	//	}
	//}

	///// 2. greyscale convertálás
	//Mat greyimage;
	//cvtColor(src, greyimage, CV_BGR2GRAY);

	//namedWindow("grey", CV_WINDOW_AUTOSIZE);
	//imshow("grey", greyimage);

	///// 3. hisztogram egyenesítés
	//Mat histedimage;
	//vector<Mat> channels;
	//split(src, channels);
	//equalizeHist(channels[0], channels[0]);
	//equalizeHist(channels[1], channels[1]);
	//equalizeHist(channels[2], channels[2]);

	//merge(channels, histedimage);


	/////Display own window
	//namedWindow("calcHist Demo2", CV_WINDOW_AUTOSIZE);
	//imshow("calcHist Demo2", histedimage);

	waitKey(0);

	return 0;
}