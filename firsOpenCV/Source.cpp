#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;

void printSharpness(Mat img, const char* name);
void third_problem(Mat src1, Mat src2, Mat src3, Mat src4);
void fourth_problem(Mat src1, Mat &output);
double getPSNR(const Mat& I1, const Mat& I2);
float calcBlurriness(const Mat &src)
{
	Mat Gx, Gy;
	Sobel(src, Gx, CV_32F, 1, 0);
	Sobel(src, Gy, CV_32F, 0, 1);
	double normGx = norm(Gx);
	double normGy = norm(Gy);
	double sumSq = normGx * normGx + normGy * normGy;
	return static_cast<float>(1. / (sumSq / src.size().area() + 1e-6));
}
/**
* @function main
*/
int main(int argc, char** argv)
{
	Mat src, dst, src2, src3, src4;

	/// Load image
	src = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	/*src2 = imread(argv[2], 1);
	src3 = imread(argv[3], 1);
	src4 = imread(argv[4], 1);*/

	if (!src.data /*|| !src2.data || !src3.data || !src4.data*/)
	{
		printf("error");
		return -1;
	}

	//third_problem(src, src2, src3, src4);
	fourth_problem(src, dst);



	/////second feladat
	////blur the image with 2D convolution in several steps (4-5).
	//
	//Mat kernel;
	//kernel = Mat::ones(3, 3, CV_32F) / (float)(3 * 3);
	//
	////1-st step
	//Mat blur1;
	//namedWindow("1", CV_WINDOW_AUTOSIZE);
	//filter2D(src, blur1, -1, kernel);
	//imshow("1", blur1);
	//printSharpness(blur1, "1");
	////2-nd step
	//Mat blur2;
	//namedWindow("2", CV_WINDOW_AUTOSIZE);
	//filter2D(blur1, blur2, -1, kernel);
	//imshow("2", blur2);
	//printSharpness(blur2, "2");

	////3-rd step
	//Mat blur3;
	//namedWindow("3", CV_WINDOW_AUTOSIZE);
	//filter2D(blur2, blur3, -1, kernel);
	//imshow("3", blur3);
	//printSharpness(blur3, "3");

	////4-th step
	//Mat blur4;
	//namedWindow("4", CV_WINDOW_AUTOSIZE);
	//filter2D(blur3, blur4, -1, kernel);
	//imshow("4", blur4);

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

	namedWindow("original", CV_WINDOW_AUTOSIZE);
	imshow("original", src);
	imshow("dest", dst);


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

void printSharpness(Mat img, const char * name)
{
	float sharpnessValue = 0;
	sharpnessValue = 1/calcBlurriness(img);
	printf("%f, %c\n", sharpnessValue, *name);
}

void third_problem(Mat src1, Mat src2, Mat src3, Mat src4) {

	double val;
	val = getPSNR(src1, src1);
	printf("origin:\t%f \n", val);
	val = getPSNR(src1, src2);
	printf("80:\t%f \n", val);
	val = getPSNR(src1, src3);
	printf("50:\t%f \n", val);
	val = getPSNR(src1, src4);
	printf("20:\t%f \n", val);
}

double getPSNR(const Mat& I1, const Mat& I2)
{
	Mat s1;
	absdiff(I1, I2, s1);       // |I1 - I2|
	s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
	s1 = s1.mul(s1);           // |I1 - I2|^2

	Scalar s = sum(s1);        // sum elements per channel

	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

	if (sse <= 1e-10) // for small values return zero
		return 0;
	else
	{
		double mse = sse / (double)(I1.channels() * I1.total());
		double psnr = 10.0 * log10((255 * 255) / mse);
		return psnr;
	}
}

void fourth_problem(Mat I, Mat &output) {
	Mat padded;                            //expand input image to optimal size
	int m = getOptimalDFTSize(I.rows);
	int n = getOptimalDFTSize(I.cols); // on the border add zero values
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

	dft(complexI, complexI);            // this way the result may fit in the source matrix

										// compute the magnitude and switch to logarithmic scale
										// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	Mat magI = planes[0];

	magI += Scalar::all(1);                    // switch to logarithmic scale
	log(magI, magI);

	// crop the spectrum, if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
											// viewable image form (float between values 0 and 1).

	//imshow("Input Image", I);    // Show the result
	//imshow("spectrum magnitude", magI);
	output = magI;
	waitKey();
}