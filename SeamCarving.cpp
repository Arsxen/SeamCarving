#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>

#define LEFT 0
#define UP 1
#define RIGHT 2

using namespace cv;
using namespace std;

void computeMKmatrix(Mat img, Mat &OutMatrixM, Mat &OutMatrixK) {
	Mat imgInt, gray;

	cvtColor(img, gray, COLOR_BGR2GRAY);

	Size sz = gray.size();

	copyMakeBorder(gray, gray, 1, 1, 1, 1, BORDER_REPLICATE);
	gray.convertTo(gray, CV_32S);

	Mat energyMat(sz, CV_32S), kMat(sz, CV_8U);

	for (int i = 0; i < energyMat.rows; ++i) {
		for (int j = 0; j < energyMat.cols; ++j) {

			int gXpos = j + 1;
			int gYpos = i + 1;

			int cu = abs(gray.at<int>(gYpos, gXpos - 1) - gray.at<int>(gYpos, gXpos + 1));
			int cl = cu + abs(gray.at<int>(gYpos - 1, gXpos) - gray.at<int>(gYpos, gXpos - 1));
			int cr = cu + abs(gray.at<int>(gYpos - 1, gXpos) - gray.at<int>(gYpos, gXpos + 1));

			if (i == 0) {
				energyMat.at<int>(i, j) = cu;
				kMat.at<uchar>(i, j) = 0;
				continue;
			}

			int cArr[] = { cl,cu,cr };
			int mArr[3];

			mArr[1] = energyMat.at<int>(i - 1, j);

			if (j == 0) {
				mArr[0] = 9999999;
				mArr[2] = energyMat.at<int>(i - 1, j + 1);
			}
			else if (j == energyMat.cols - 1) {
				mArr[0] = energyMat.at<int>(i - 1, j - 1);
				mArr[2] = 9999999;
			}
			else {
				mArr[0] = energyMat.at<int>(i - 1, j - 1);
				mArr[2] = energyMat.at<int>(i - 1, j + 1);
			}

			transform(begin(cArr), end(cArr), begin(mArr), begin(mArr), plus<int>());

			int *minVal = min_element(begin(mArr), end(mArr));
			energyMat.at<int>(i, j) = *minVal;
			kMat.at<uchar>(i, j) = (uchar)distance(mArr, minVal);

		}
	}
	gray.release();

	Mat enNorm;

	normalize(energyMat, enNorm, 255, 0, NORM_MINMAX);
	convertScaleAbs(enNorm, OutMatrixM);

	OutMatrixK = kMat;
}

bool comparePair(const pair<uchar, int> &i, const pair<uchar, int> &j) {
	return i.first < j.first;
}

void findBestSeam(Mat matrixM, Mat matrixK, vector<int> &bestSeamPath) {
	vector<int> path;
	Mat lastRow = matrixM.row(matrixM.rows - 1);
	Point minLoc;

	minMaxLoc(lastRow, NULL, NULL, &minLoc, NULL);

	int x_pos = minLoc.x;

	path.push_back(x_pos);

	for (int i = matrixK.rows - 1; i > 0; i--) {
		uchar value = matrixK.at<uchar>(i, x_pos);

		if (value == LEFT) x_pos--;
		else if (value == RIGHT) x_pos++;

		path.push_back(x_pos);
	}

	bestSeamPath = path;
}

void insertSeam(Mat img, vector<int> bestSeamPath, Mat &imgOut) {
	Mat newImg(img.rows, img.cols + 1, CV_8UC3);
	vector<int>::reverse_iterator it = bestSeamPath.rbegin();

	for (int i = 0; i < img.rows; i++) {
		Mat left, right, row;
		Mat pixel(1, 1, CV_8UC3);
		img.row(i).colRange(0, it[i] + 1).copyTo(left);
		img.row(i).colRange(it[i] + 1, img.cols).copyTo(right);
		if (right.empty()) {
			pixel.at<Vec3b>(0, 0) = left.at<Vec3b>(0, left.cols - 1);
			hconcat(left, pixel, row);
		}
		else {
			uchar red = (left.at<Vec3b>(0, left.cols-1)[2] + right.at<Vec3b>(0,0)[2]) / 2 ;
			uchar green = (left.at<Vec3b>(0, left.cols-1)[1] + right.at<Vec3b>(0,0)[1]) / 2 ;
			uchar blue = (left.at<Vec3b>(0, left.cols-1)[0] + right.at<Vec3b>(0,0)[0]) / 2 ;
			pixel.at<Vec3b>(0, 0) = Vec3b(blue, green, red);
			vector<Mat> matrices = { left, pixel, right };
			hconcat(matrices, row);
		}
		row.copyTo(newImg.row(i));
	}

	imgOut = newImg;
}

void removeSeam(Mat img, vector<int> bestSeamPath, Mat &imgOut) {
	Mat newImg(img.rows, img.cols - 1, CV_8UC3);
	vector<int>::reverse_iterator it = bestSeamPath.rbegin();

	for (int i = 0; i < img.rows; i++) {
		Mat left, right, row;
		img.row(i).colRange(0, it[i]).copyTo(left);
		img.row(i).colRange(it[i] + 1, img.cols).copyTo(right);
		if (left.empty()) {
			right.copyTo(newImg.row(i));
		}
		else if (right.empty()) {
			left.copyTo(newImg.row(i));
		}
		else if (left.empty() && right.empty()) {
			continue;
		}
		else {
			hconcat(left, right, row);
			row.copyTo(newImg.row(i));
		}
	}

	imgOut = newImg;
}

void drawVerticalPath(Mat img, vector<int> bestSeamPath, Mat &imgOut) {
	Mat newImg = img.clone();
	vector<int>::reverse_iterator it = bestSeamPath.rbegin();

	for (int i = 0; i < img.rows; i++) {
		int x_pos = it[i];
		newImg.at<Vec3b>(i, x_pos) = Vec3b(0, 0, 255);
	}

	imgOut = newImg;
}



int main() {
	Mat img = imread("img\\rain6.jpg");
	namedWindow("Image", WINDOW_AUTOSIZE);
	imshow("Image", img);
	Mat imgDul = img.clone();
	int c = waitKey(0);
	namedWindow("Energy", WINDOW_AUTOSIZE);
	namedWindow("Seam", WINDOW_AUTOSIZE);
	while (c != 27) {

		Mat matrixM, matrixK, newImg, newImgI, pathImg;
		vector<int> path;

		if (c == 97 || c == 100) {

			if (c == 97) {
				computeMKmatrix(img, matrixM, matrixK);
				findBestSeam(matrixM, matrixK, path);
				drawVerticalPath(img, path, pathImg);

				imshow("Energy", matrixM);
				imshow("Seam", pathImg);

				removeSeam(img, path, newImg);
				imshow("Image", newImg);
				imgDul = newImg;
			}
			else {
				computeMKmatrix(imgDul, matrixM, matrixK);
				findBestSeam(matrixM, matrixK, path);
				removeSeam(imgDul, path, imgDul);

				computeMKmatrix(img, matrixM, matrixK);
				drawVerticalPath(img , path, pathImg);

				imshow("Energy", matrixM);
				imshow("Seam", pathImg);

				insertSeam(img , path, newImg);
				imshow("Image", newImg);
				
			}

			img = newImg;
		}

		if (c == 115 || c == 119) {

			Mat rotateImg;
			rotate(img, rotateImg, ROTATE_90_COUNTERCLOCKWISE);

			if (c == 115) {

				computeMKmatrix(rotateImg, matrixM, matrixK);
				findBestSeam(matrixM, matrixK, path);
				drawVerticalPath(rotateImg, path, pathImg);

				rotate(matrixM, matrixM, ROTATE_90_CLOCKWISE);
				rotate(pathImg, pathImg, ROTATE_90_CLOCKWISE);

				imshow("Energy", matrixM);
				imshow("Seam", pathImg);

				removeSeam(rotateImg, path, newImg);

				rotate(newImg, newImg, ROTATE_90_CLOCKWISE);

				imshow("Image", newImg);

				imgDul = newImg;
			}
			else if (c == 119) {
				Mat rotateImgDul;
				rotate(imgDul, rotateImgDul, ROTATE_90_COUNTERCLOCKWISE);
				computeMKmatrix(rotateImgDul, matrixM, matrixK);
				findBestSeam(matrixM, matrixK, path);
				removeSeam(rotateImgDul, path, rotateImgDul);
				rotate(rotateImgDul, imgDul, ROTATE_90_CLOCKWISE);

				computeMKmatrix(rotateImg, matrixM, matrixK);
				drawVerticalPath(rotateImg, path, pathImg);

				rotate(matrixM, matrixM, ROTATE_90_CLOCKWISE);
				rotate(pathImg, pathImg, ROTATE_90_CLOCKWISE);

				imshow("Energy", matrixM);
				imshow("Seam", pathImg);

				insertSeam(rotateImg, path, newImg);
				rotate(newImg, newImg, ROTATE_90_CLOCKWISE);

				imshow("Image", newImg);
			}

			img = newImg;
		}
		c = waitKey(0);
	}



	return 0;
}