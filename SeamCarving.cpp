/**
 * Seam Carving Project
 * Note*
 * Seam insertion - this is process is quite complicated because of best seam computation
 * If insert seam normally and then use seam inserted image to compute M & K matrix.
 * The result will be the same seam as a previous one.
 * To fix these problem. A duplicated image is require.
 * Duplicated image get a seam removal, while original image get a seam insertion,
 * and use duplicated image to compute M & K matrix instead of seam inserted one.
 * So, the result will be a new seam.
 *
 * Horizontal Seam insertion/removal - counter clockwise rotate image 90 degree then insert/remove seam.
 * */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>

#define LEFT 0
#define RIGHT 2

using namespace cv;
using namespace std;

/**
 * Compute M matrix and K matrix from Image (Forward)
 * @param img -image that use to compute M and K matrix
 * @param OutMatrixM -destination matrix that use to store M matrix
 * @param OutMatrixK -destination matrix that use to store K matrix
 */
void computeMKmatrix(Mat img, Mat &OutMatrixM, Mat &OutMatrixK) {

	Mat gray;

	cvtColor(img, gray, COLOR_BGR2GRAY); //Convert image to greyscale

	Size sz = gray.size(); //Get width and height of image

	copyMakeBorder(gray, gray, 1, 1, 1, 1, BORDER_REPLICATE); //Padding greyscale image with duplicate
	gray.convertTo(gray, CV_32S); //Convert matrix type from unsigned char to int (prevent overflow from addition)

	Mat mMatrix(sz, CV_32S), kMat(sz, CV_8U); //M matrix and K matrix

	for (int i = 0; i < mMatrix.rows; ++i) {
		for (int j = 0; j < mMatrix.cols; ++j) {

			//Calculate X and Y position when access pixel in greyscale image
			//because greyscale image got padding with duplicate.
			int gXpos = j + 1;
			int gYpos = i + 1;

			int cu = abs(gray.at<int>(gYpos, gXpos - 1) - gray.at<int>(gYpos, gXpos + 1));      //Calculate CU
			int cl = cu + abs(gray.at<int>(gYpos - 1, gXpos) - gray.at<int>(gYpos, gXpos - 1)); //Calculate CL
			int cr = cu + abs(gray.at<int>(gYpos - 1, gXpos) - gray.at<int>(gYpos, gXpos + 1)); //Calculate CR

			if (i == 0) {                   //Top row
				mMatrix.at<int>(i, j) = cu; //set M matrix value to CU
				kMat.at<uchar>(i, j) = 0;   //set K matrix value to 0
				continue;                   //Skip to next pixel
			}

			int cArr[] = { cl,cu,cr };  //Create array of CL, CU and CR
			int mArr[3];                //Create array of ML, MU and MR

			mArr[1] = mMatrix.at<int>(i - 1, j); //MU = M(i-1,j)

			if (j == 0) {                                   //First column of image
				mArr[0] = 9999999;                          //Set ML to very high value to prevent it from being chosen.
				mArr[2] = mMatrix.at<int>(i - 1, j + 1);    //MR = M(i-1,j+1)
			}
			else if (j == mMatrix.cols - 1) {               //Last column of image
				mArr[0] = mMatrix.at<int>(i - 1, j - 1);    //ML = M(i-1,j-1)
				mArr[2] = 9999999;                          //Set MR to very high value to prevent it from being chosen.
			}
			else {
				mArr[0] = mMatrix.at<int>(i - 1, j - 1);    //ML = M(i-1,j-1)
				mArr[2] = mMatrix.at<int>(i - 1, j + 1);    //MR = M(i-1,j+1)
			}

			//Adding two array (ML,MU,MR + CL,CU,CR) together using std::transform function
			transform(begin(cArr), end(cArr), begin(mArr), begin(mArr), plus<int>());

			int *minVal = min_element(begin(mArr), end(mArr));      //Find lowest energy value between ML,MU,MR
			mMatrix.at<int>(i, j) = *minVal;                        //M(i,j) = lowest energy value
			kMat.at<uchar>(i, j) = (uchar)distance(mArr, minVal);   //K(i,j) = Direction of lowest energy value (0 - Up-left, 1 - Up, 2 - Up-right)
																	//std::distance use to find index of lowest energy value from the array.
		}
	}
	gray.release();

	Mat enNorm;

	normalize(mMatrix, enNorm, 255, 0, NORM_MINMAX);    //Normalize value range to 0-255
	convertScaleAbs(enNorm, OutMatrixM);                //Convert matrix type from CV_32S to CV_8U

	OutMatrixK = kMat;
}

/**
 * Find the best seam (seam with the lowest energy) from M and K matrix
 * @param matrixM -M matrix that use to find the best seam
 * @param matrixK -K matrix that use to find the best seam
 * @param bestSeamPath -destination vector<int> that use to store best seam
 *        (vector store x position of each row from bottom to top)
 */
void findBestSeam(Mat matrixM, Mat matrixK, vector<int> &bestSeamPath) {
	vector<int> path;
	Point minLoc;
	Mat lastRow = matrixM.row(matrixM.rows - 1);    //Get last row of image.

	//Find X-position of lowest energy value
	minMaxLoc(lastRow, NULL, NULL, &minLoc, NULL);
	int x_pos = minLoc.x;

	path.push_back(x_pos);                          //Push X-position to vector.

	//Loop from bottom to top of image
	for (int i = matrixK.rows - 1; i > 0; i--) {
		uchar value = matrixK.at<uchar>(i, x_pos);

		if (value == LEFT) x_pos--;                 //If K(i,j) = 0, X-position - 1
		else if (value == RIGHT) x_pos++;           //If K(i,j) = 2, X-position + 1

		path.push_back(x_pos);                      //Push X-position to vector.
	}

	bestSeamPath = path;
}

/**
 * Insert Seam at a given path by adding new pixels to right side of best seam
 * (new pixels is compute from average of left and right pixel)
 * @param img -img that want to insert seam
 * @param bestSeamPath -vector of best seam position
 * @param imgOut -destination matrix that use to store inserted seam image
 */
void insertSeam(Mat img, vector<int> bestSeamPath, Mat &imgOut) {
	Mat newImg(img.rows, img.cols + 1, CV_8UC3);
	vector<int>::reverse_iterator it = bestSeamPath.rbegin();   //Read vector from back to front.

	//Loop through each row of image
	for (int i = 0; i < img.rows; i++) {
		Mat left, right, row;
		Mat pixel(1, 1, CV_8UC3);
		img.row(i).colRange(0, it[i] + 1).copyTo(left);         //Copy pixels[0, best_seam_position] to left matrix.
		img.row(i).colRange(it[i] + 1, img.cols).copyTo(right); //Copy pixels[best_seam_position+1, last pixel in that row] to right matrix.

		if (right.empty()) {                                            //If there is no pixels at right side of image.
			pixel.at<Vec3b>(0, 0) = left.at<Vec3b>(0, left.cols - 1);   //New pixel = pixel at best seam position.
			hconcat(left, pixel, row);                                  //Combine left Mat and new pixel to a one row matrix.
		}
		else {
			//Calculate value of a new pixel
			uchar red = (left.at<Vec3b>(0, left.cols - 1)[2] + right.at<Vec3b>(0, 0)[2]) / 2;
			uchar green = (left.at<Vec3b>(0, left.cols - 1)[1] + right.at<Vec3b>(0, 0)[1]) / 2;
			uchar blue = (left.at<Vec3b>(0, left.cols - 1)[0] + right.at<Vec3b>(0, 0)[0]) / 2;

			pixel.at<Vec3b>(0, 0) = Vec3b(blue, green, red);

			vector<Mat> matrices = { left, pixel, right };
			hconcat(matrices, row);                                     //Combine left mat, a new pixel and right mat to a one row matrix.
		}
		row.copyTo(newImg.row(i));                                      //Replace image at i row wtih a new combined row.
	}

	imgOut = newImg;
}

/**
 * Remove Seam at a given path by copying pixel in each row
 * exclude a pixel at best seam position to new image
 * @param img -img that want to remove seam
 * @param bestSeamPath -vector of best seam position
 * @param imgOut -destination matrix that use to store removed seam image
 */
void removeSeam(Mat img, vector<int> bestSeamPath, Mat &imgOut) {
	Mat newImg(img.rows, img.cols - 1, CV_8UC3);
	vector<int>::reverse_iterator it = bestSeamPath.rbegin();       //Read vector from back to front.

	for (int i = 0; i < img.rows; i++) {
		Mat left, right, row;
		img.row(i).colRange(0, it[i]).copyTo(left);                 //Copy pixels[0,best_seam_position) to left matrix.
		img.row(i).colRange(it[i] + 1, img.cols).copyTo(right);     //Copy pixels(best_seam_position, last pixel in that row] to right matrix.

		if (left.empty()) {                                         //If left matrix is empty.
			right.copyTo(newImg.row(i));                            //Replace image at i row with right matrix.
																	//because if left matrix is empty, it mean that right matrix contain all pixel
																	//in that row except a pixel at best seam position.
		}
		else if (right.empty()) {                                   //if right matrix is empty.
			left.copyTo(newImg.row(i));                             //Replace image at i row with left matrix.
																	//because if right matrix is empty, it mean that left matrix contain all pixel
																	//in that row except a pixel at best seam position.
		}
		else if (left.empty() && right.empty()) {                   //if both left & right matrix is empty, skip.
			continue;
		}
		else {
			hconcat(left, right, row);                              //Combine left & right matrix together,
																	//So row matrix contain all pixel in that row except a pixel at best seam position
			row.copyTo(newImg.row(i));                              //Replace image at i row with a row matrix
		}
	}

	imgOut = newImg;
}

/**
 * Draw a line according to a given path
 * @param img -img that want to draw a line on
 * @param bestSeamPath -vector of best seam position
 * @param imgOut -destination matrix that use to store new image
 */
void drawVerticalPath(Mat img, vector<int> bestSeamPath, Mat &imgOut) {
	Mat newImg = img.clone();
	vector<int>::reverse_iterator it = bestSeamPath.rbegin();       //Read vector from back to front

	for (int i = 0; i < img.rows; i++) {
		int x_pos = it[i];
		newImg.at<Vec3b>(i, x_pos) = Vec3b(0, 0, 255);              //Set color at best seam position to red
	}

	imgOut = newImg;
}


/**
 * Main Function
 */
int main() {
	Mat img = imread("img\\rain6.jpg");                             //Read image
	namedWindow("Image", WINDOW_AUTOSIZE);                          //Create window
	imshow("Image", img);                                           //Show image
	Mat imgDul = img.clone();                                       //Clone image for seam insertion
	int c = waitKey(0);
	namedWindow("Energy", WINDOW_AUTOSIZE);                         //Create window
	namedWindow("Seam", WINDOW_AUTOSIZE);                           //Create window

	//Esc = 27, a = 97, d = 97, s = 115, w = 119
	//Press Esc to exit loop and end program
	while (c != 27) {

		Mat matrixM, matrixK, newImg, newImgI, pathImg;
		vector<int> path;

		//Vertical direction
		//a -> remove seam (reduce image width), d -> insert seam (increase image width)
		if (c == 97 || c == 100) {

			if (c == 97) {
				computeMKmatrix(img, matrixM, matrixK);             //Compute M & K matrix
				findBestSeam(matrixM, matrixK, path);               //Find best seam from M & K matrix
				drawVerticalPath(img, path, pathImg);               //Draw a best seam line on image

				imshow("Energy", matrixM);                          //Show image
				imshow("Seam", pathImg);                            //Show image

				removeSeam(img, path, newImg);                      //Remove seam from image
				imshow("Image", newImg);                            //Show image

				imgDul = newImg;                                    //Set duplicated image to seam removed image
			}
			else {
				computeMKmatrix(imgDul, matrixM, matrixK);          //Compute M & K matrix from duplicated image
				findBestSeam(matrixM, matrixK, path);               //Find best seam from M & K matrix
				removeSeam(imgDul, path, imgDul);                   //Remove seam from duplicated image

				computeMKmatrix(img, matrixM, matrixK);             //Compute M & K matrix (this is for visual only)
				drawVerticalPath(img, path, pathImg);               //Draw a best seam line on image

				imshow("Energy", matrixM);                          //Show image of M matrix
				imshow("Seam", pathImg);                            //Show image of best seam

				insertSeam(img, path, newImg);                      //Insert seam to image
				imshow("Image", newImg);                            //Show image

			}

			img = newImg;
		}

		//Horizontal Direction
		//s -> remove seam (reduce image height), w -> insert seam (increase image height)
		if (c == 115 || c == 119) {									//Horizontal Seam insertion/removal

			Mat rotateImg;
			rotate(img, rotateImg, ROTATE_90_COUNTERCLOCKWISE);     //Rotate image

			if (c == 115) {

				computeMKmatrix(rotateImg, matrixM, matrixK);		//Compute M & K matrix
				findBestSeam(matrixM, matrixK, path);				//Find best seam from M & K matrix
				drawVerticalPath(rotateImg, path, pathImg);			//Draw a best seam line on image

				rotate(matrixM, matrixM, ROTATE_90_CLOCKWISE);		//Rotate M matrix (this is for visual)
				rotate(pathImg, pathImg, ROTATE_90_CLOCKWISE);		//Rotate best seam line image

				imshow("Energy", matrixM);							//Show image of M matrix
				imshow("Seam", pathImg);							//Show image of best seam

				removeSeam(rotateImg, path, newImg);				//Remove seam from image

				rotate(newImg, newImg, ROTATE_90_CLOCKWISE);		//Rotate it back

				imshow("Image", newImg);							//Show image

				imgDul = newImg;
			}
			else if (c == 119) {
				Mat rotateImgDul;
				rotate(imgDul, rotateImgDul, ROTATE_90_COUNTERCLOCKWISE);

				computeMKmatrix(rotateImgDul, matrixM, matrixK);	//Compute M & K matrix from duplicated image
				findBestSeam(matrixM, matrixK, path);				//Find best seam from M & K matrix
				removeSeam(rotateImgDul, path, rotateImgDul);		//Remove seam from duplicated image

				rotate(rotateImgDul, imgDul, ROTATE_90_CLOCKWISE);

				computeMKmatrix(rotateImg, matrixM, matrixK);		//Compute M & K matrix (this is for visual only)
				drawVerticalPath(rotateImg, path, pathImg);			//Draw a best seam line on image

				rotate(matrixM, matrixM, ROTATE_90_CLOCKWISE);
				rotate(pathImg, pathImg, ROTATE_90_CLOCKWISE);

				imshow("Energy", matrixM);							//Show image of M matrix
				imshow("Seam", pathImg);							//Show image of best seam

				insertSeam(rotateImg, path, newImg);				//Insert seam to image
				rotate(newImg, newImg, ROTATE_90_CLOCKWISE);		

				imshow("Image", newImg);							//Show image
			}

			img = newImg;
		}
		c = waitKey(0);
	}



	return 0;
}