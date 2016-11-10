#include <iostream>
#include <vector>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

static const int src_img_rows = 700;
static const int src_img_cols = 800;

static const double R = 1;
static const double G = 1;
static const double B = 0;

static const int THRESH = 120; //しきい値

using namespace cv;
using namespace std;

void onTrackbarChanged(int thres, void*);
Point2i calculate_center(Mat);
void getCoordinates(int event, int x, int y, int flags, void* param);
Mat undist(Mat);
double get_points_distance(Point2i, Point2i);
void colorExtraction(cv::Mat* src, cv::Mat* dst,
	int code,
	int ch1Lower, int ch1Upper,
	int ch2Lower, int ch2Upper,
	int ch3Lower, int ch3Upper
	);

Mat image1;
Mat src_img, src_frame;
Mat element = Mat::ones(3, 3, CV_8UC1); //追加　3×3の行列で要素はすべて1　dilate処理に必要な行列
int Ax, Ay, Bx, By, Cx, Cy, Dx, Dy;
int Tr, Tg, Tb;
Point2i pre_point; //@comment Point構造体<int型>

int flag = 0;
int ct = 0;
Mat dst_img, colorExtra;

ofstream ofs("out4.csv");

int main(int argc, char *argv[])
{
	//@comment 動画の読み込み
	VideoCapture video("./test2.MP4");
	if (!video.isOpened())
	{
		return -1;
	}
	namedWindow("src", 1);
	namedWindow("dst",1);
	namedWindow("video", 1);
	namedWindow("test1",1);
	namedWindow("binari",1);
	
	video >> src_frame;
	resize(src_frame, src_img, Size(src_img_cols, src_img_rows), CV_8UC3);
	//src_img = undist(src_img) ; //@comment カメラの歪みをとる(GoPro魚眼)
	//------------------座標取得-----------------------------------------------
	//@comment 画像中からマウスで4点を取得その後ESCキーを押すと変換処理が開始する

	namedWindow("getCoordinates");
	imshow("getCoordinates", src_img);
	//@comment 変換したい四角形の四隅の座標をとる(クリック)
	cvSetMouseCallback("getCoordinates", getCoordinates, NULL);
	waitKey(0);
	destroyAllWindows();


	//------------------透視変換-----------------------------------------------
	Point2f pts1[] = { Point2f(Ax, Ay), Point2f(Bx, By),
		Point2f(Cx, Cy), Point2f(Dx, Dy) };

	Point2f pts2[] = { Point2f(0, src_img_rows), Point2f(0, 0),
		Point2f(src_img_cols, 0), Point2f(src_img_cols, src_img_rows) };

	//@comment 透視変換行列を計算
	Mat perspective_matrix = getPerspectiveTransform(pts1, pts2);
	Mat dst_img, colorExtra;

	//@comment 変換(線形補完)
	warpPerspective(src_img, dst_img, perspective_matrix, src_img.size(), INTER_LINEAR);

	//@comment 変換前後の座標を描画
	line(src_img, pts1[0], pts1[1], Scalar(255, 0, 255), 2, CV_AA);
	line(src_img, pts1[1], pts1[2], Scalar(255, 255, 0), 2, CV_AA);
	line(src_img, pts1[2], pts1[3], Scalar(255, 255, 0), 2, CV_AA);
	line(src_img, pts1[3], pts1[0], Scalar(255, 255, 0), 2, CV_AA);
	line(src_img, pts2[0], pts2[1], Scalar(255, 0, 255), 2, CV_AA);
	line(src_img, pts2[1], pts2[2], Scalar(255, 255, 0), 2, CV_AA);
	line(src_img, pts2[2], pts2[3], Scalar(255, 255, 0), 2, CV_AA);
	line(src_img, pts2[3], pts2[0], Scalar(255, 255, 0), 2, CV_AA);

	namedWindow("plotCoordinates",1);
	imshow("plotCoordinates",src_img);

	namedWindow("dst", 1);
	imshow("dst", dst_img);
	int frame = 0;
	while (1){

		video >> src_frame;

		if (frame % 10 == 0){

			//@comment 画像をリサイズ(大きすぎるとディスプレイに入りらないため)
			resize(src_frame, src_frame, Size(src_img_cols, src_img_rows), CV_8UC3);
			//src_frame = undist(src_frame); //@comment カメラの歪みをとる(GoPro魚眼)

			//}
			//else{
			//--------------------グレースケール化---------------------------------------

			//(2) RGB値設定 
			int x, y;
			uchar r1, g1, b1, d;
			Vec3b color1;
			//@comment hsvを利用して赤色を抽出
			//入力画像、出力画像、変換、h最小値、h最大値、s最小値、s最大値、v最小値、v最大値
			warpPerspective(src_frame, dst_img, perspective_matrix, src_frame.size(), INTER_LINEAR);
			colorExtraction(&dst_img, &colorExtra, CV_BGR2HSV, 150, 180, 70, 255, 70, 255);
			cvtColor(colorExtra, colorExtra, CV_BGR2GRAY);//@comment グレースケールに変換


			//２値化
			//------------------しきい値目測用--------------------------------------------


			int value = 0;
			createTrackbar("value", "binari", &value, 255, onTrackbarChanged);
			setTrackbarPos("value", "binari", 0);

			Mat binari_2;

			//(2)しきい値決めうち

			//----------------------二値化-----------------------------------------------
			threshold(colorExtra, binari_2, 0, 255, THRESH_BINARY);
			dilate(binari_2, binari_2, element, Point(-1, -1), 3); //膨張処理3回 最後の引数で回数を設定

			//---------------------重心取得---------------------------------------------
			Point2i point = calculate_center(binari_2);//@comment momentで白色部分の重心を求める
			cout << "posion: " << point.x << " " << point.y << endl;//@comment 重心点の表示
			if (point.x != 0){
				int ypos = src_img_rows - (point.y + 6 * ((1000 / point.y) + 1));
				cout << point.x << " " << ypos << endl; //@comment 変換画像中でのロボットの座標(重心)
				ofs << point.x << ", " << ypos << endl; //@comment 変換
			}
			//cout << flag<<endl;
			//@comment 重心点のプロット 
			//画像，円の中心座標，半径，色(青)，線太さ，種類(-1, CV_AAは塗りつぶし) 
			circle(dst_img, Point(point.x, point.y + 6 * ((1000 / point.y)+1)), 5, Scalar(200, 0, 0), -1, CV_AA);


			//---------------------表示部分----------------------------------------------
			Mat base(src_img_rows, src_img_cols * 2, CV_8UC3);
			Mat roi1(base, Rect(0, 0, src_frame.cols, src_frame.rows));
			src_frame.copyTo(roi1);
			//Mat roi2(base, Rect(dst_img.cols, 0, dst_img.cols, dst_img.rows));
			//dst_img.copyTo(roi2);

			imshow("video", src_frame);

			//namedWindow("dst");
			imshow("test1", dst_img);//@comment 出力画像

			//namedWindow("colorExt");
			imshow("colorExt", colorExtra);//@comment 赤抽出画像

			//cout << "frame" << ct++ << endl;

			if (src_frame.empty() || waitKey(30) >= 0)
			{
				destroyAllWindows();
				return 0;
			}

		}
		//}
		//destroyAllWindows();
		frame++;
	}
		ofs.close();
	
}




double get_points_distance(Point2i point, Point2i pre_point){

	return sqrt((point.x - pre_point.x) * (point.x - pre_point.x)
		+ (point.y - pre_point.y) * (point.y - pre_point.y));
}

//@commentトラックバー操作イベントに応じた処理
void onTrackbarChanged(int thres, void*)
{

}

Point2i calculate_center(Mat gray)
{

	Point2i center = Point2i(0, 0);
	//std::cout << center << std::endl;
	Moments moment = moments(gray, true);

	if (moment.m00 != 0)
	{
		center.x = (int)(moment.m10 / moment.m00);
		center.y = (int)(moment.m01 / moment.m00);
	}

	return center;
}

void getCoordinates(int event, int x, int y, int flags, void* param)
{

	static int count = 0;
	switch (event){
	case CV_EVENT_LBUTTONDOWN:

		if (count == 0){
			Ax = x, Ay = y;
			cout << "Ax :" << x << ", Ay: " << y << endl;
		}
		else if (count == 1){
			Bx = x, By = y;
			cout << "Bx :" << x << ", By: " << y << endl;
		}
		else if (count == 2){
			Cx = x, Cy = y;
			cout << "Cx :" << x << ", Cy: " << y << endl;
		}
		else if (count == 3){
			Dx = x, Dy = y;
			cout << "Dx :" << x << ", Dy: " << y << endl;
		}
		else{
			cout << "rgb(" << x << "," << y << ")  ";

			Vec3b target_color = src_img.at<Vec3b>(y, x);
			uchar r, g, b;
			Tr = target_color[2];
			Tg = target_color[1];
			Tb = target_color[0];
			cout << "r:" << Tr << " g:" << Tg << " b:" << Tb << endl;
		}
		count++;
		break;
	default:
		break;
	}
}

Mat undist(Mat src_img)
{
	Mat dst_img;

	//カメラマトリックス
	Mat cameraMatrix = (Mat_<double>(3, 3) << 469.96, 0, 400, 0, 467.68, 300, 0, 0, 1);
	//歪み行列
	Mat distcoeffs = (Mat_<double>(1, 5) << -0.18957, 0.037319, 0, 0, -0.00337);

	undistort(src_img, dst_img, cameraMatrix, distcoeffs);
	return dst_img;
}

void colorExtraction(cv::Mat* src, cv::Mat* dst,
	int code,
	int ch1Lower, int ch1Upper,
	int ch2Lower, int ch2Upper,
	int ch3Lower, int ch3Upper
	)
{
	cv::Mat colorImage;
	int lower[3];
	int upper[3];

	cv::Mat lut = cv::Mat(256, 1, CV_8UC3);

	cv::cvtColor(*src, colorImage, code);

	lower[0] = ch1Lower;
	lower[1] = ch2Lower;
	lower[2] = ch3Lower;

	upper[0] = ch1Upper;
	upper[1] = ch2Upper;
	upper[2] = ch3Upper;

	for (int i = 0; i < 256; i++){
		for (int k = 0; k < 3; k++){
			if (lower[k] <= upper[k]){
				if ((lower[k] <= i) && (i <= upper[k])){
					lut.data[i*lut.step + k] = 255;
				}
				else{
					lut.data[i*lut.step + k] = 0;
				}
			}
			else{
				if ((i <= upper[k]) || (lower[k] <= i)){
					lut.data[i*lut.step + k] = 255;
				}
				else{
					lut.data[i*lut.step + k] = 0;
				}
			}
		}
	}
	//LUTを使用して二値化
	cv::LUT(colorImage, lut, colorImage);

	//namedWindow("colorImage", 1);

	//Channel毎に分解
	std::vector<cv::Mat> planes;
	cv::split(colorImage, planes);

	//マスクを作成
	cv::Mat maskImage;
	cv::bitwise_and(planes[0], planes[1], maskImage);
	cv::bitwise_and(maskImage, planes[2], maskImage);

	//出力
	cv::Mat maskedImage;
	src->copyTo(maskedImage, maskImage);
	*dst = maskedImage;

}
