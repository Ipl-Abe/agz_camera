#include <iostream>
#include <vector>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

static const int src_img_cols = 300; //width
static const int src_img_rows = 300; //height


using namespace cv;
using namespace std;


Point2i calculate_center(Mat);
void getCoordinates(int event, int x, int y, int flags, void* param);
Mat undist(Mat);
double get_points_distance(Point2i, Point2i);
void set_target(cv::Point2i& targt, std::vector<cv::Point2i>& allTarget);
std::string robot_action(cv::Point2i Current, cv::Point2i Previous, cv::Point2i Target);
bool is_update_target(cv::Point2i Current, cv::Point2i Target);
std::string is_out(cv::Point2i Robot);
void colorExtraction(cv::Mat* src, cv::Mat* dst,
	int code,
	int ch1Lower, int ch1Upper,
	int ch2Lower, int ch2Upper,
	int ch3Lower, int ch3Upper
	);

Mat image1;
Mat src_img, src_frame;
Mat element = Mat::ones(3, 3, CV_8UC1); //@comment 追加　3×3の行列で要素はすべて1　dilate処理に必要な行列
int Ax, Ay, Bx, By, Cx, Cy, Dx, Dy;
int Tr, Tg, Tb;
Point2i pre_point; //@comment Point構造体<int型>

int flag = 0;
//int ct = 0;
Mat dst_img, colorExtra;

ofstream ofs("out4.csv");
//@動画出力用変数
const string  str = "test.avi";

Point2i target, P0 = { 0, 0 }, P1 = { 0, 0 };
std::vector<Point2i> allTarget;
std::vector<Point2i>::iterator target_itr;
string action;
Point2i a, b, c, d;	//内側領域の頂点
int width;
int depth;

int main(int argc, char *argv[])
{

	cout << "please type width and depth of field (m)" << endl << "width : ";
	cin >> width;
	cout << "depth : ";
	cin >> depth;

	

	//@comment カメラの呼び出し pcのカメラ : 0 webカメラ : 1 
	VideoCapture cap(1);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640); //@comment webカメラの横幅を設定
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480); //@comment webカメラの縦幅を設定
	if (!cap.isOpened()) return -1; //@comment 呼び出しミスがあれば終了

	VideoWriter write("out2.avi", CV_FOURCC('M', 'J', 'P', 'G'), cap.get(CV_CAP_PROP_FPS),
		cv::Size(src_frame.rows, src_frame.cols), true);
	if (!write.isOpened()) return -1;

	namedWindow("src", 1);
	namedWindow("dst", 1);
	namedWindow("video", 1);
	namedWindow("test1", 1);
	namedWindow("binari", 1);

	cap >> src_frame; //@comment 1フレーム取得
	cap >> src_frame;
	cap >> src_frame;
	cap >> src_frame;

	resize(src_frame, src_frame, Size(src_img_cols, src_img_rows), CV_8UC3); //@取得画像のリサイズ
	//src_img = undist(src_img) ; //@comment カメラの歪みをとる(GoPro魚眼)


	//------------------座標取得-----------------------------------------------
	//@comment 画像中からマウスで4点を取得その後ESCキーを押すと変換処理が開始する

	namedWindow("getCoordinates");
	imshow("getCoordinates", src_frame);
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
	warpPerspective(src_frame, dst_img, perspective_matrix, src_frame.size(), INTER_LINEAR);

	//@comment 変換前後の座標を描画
	line(src_frame, pts1[0], pts1[1], Scalar(255, 0, 255), 2, CV_AA);
	line(src_frame, pts1[1], pts1[2], Scalar(255, 255, 0), 2, CV_AA);
	line(src_frame, pts1[2], pts1[3], Scalar(255, 255, 0), 2, CV_AA);
	line(src_frame, pts1[3], pts1[0], Scalar(255, 255, 0), 2, CV_AA);
	line(src_frame, pts2[0], pts2[1], Scalar(255, 0, 255), 2, CV_AA);
	line(src_frame, pts2[1], pts2[2], Scalar(255, 255, 0), 2, CV_AA);
	line(src_frame, pts2[2], pts2[3], Scalar(255, 255, 0), 2, CV_AA);
	line(src_frame, pts2[3], pts2[0], Scalar(255, 255, 0), 2, CV_AA);

	namedWindow("plotCoordinates", 1);
	imshow("plotCoordinates", src_frame);

	namedWindow("dst", 1);
	imshow("dst", dst_img);


	int frame = 0; //@comment フレーム数保持変数

	set_target(target, allTarget);
	target_itr = allTarget.begin();

	while (1){

		if (target_itr == allTarget.end() - 1){
			target_itr = allTarget.begin();
		}

		cap >> src_frame;

		if (frame % 1 == 0){ //@comment　フレームの取得数を調節可能

			//@comment 画像をリサイズ(大きすぎるとディスプレイに入りらないため)
			resize(src_frame, src_frame, Size(src_img_cols, src_img_rows), CV_8UC3);
			//src_frame = undist(src_frame); //@comment カメラの歪みをとる(GoPro魚眼)

			//--------------------グレースケール化---------------------------------------

			//変換(線形補完)
			warpPerspective(src_frame, dst_img, perspective_matrix, src_frame.size(), INTER_LINEAR);
			//@comment hsvを利用して赤色を抽出
			//入力画像、出力画像、変換、h最小値、h最大値、s最小値、s最大値、v最小値、v最大値
			colorExtraction(&dst_img, &colorExtra, CV_BGR2HSV, 150, 180, 70, 255, 70, 255);
			cvtColor(colorExtra, colorExtra, CV_BGR2GRAY);//@comment グレースケールに変換


			//２値化
			//------------------しきい値目測用--------------------------------------------
			Mat binari_2;

			//----------------------二値化-----------------------------------------------
			threshold(colorExtra, binari_2, 0, 255, THRESH_BINARY);
			dilate(binari_2, binari_2, element, Point(-1, -1), 3); //膨張処理3回 最後の引数で回数を設定

			//---------------------重心取得---------------------------------------------
			Point2i point = calculate_center(binari_2);//@comment momentで白色部分の重心を求める
			//cout << "posion: " << point.x << " " << point.y << endl;//@comment 重心点の表示
			int ypos;
			if (point.x != 0){
				ypos = src_img_rows - (point.y + 6 * ((1000 / point.y) + 1));
				//cout << point.x << " " << ypos << endl; //@comment 変換画像中でのロボットの座標(重心)
				ofs << point.x << ", " << ypos << endl; //@comment 変換
			}

			//---------------------ロボットの動作取得------------------------------------
			P1 = { point.x, src_img_rows - ypos };
			if (target_itr != allTarget.end() && P1.x != 0 && P1.y != 0 && P0.x != 0 && P0.y != 0){
				line(dst_img, P1, P0, Scalar(255, 0, 0), 2, CV_AA);
				if (is_update_target(P1, *target_itr)){
					// ターゲットの更新
					std::cout << "UPDATE" << std::endl;
					target_itr++;
				}
				action = robot_action(P0, P1, *target_itr);
				std::cout << "target: " << target_itr->x << ", " << target_itr->y << "	position: " << P1.x << ", " << P1.y
					<< is_out(P1) << action << std::endl;
			}
			P0 = P1;


			//-------------------重心点のプロット----------------------------------------- 
			if (!point.y == 0){ //@comment point.y == 0の場合はexceptionが起こる( 0除算 )
				//@comment 画像，円の中心座標，半径，色(青)，線太さ，種類(-1, CV_AAは塗りつぶし)
				circle(dst_img, Point(point.x, point.y + 6 * ((1000 / point.y) + 1)), 8, Scalar(0, 0, 0), -1, CV_AA);
			}

			//------------------ターゲットのプロット--------------------------------------
			int n = 0;
			for (vector<Point2i>::iterator itr = allTarget.begin(); itr != allTarget.end(); itr++){
				cv::circle(dst_img, cv::Point(*itr), 48, cv::Scalar(255, 255, 0), 3, 4);
				cv::putText(dst_img, std::to_string(n), cv::Point(*itr), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 0.5, CV_AA);
				n++;
			}
			cv::circle(dst_img, cv::Point(*target_itr), 48, cv::Scalar(0, 0, 0), 3, 4);

			cv::putText(dst_img, is_out(P1), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 1.0, CV_AA);
			cv::putText(dst_img, action, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 1.0, CV_AA);


			//------------------マスのプロット--------------------------------------
			double w = (double)src_img_cols / (double)width;
			double d = (double)src_img_rows / (double)depth;

			for (int i = 0; i <= src_img_cols; i += 100){
				for (int j = 0; j <= src_img_rows; j += 100){

					line(dst_img, Point(i, j), Point(i, src_img_cols), Scalar(200, 200, 200), 3);
					line(dst_img, Point(i, j), Point(src_img_rows, j), Scalar(200, 200, 200), 3);
				}
			}


			//------------------直進領域のプロット--------------------------------------
			cv::Point2i A = { 100, src_img_rows - 100 }, B = { 100, 100 }, C = { src_img_cols - 100, 100 }, D = { src_img_cols - 100, src_img_rows - 100 };

			line(dst_img, Point(A), Point(B), Scalar(200, 0, 0), 3);
			line(dst_img, Point(B), Point(C), Scalar(200, 0, 0), 3);
			line(dst_img, Point(C), Point(D), Scalar(200, 0, 0), 3);
			line(dst_img, Point(D), Point(A), Scalar(200, 0, 0), 3);


			//---------------------表示部分----------------------------------------------

			//選択した四隅を線で結ぶ
			//line(dst_img, Point(Ax, Ay), Point(Bx, By), Scalar(200, 0, 0), 3);
			//line(dst_img, Point(Bx, By), Point(Cx, Cy), Scalar(0, 200, 0), 3);
			//line(dst_img, Point(Cx, Cy), Point(Dx, Dy), Scalar(200, 0, 0), 3);
			//line(dst_img, Point(Dx, Dy), Point(Ax, Ay), Scalar(0, 200, 0), 3);

			//imshow("drawing", src_img);
			//---------------------変換後の画像に1m四方の升目形成----------------------------

			//imshow("dst_img", dst_img);

			//Mat Eximg = dst_img;
			
			/*
			//内側領域と外側領域の作成
			line(dst_img, Point(100 * w, 100 * d), Point(100 * w, (depth - 100)*d), Scalar(200, 0, 0), 2);
			line(dst_img, Point(100 * w, 100 * d), Point((width - 100)*w, 100 * d), Scalar(200, 0, 0), 2);
			line(dst_img, Point((width - 100)*w, 100 * d), Point((width - 100)*w, (depth - 100)*d), Scalar(200, 0, 0), 2);
			line(dst_img, Point(100 * w, (depth - 100)*d), Point((width - 100)*w, (depth - 100)*d), Scalar(200, 0, 0), 2);
			*/
			//内側領域と外側領域の作成
//			line(dst_img, Point(100, 100), Point(100, src_img_cols), Scalar(200, 0, 0), 2);
	//		line(dst_img, Point(100, 100), Point(src_img_rows, 100), Scalar(200, 0, 0), 2);
		//	line(dst_img, Point(src_img_rows, 100), Point(src_img_rows, src_img_cols), Scalar(200, 0, 0), 2);
			//line(dst_img, Point(100, src_img_cols - 100), Point(src_img_rows, src_img_cols - 100), Scalar(200, 0, 0), 2);

			//imshow("video", src_frame);
			imshow("red_point", dst_img);//@comment 出力画像
			imshow("colorExt", colorExtra);//@comment 赤抽出画像
			//cout << "frame" << ct++ << endl; //@comment frame数表示
			//write << dst_img;
			std::cout << frame << std::endl;
			if (src_frame.empty() || waitKey(30) >= 0)
			{
				destroyAllWindows();
				return 0;
			}
		}
		frame++;
		write << dst_img;
	}
	ofs.close(); //@comment ファイルストリームの解放
}

//@comment 2点間の距離取得関数
double get_points_distance(Point2i point, Point2i pre_point){

	return sqrt((point.x - pre_point.x) * (point.x - pre_point.x)
		+ (point.y - pre_point.y) * (point.y - pre_point.y));
}

//@comment 重心取得用関数
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

//@comment 入力画像から4点を設定する関数
void getCoordinates(int event, int x, int y, int flags, void* param)
{

	static int count = 0;
	switch (event){
	case CV_EVENT_LBUTTONDOWN://@comment 左クリックが押された時

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
//@comment カメラキャリブレーション用関数(gopro用)
Mat undist(Mat src_img)
{
	Mat dst_img;

	//@comment カメラマトリックス(gopro)
	Mat cameraMatrix = (Mat_<double>(3, 3) << 469.96, 0, 400, 0, 467.68, 300, 0, 0, 1);
	//@comment 歪み行列(gopro)
	Mat distcoeffs = (Mat_<double>(1, 5) << -0.18957, 0.037319, 0, 0, -0.00337);

	undistort(src_img, dst_img, cameraMatrix, distcoeffs);
	return dst_img;
}

//@comment 色抽出用関数 
void colorExtraction(cv::Mat* src, cv::Mat* dst,
	int code,
	int ch1Lower, int ch1Upper, //@comment H(色相)　最小、最大
	int ch2Lower, int ch2Upper, //@comment S(彩度)　最小、最大
	int ch3Lower, int ch3Upper  //@comment V(明度)　最小、最大
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
	//@comment LUTを使用して二値化
	cv::LUT(colorImage, lut, colorImage);

	//@comment Channel毎に分解
	std::vector<cv::Mat> planes;
	cv::split(colorImage, planes);

	//@comment マスクを作成
	cv::Mat maskImage;
	cv::bitwise_and(planes[0], planes[1], maskImage);
	cv::bitwise_and(maskImage, planes[2], maskImage);

	//@comemnt 出力
	cv::Mat maskedImage;
	src->copyTo(maskedImage, maskImage);
	*dst = maskedImage;

}

//@comment ターゲット設定
void set_target(cv::Point2i& targt, std::vector<cv::Point2i>& allTarget){

	int height = src_img_rows;
	int width = src_img_cols;
	allTarget.clear();

	int n = 0;
	for (int j = 0; j < (height / 100) - 2; j++){
		if (j % 2 == 0){
			for (int i = 0; i < (width / 100) - 2; i++){
				target.x = (i + 1) * width / (width / 100) + 50;	target.y = height - (j + 1) * height / (height / 100) - 50;
				allTarget.push_back(target);
				n++;
			}
		}
		else{
			for (int i = (width / 100) - 3; i >= 0; i--){
				target.x = (i + 1) * width / (width / 100) + 50;	target.y = height - (j + 1) * height / (height / 100) - 50;
				allTarget.push_back(target);
				n++;
			}
		}
	}
}

//@comment ロボットの動作決定関数
std::string robot_action(cv::Point2i Current, cv::Point2i Previous, cv::Point2i Target){
	cv::Point2i P0 = Current - Previous;
	cv::Point2i P1 = Target - Current;

	int dx1 = abs(Current.x - 100);
	int dx2 = abs(Current.x - src_img_rows - 100);

	double angle = asin(P0.cross(P1) / (sqrt(P0.x * P0.x + P0.y * P0.y) * sqrt(P1.x * P1.x + P1.y * P1.y))) / CV_PI * 180;
	if (P0.dot(P1) >= 0){
		if (P0.cross(P1) >= 0){
			angle = 360.0 - angle;
		}
		else if (P0.cross(P1) < 0){
			angle = -360.0 - angle;
		}
	}

	if (-30 < angle && angle < 30){
		return std::string("f");
	}
	else if (angle <= -30){
		if ((dx1 < 50.0) || (dx2 < 50.0)){
			return std::string("r (fast)");
		}
		else return std::string("r (slow)");
	}
	else{
		if ((dx1 < 50.0) || (dx2 < 50.0)){
			return std::string("l (fast)");
		}
		else return std::string("l (slow)");
	}
}

//@comment true->ターゲット更新, false->ターゲット更新なし
bool is_update_target(cv::Point2i Current, cv::Point2i Target){
	//Target.y = src_img_rows - Target.y;
	int dx = Current.x - Target.x;
	int dy = Current.y - Target.y;
	double d = sqrt(dx * dx + dy * dy);

	if (d < 50.0){
		return true;
	}
	else return false;
}

std::string is_out(cv::Point2i Robot){
	cv::Point2i A = { 100, src_img_rows - 100 }, B = { 100, 100 }, C = { src_img_cols - 100, 100 }, D = { src_img_cols - 100, src_img_rows - 100 };
	cv::Point2i BA = A - B, BC = C - B, BP = Robot - B;
	cv::Point2i DC = C - D, DA = A - D, DP = Robot - D;
	int c1, c2, c3, c4;
	bool flag1 = false, flag2 = false;

	line(dst_img, Point(A), Point(B), Scalar(200, 200, 200), 3);
	line(dst_img, Point(B), Point(C), Scalar(200, 200, 200), 3);
	line(dst_img, Point(C), Point(D), Scalar(200, 200, 200), 3);
	line(dst_img, Point(D), Point(A), Scalar(200, 200, 200), 3);

	c1 = BA.cross(BP);
	c2 = BP.cross(BC);
	c3 = DC.cross(DP);
	c4 = DP.cross(DA);

	if ((c1 >= 0 && c2 >= 0) || (c1 < 0 && c2 < 0)){
		flag1 = true;
	}
	if ((c3 >= 0 && c4 >= 0) || (c3< 0 && c4 < 0)){
		flag2 = true;
	}
	if (flag1 && flag2){
		return "IN";
	}
	else return "OUT";
}

