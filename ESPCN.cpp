#include <iostream>
#include <opencv2/dnn_superres.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/quality.hpp>
#include <chrono>
#include <ctime> 
 #include <pthread.h>

using namespace std;
using namespace cv;
using namespace dnn;
using namespace dnn_superres;


typedef struct pThreadWorker {
    pthread_t                           thread;                             //!< Thread identifier
    string                              setting;
    string                              value;                            

}pThreadWorker;
pThreadWorker                               pthreads[1];
void *Super_Resolution(void *arg);
Mat upscaleImage(Mat img, string modelName, string modelPath, int scale){
	DnnSuperResImpl sr;
	sr.readModel(modelPath);
	sr.setModel(modelName,scale);
	// Output image
	Mat outputImage;
	sr.upsample(img, outputImage);
	return outputImage;
}
double getPSNR(const Mat& I1, const Mat& I2)
{
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2
    Scalar s = sum(s1);        // sum elements per channel
    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
    if( sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double mse  = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}
int main(int argc, char *argv[])
{
	//Single Core Settings
	cpu_set_t cpu1;
	int temp;
	CPU_ZERO(&cpu1);
   	CPU_SET(0, &cpu1);
	//Single Core Settings

	//For Multi-Core, remove //Single Core Settings code blocks.

	int create_result = pthread_create(&pthreads[0].thread, NULL, Super_Resolution, (void *)0);
	if (create_result != 0)
	{   
			cout<<"Error creating pose_data reader worker thread"<<endl;      
	} 
	temp = pthread_setaffinity_np(pthreads[0].thread, sizeof(cpu_set_t), &cpu1);

	//Single Core Settings
	for (int j = 0; j < CPU_SETSIZE; j++)
		if (CPU_ISSET(j, &cpu1))
			printf("CPU1: CPU %d\n", j);

	 // Waiting for the created thread to terminate
    pthread_join(pthreads[0].thread, NULL);
	//Single Core Settings

	//For Multi-Core, remove //Single Core Settings code blocks.

	return 0;
}

void *Super_Resolution(void *arg)
{   
	// Read image
	Mat img_x2 = imread("metin.png");	

	// ESPCN (x2)
	string path_x2 = "ESPCN_x2.pb";
	string modelName_x2 = "espcn";
	int scale_x2 = 2;
	auto start_x2= std::chrono::system_clock::now();
	Mat result_x2 = upscaleImage(img_x2, modelName_x2, path_x2, scale_x2);

	auto end_x2 = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds_x2 = end_x2-start_x2;

	
	// Read image
	Mat img_x3 = imread("metin.png");	

	// ESPCN (x3)
	string path_x3 = "ESPCN_x3.pb";
	string modelName_x3 = "espcn";
	int scale_x3 = 3;
	auto start_x3= std::chrono::system_clock::now();
	Mat result_x3 = upscaleImage(img_x3, modelName_x3, path_x3, scale_x3);

	auto end_x3 = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds_x3 = end_x3-start_x3;

	
	// Read image
	Mat img_x4 = imread("metin.png");	

	// ESPCN (x4)
	string path_x4 = "ESPCN_x4.pb";
	string modelName_x4 = "espcn";
	int scale_x4 = 4;
	auto start_x4= std::chrono::system_clock::now();
	Mat result_x4 = upscaleImage(img_x4, modelName_x4, path_x4, scale_x4);

	auto end_x4 = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds_x4 = end_x4-start_x4;

	// Image resized using OpenCV
	Mat resized_x2;
	cv::resize(img_x2, resized_x2, cv::Size(), scale_x2, scale_x2);
	double psnr_x2;
	psnr_x2=getPSNR(resized_x2,result_x2);
	
	Scalar q_x2 = quality::QualitySSIM::compute(resized_x2, result_x2, noArray());
	double ssim_x2 = mean(Vec3d((q_x2[0]), q_x2[1], q_x2[2]))[0];

	Mat resized_x3;
	cv::resize(img_x3, resized_x3, cv::Size(), scale_x3, scale_x3);
	double psnr_x3;
	psnr_x3=getPSNR(resized_x3,result_x3);
	
	Scalar q_x3 = quality::QualitySSIM::compute(resized_x3, result_x3, noArray());
	double ssim_x3 = mean(Vec3d((q_x3[0]), q_x3[1], q_x3[2]))[0];

	Mat resized_x4;
	cv::resize(img_x4, resized_x4, cv::Size(), scale_x4, scale_x4);
	double psnr_x4;
	psnr_x4=getPSNR(resized_x4,result_x4);
	
	Scalar q_x4 = quality::QualitySSIM::compute(resized_x4, result_x4, noArray());
	double ssim_x4 = mean(Vec3d((q_x4[0]), q_x4[1], q_x4[2]))[0];
	
	cout <<"ESPCN:\t\t\t  x2\t\t  x3\t\t  x4"<<endl;
	cout <<"Elapsed time:\t\t"<<elapsed_seconds_x2.count()<<"s"<<"\t"<<elapsed_seconds_x3.count()<<"s"<<"\t"<<elapsed_seconds_x4.count()<<"s"<<endl;
	cout <<"PSNR:\t\t\t"<<psnr_x2<<"dB"<<"\t"<<psnr_x3<<"dB"<<"\t"<<psnr_x4<<"dB"<<endl;
	cout <<"SSIM:\t\t\t"<<ssim_x2<<"\t"<<ssim_x3<<"\t"<<ssim_x4<<endl;


	imwrite("metin_orj.jpg",resized_x4);
	imwrite("metin_sr.jpg",result_x4);
	/*imshow("Original image",img);
	imshow("SR upscaled",result);
	imshow("OpenCV upscaled",resized);
	waitKey(0);
	destroyAllWindows();*/
}