#pragma once

#include "FeatureCloud.h"
#include "pointProcess.h"
#include "svmTrainer.h"

class recognition3D
{
public:
	recognition3D();
	~recognition3D();

	enum class Axis{ x, y , z };

	static inline bool probcmp(std::pair<int, int> x, std::pair<int, int> y)
	{
		return x.second > y.second;
	}

	bool set_ObjectLibrary(std::vector< PointCloud::Ptr > &objectLib, bool local = false);
	
	bool set_ObjectLibrary_label(std::vector<std::string>& fileNames);

	bool set_svm_train_model();

	void svm_classifier_train(std::string savefilename);

	void load_svmClassifier(svmTrainer::classifier classifier_, std::string file);

	int svm_predict(PointCloud::Ptr input);

	//计算转换矩阵
	float RotationTranslationCompute(
		int targetIndex,
		FeatureCloud& cloudsource,
		Eigen::Matrix4f &tranResult);

	//旋转创造数据集
	static void rotateModelToDataSet(PointCloud::Ptr input, 
		int angle, Axis axis,
		std::vector<PointCloud::Ptr> outputList, std::string path,std::string filename);
	//cvfh转化为cvMat
	cv::Mat cvfh_to_cvMat(FeatureCloud& cloud);
	//esf转化为cvMat
	cv::Mat esf_to_cvMat(FeatureCloud& cloud);
	//ourcvfh+esf 转化为cvMat
	cv::Mat ourcvfh_and_esf_to_cvMat(FeatureCloud& cloud);

	static bool compareConfidence(std::pair<int, double> pair1,
		std::pair<int, double> pair2)
	{
		return pair1.second < pair2.second;
	}
	
private:
	//模板库
	std::vector< FeatureCloud > objectLibrary;
	std::vector< int > objectLib_labelVec;
	cv::Mat objectLibrary_sample;
	cv::Mat objectLibrary_label;
	//点云处理器
	pointProcess pointProcesser;
	svmTrainer svmTrainer_;
};

//希尔排序
template <typename T>
void shellsort(T A[], int l, int h, int* index)
{
	if (l < h)
	{
		int d;//增量
		int tmp;
		int j;
		int size = h - l + 1;
		for (d = size / 2; d >= 1; d /= 2)
		{
			//组内使用插入排序
			//d+l是第一组的第二个元素
			for (int i = d + l; i <= h; ++i)
			{
				tmp = A[i];
				j = i - d;//该组中，当前元素的前一个元素
						  //寻找要插入的位置
				while (j >= l && A[j] > tmp)
				{
					T temp = A[j];
					A[j] = A[j + d];
					A[j + d] = temp;

					int indexTemp = j;
					index[j] = index[j + d];
					index[j + d] = indexTemp;

					j -= d;
				}
				A[j + d] = tmp;
				index[j + d] = i;
			}
		}
	}
}

template <typename T>
void bubbleSort(T A[], int l, int h, int* index)
{
	if (l < h)
	{
		int size = h - l + 1;
		for (int i = 0; i < size; ++i)
		{
			for (int j = 0; j < size - i - 1; ++j)
			{
				if (A[j] > A[j + 1])
				{
					T temp = A[j];
					A[j] = A[j + 1];
					A[j + 1] = temp;

					int tempIndex = index[j];
					index[j] = index[j + 1];
					index[j + 1] = tempIndex;
				}
			}
		}
	}
}