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

	bool set_ObjectLibrary(std::vector< PointCloud::Ptr > &objectLib);
	
	bool set_ObjectLibrary_label(std::vector<std::string>& fileNames);

	//��ת�������ݼ�
	static void rotateModelToDataSet(PointCloud::Ptr input, 
		int angle, Axis axis,
		std::vector<PointCloud::Ptr> outputList, std::string path,std::string filename);

	//cvfhת��ΪcvMat
	cv::Mat cvfh_to_cvMat(FeatureCloud& cloud);

	//esfת��ΪcvMat
	cv::Mat esf_to_cvMat(FeatureCloud& cloud);

	//ourcvfh+esf ת��ΪcvMat
	cv::Mat ourcvfh_and_esf_to_cvMat(FeatureCloud& cloud);

	static bool compareConfidence(std::pair<int, double> pair1,
		std::pair<int, double> pair2)
	{
		return pair1.second < pair2.second;
	}
	
	bool set_svm_train_model();

	void svm_classifier_train(std::string savefilename);

	int svm_predict(PointCloud::Ptr input);

	void load_svmClassifier(svmTrainer::classifier classifier_, std::string file);

private:
	//ģ���
	std::vector< FeatureCloud > objectLibrary;
	std::vector< int > objectLib_labelVec;
	cv::Mat objectLibrary_sample;
	cv::Mat objectLibrary_label;
	//���ƴ�����
	pointProcess pointProcesser;
	svmTrainer svmTrainer_;
};

//ϣ������
template <typename T>
void shellsort(T A[], int l, int h, int* index)
{
	if (l < h)
	{
		int d;//����
		int tmp;
		int j;
		int size = h - l + 1;
		for (d = size / 2; d >= 1; d /= 2)
		{
			//����ʹ�ò�������
			//d+l�ǵ�һ��ĵڶ���Ԫ��
			for (int i = d + l; i <= h; ++i)
			{
				tmp = A[i];
				j = i - d;//�����У���ǰԪ�ص�ǰһ��Ԫ��
						  //Ѱ��Ҫ�����λ��
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