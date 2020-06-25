#pragma once

#include "FeatureCloud.h"
#include "pointProcess.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <utility>　　//pair的头文件

class svmTrainer
{
public:
	svmTrainer();
	~svmTrainer();

	 enum class classifier{SVM, KNN, RTrees};
private:
	cv::Ptr<cv::ml::SVM> svm;
	cv::Ptr<cv::ml::RTrees> forest;
	cv::Ptr<cv::ml::KNearest> knn;
	cv::Ptr<cv::ml::Boost> boosting;

	cv::Ptr<cv::ml::TrainData> trainDataProcesser;

	cv::Mat trainData;
	cv::Mat trainDataLabel;
	cv::Mat testData;
	cv::Mat testDataLabel;

public:
	//vfh特征向量转cvMat
	cv::Mat vector2Mat(pcl::PointCloud<pcl::VFHSignature308>::Ptr inputDescriptor);

	void prepareTrainTestData(cv::Mat& _trainingData, cv::Mat& _trainingLabels);

	void trainSVM(cv::Mat& _trainingData, cv::Mat& _trainingLabels, std::string fileName);

	void trainRTrees(cv::Mat& _trainingData, cv::Mat& _trainingLabels, std::string fileName);

	void trainKNN(cv::Mat& _trainingData, cv::Mat& _trainingLabels, std::string fileName);

	void trainBoost(cv::Mat& _trainingData, cv::Mat& _trainingLabels, std::string fileName);

	void loadSVM(classifier classifier_, std::string fileName);

	float getConfidence(float distance);

	void validateSVM(cv::Ptr<cv::ml::StatModel> classifier, cv::Mat& _testData, cv::Mat& _testLabels);

	void predict_SVM(cv::Mat& query, std::pair<int, double> & result);

	void predict_RTrees(cv::Mat& query, std::pair<int, double> & result);

	void predict(classifier classifier_, cv::Mat& query, int& result);
};

