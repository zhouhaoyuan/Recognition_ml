#include "svmTrainer.h"
#include <algorithm>


svmTrainer::svmTrainer()
{
}


svmTrainer::~svmTrainer()
{
}
//vfh特征向量转cvMat
cv::Mat svmTrainer::vector2Mat(pcl::PointCloud<pcl::VFHSignature308>::Ptr inputDescriptor)
{	
	cv::Mat testArray = cv::Mat::zeros(1, 308, CV_32FC1);
	if (inputDescriptor == nullptr)
	{
		std::cout << "Error: vector2Mat(), inputDescriptor is null\n";
		return testArray;
	}
	
	float* RowPtr = testArray.ptr<float>(0);
	for (size_t i = 0; i < 308; ++i)
	{
		RowPtr[i] = (float)inputDescriptor->points[0].histogram[i];
	}
	return testArray;
}

void svmTrainer::prepareTrainTestData(cv::Mat& _trainingData, cv::Mat& _trainingLabels)
{
	//1.准备数据
	trainDataProcesser = cv::ml::TrainData::create(_trainingData, cv::ml::SampleTypes::ROW_SAMPLE, _trainingLabels);
	trainDataProcesser->setTrainTestSplitRatio(0.8, true);
	int n_train_samples = trainDataProcesser->getNTrainSamples();
	int n_test_samples = trainDataProcesser->getNTestSamples();
	std::cout << "\nFound " << n_train_samples << " Train Samples, and "
		<< n_test_samples << " Test Samples" << std::endl;

	cv::Mat testDataIdx = trainDataProcesser->getTestSampleIdx();
	cv::Mat trainDataIdx = trainDataProcesser->getTrainSampleIdx();

    trainData = trainDataProcesser->getSubMatrix(_trainingData, trainDataIdx, cv::ml::SampleTypes::ROW_SAMPLE);
	trainDataLabel = trainDataProcesser->getSubMatrix(_trainingLabels, trainDataIdx, cv::ml::SampleTypes::ROW_SAMPLE);
	testData = trainDataProcesser->getSubMatrix(_trainingData, testDataIdx, cv::ml::SampleTypes::ROW_SAMPLE);
	testDataLabel = trainDataProcesser->getSubMatrix(_trainingLabels, testDataIdx, cv::ml::SampleTypes::ROW_SAMPLE);
}
void svmTrainer::trainSVM(cv::Mat& _trainingData, cv::Mat& _trainingLabels, std::string fileName)
{
	prepareTrainTestData(_trainingData, _trainingLabels);
	//2.设置SVM参数
	svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);//可以处理非线性分割的问题
	svm->setKernel(cv::ml::SVM::RBF);
	//svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::Type::MAX_ITER, 1000, 1e-6));
	//svm->setC(0.1);
	cv::ml::ParamGrid c_grid(0.0001, 1000, 10);
	//svm->setGamma(0.01);
	cv::ml::ParamGrid gamma_grid(0.0001, 1000, 10);
	//cv::ml::ParamGrid c_grid(cv::ml::SVM::getDefaultGrid(cv::ml::SVM::ParamTypes::C));
	//cv::ml::ParamGrid gamma_grid(cv::ml::SVM::getDefaultGrid(cv::ml::SVM::ParamTypes::GAMMA));
	//3.训练支持向量
	svm->trainAuto(trainDataProcesser, 10, c_grid, gamma_grid);
	//svm->train(trainDataProcesser);
		
	//训练集/验证集的误差
	cv::Mat results;
	float train_performance = svm->calcError(trainDataProcesser, false, results);

	//std::cout <<"\n[SVM] train_performance : " <<results << std::endl;
	std::cout << "\n[SVM] train_performance : " << train_performance << "%" << std::endl;

	float test_performance = svm->calcError(trainDataProcesser, true, results);

	//std::cout << "\n[SVM] test_performance : " << results << std::endl;
	std::cout << "\n[SVM] test_performance : " << test_performance << "%" << std::endl;

	validateSVM(svm, trainData, trainDataLabel);
	validateSVM(svm, testData, testDataLabel);

	std::cout << "\nSVM trained " << std::endl;
	fileName += "_svm.xml";
	svm->save(fileName.c_str());
}
void svmTrainer::trainRTrees(cv::Mat& _trainingData, cv::Mat& _trainingLabels, std::string fileName)
{
	prepareTrainTestData(_trainingData, _trainingLabels);
	//2.设置RTrees参数
	forest = cv::ml::RTrees::create();
	forest->setMaxDepth(10);
	forest->setMinSampleCount(10);
	forest->setMaxCategories(5);
	forest->setCalculateVarImportance(true);
	forest->setActiveVarCount(50);//随机的特征数目
	forest->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 1e-6));
	forest->train(trainDataProcesser, 0);

	//训练集/验证集的误差
	cv::Mat results;
	float train_performance = forest->calcError(trainDataProcesser, false, results);
	float test_performance = forest->calcError(trainDataProcesser, true, results);

	std::cout << "\n[RTrees] train_performance : " << train_performance << "%" << std::endl;
	validateSVM(forest, trainData, trainDataLabel);
	std::cout << "\n[RTrees] test_performance : " << test_performance << "%" << std::endl;
	validateSVM(forest, testData, testDataLabel);

	cv::Mat varImportance = forest->getVarImportance();
	std::cout << "\nthe variance importance : \n" << varImportance  << std::endl;

	std::cout << "\nRTrees trained " << std::endl;
	fileName += "_rtrees.xml";
	forest->save(fileName.c_str());
}
void svmTrainer::trainKNN(cv::Mat& _trainingData, cv::Mat& _trainingLabels, std::string fileName)
{
	prepareTrainTestData(_trainingData, _trainingLabels);
	//设置KNN
	knn = cv::ml::KNearest::create();
	knn->setIsClassifier(true);
	knn->setAlgorithmType(cv::ml::KNearest::BRUTE_FORCE);//cv::ml::KNearest::KDTREE,BRUTE_FORCE
	//knn->setEmax(20);//for the KDTREE
	knn->setDefaultK(15);

	knn->train(trainDataProcesser, cv::ml::StatModel::UPDATE_MODEL);

	// 训练集 / 验证集的误差
	cv::Mat results;
	float train_performance = knn->calcError(trainDataProcesser, false, results);
	float test_performance = knn->calcError(trainDataProcesser, true, results);

	std::cout << "\n[RTrees] train_performance : " << train_performance << "%" << std::endl;
	std::cout << "\n[RTrees] test_performance : " << test_performance << "%" << std::endl;

	validateSVM(knn, trainData, trainDataLabel);
	validateSVM(knn, testData, testDataLabel);

	std::cout << "\nKNN trained " << std::endl;
}
void svmTrainer::trainBoost(cv::Mat& _trainingData, cv::Mat& _trainingLabels, std::string fileName)
{
	prepareTrainTestData(_trainingData, _trainingLabels);
	//设置KNN
	boosting = cv::ml::Boost::create();
	
	std::vector<double> priors(2);
	priors[0] = 1;
	priors[1] = 25;

	boosting->setBoostType(cv::ml::Boost::Types::REAL);
	boosting->setWeakCount(100);//弱分类器
	boosting->setWeightTrimRate(0.95);//权重消减率
	boosting->setMaxDepth(5);
	boosting->setUseSurrogates(false);
	boosting->setPriors(cv::Mat(priors));
	boosting->train(trainDataProcesser);


}
void svmTrainer::loadSVM(classifier classifier_, std::string fileName)
{
	switch (classifier_)
	{
	case classifier::SVM:
	{
		svm = cv::ml::SVM::create();
		svm = cv::ml::SVM::load(fileName);
	}
	break;
	case classifier::RTrees:
	{
		forest = cv::ml::RTrees::create();
		forest = cv::ml::RTrees::load(fileName);
	}
	break;
	}

}
//这有待研究怎么用
float svmTrainer::getConfidence(float distance)
{
	float conf = 1.0 / (1.0 + exp(-1 * distance));
	return conf;
}

void svmTrainer::validateSVM(cv::Ptr<cv::ml::StatModel> classifier, cv::Mat& _testData, cv::Mat& _testLabels)
{
	float hits = 0;
	float miss = 0;
	for (size_t idx = 0; idx < _testData.rows; idx++) 
	{
		std::cout << " [test] " << "idx:" << idx 
			<< " predicted " << classifier->predict(_testData.row(idx)) 
			<< " expected "<< _testLabels.at<int>(idx, 0) 
			<< " [ confidence ] "<< getConfidence(classifier->predict(_testData.row(idx))) << std::endl;

		if (classifier->predict(_testData.row(idx)) == _testLabels.at<int>(idx, 0))
			hits++;
		else
			miss++;
	}
	printf(" [accuracy] %f \n\n", (hits / (hits + miss)));
}
void svmTrainer::predict_SVM(cv::Mat& query, std::pair<int, double>& result)
{
	result.first = svm->predict(query);
	result.second = getConfidence(svm->predict(query));
}
void svmTrainer::predict_RTrees(cv::Mat& query, std::pair<int, double> & result)
{
	result.first = forest->predict(query);
	result.second = getConfidence(forest->predict(query));
}
void svmTrainer::predict(classifier classifier_, cv::Mat& query, int& result)
{
	switch (classifier_) 
	{
	case classifier::SVM:
	{
		result = (int)svm->predict(query);
	}
	break;
	case classifier::RTrees:
	{
		result = (int)forest->predict(query);
	}
	break;
	case classifier::KNN:
	{
		result = (int)knn->predict(query);
	}
	break;
	}
}
