#include "recognition3D.h"



recognition3D::recognition3D()
{
}


recognition3D::~recognition3D()
{
}

bool recognition3D::set_ObjectLibrary(std::vector< PointCloud::Ptr > &objectLib, bool local)
{
	std::cout << "\n\n ----------- setup the objectLibrary ---------- \n\n";

	if (objectLib.empty())
	{
		std::cout << "Error: the objectLib is empty\n";
		return false;
	}

	objectLibrary.clear();

	for (size_t i = 0; i < objectLib.size(); ++i)
	{
		std::cout << "\n ---------- the object : " << i + 1 << " ------- \n\n";

		FeatureCloud tempObj;
		PointCloud::Ptr tempCloudPtr(new PointCloud());
		pcl::copyPointCloud(*objectLib[i], *tempCloudPtr);

		std::vector<int> mapping;
		pcl::removeNaNFromPointCloud(*tempCloudPtr, *tempCloudPtr, mapping);
		//Statisstical filter
		pointProcess::StatisticalOutlierRemoval_Filter(tempCloudPtr, tempCloudPtr, 30, 1.0);
		//Resolution Calculation
		float resolution = 0.0;
		resolution = pointProcess::computeResolution(tempCloudPtr);
		//降采样获取关键点
		if (false)
		{
			float leafSize = resolution * 4;
			pointProcess::VoxelGrid_Filter(tempCloudPtr, tempCloudPtr, leafSize);
			//更新分辨率
			resolution = pointProcess::computeResolution(tempCloudPtr);
		}	
		//设置点云
		tempObj.setPointCloud(tempCloudPtr);
		//Normal Calculation
		int normal_K = 30;
		float normal_R = resolution * 3;
		pointProcesser.computeSurfaceNormals(tempObj, normal_K, normal_R);
		//设置关键点
		PointCloud::Ptr keypointCloudPtr(new PointCloud());
		pcl::copyPointCloud(*tempObj.getPointCloud(), *keypointCloudPtr);
		Normals::Ptr keypointNormalPtr(new Normals());
		pcl::copyPointCloud(*tempObj.getNormals(), *keypointNormalPtr);
		tempObj.setKeypoints(keypointCloudPtr);
		tempObj.setKeypointNormals(keypointNormalPtr);

     	//Feature describe
		resolution = pointProcess::computeResolution(tempObj.getKeypoints());

		if (local)
		{
			float FPFH_radius = resolution * 4;
			float SHOT_radius = resolution * 4;

			pointProcesser.computeFeatures_FPFH(tempObj, FPFH_radius);
			//pointProcesser.computeFeatures_SHOT(tempObj, SHOT_radius);
		}
		else
		{
			//pointProcesser.computeFeatures_VFH(tempObj);
			//pointProcesser.computeFeatures_CVFH(tempObj);
			//pointProcesser.computeFeatures_ImprovedCVFH(tempObj);
			//pointProcesser.computeFeatures_OUR_CVFH(tempObj);
			pointProcesser.computeFeatures_ESF(tempObj);
		}
	
		objectLibrary.push_back(tempObj);
	}
	return true;
}

bool recognition3D::set_ObjectLibrary_label(std::vector<std::string>& fileNames)
{
	if (fileNames.empty())
	{
		std::cout << "Error : the fileNames vector is empty!\n";
		return false;
	}
	for (size_t i = 0; i < fileNames.size(); ++i)
	{
		std::string tmp_string = fileNames[i];
		int nposBEGIN = tmp_string.find_last_of("_");
		int nposEND = tmp_string.find_last_of(".");
		std::string tmp_num_string = tmp_string.substr(nposBEGIN + 1, nposEND - nposBEGIN - 1);
		int tmp_num = (int)atof(tmp_num_string.c_str());//atof()把char*字符串转换成浮点数
		std::cout << "ObjectFile: " << fileNames[i] << "  ---  " << tmp_num << std::endl;

		objectLib_labelVec.push_back(tmp_num);
	}
	return true;
}


//旋转创造数据集
void recognition3D::rotateModelToDataSet(PointCloud::Ptr input,
	int angle, Axis axis,
	std::vector<PointCloud::Ptr> outputList, std::string path, std::string filename)
{

	pcl::MomentOfInertiaEstimation <PointT> feature_extractor;
	feature_extractor.setInputCloud(input);
	feature_extractor.compute();
	pcl::PointXYZ min_point_AABB;
	pcl::PointXYZ max_point_AABB;
	Eigen::Vector3f major_vector, middle_vector, minor_vector;
	Eigen::Vector3f mass_center;

	feature_extractor.getAABB(min_point_AABB, max_point_AABB);
	feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);
	feature_extractor.getMassCenter(mass_center);

	std::cout << "major_vector\n" << major_vector << std::endl;
	std::cout << "middle_vector\n" << middle_vector << std::endl;
	std::cout << "minor_vector\n" << minor_vector << std::endl;

	Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();

	switch (axis)
	{
	case Axis::x :
	{
		Eigen::Vector3f rotationAxis = major_vector;// + mass_center;
		std::cout << "rotationAxis\n" << rotationAxis << std::endl;
		for (int angle_ = 0; angle_ < 180; angle_ += angle)
		{
			float square = sqrt(rotationAxis(0)*rotationAxis(0) + rotationAxis(1)*rotationAxis(1) + rotationAxis(2)*rotationAxis(2));
			rotationAxis(0) /= square;
			rotationAxis(1) /= square;
			rotationAxis(2) /= square;

			float theta = angle_ * CV_PI / 180;

			rotation(0, 0) = cos(theta) + rotationAxis(0) * rotationAxis(0) * (1 - cos(theta));
			rotation(0, 1) = (1 - cos(theta))*rotationAxis(0)*rotationAxis(1) - rotationAxis(2)*sin(theta);
			rotation(0, 2) = rotationAxis(1)*sin(theta) + rotationAxis(0)*rotationAxis(2)*(1 - cos(theta));
			rotation(1, 0) = rotationAxis(0)*rotationAxis(1)*(1 - cos(theta)) + rotationAxis(2)*sin(theta);
			rotation(1, 1) = cos(theta) + pow(rotationAxis(1), 2) * (1 - cos(theta));
			rotation(1, 2) = rotationAxis(1)*rotationAxis(2)*(1 - cos(theta)) - rotationAxis(0)*sin(theta);
			rotation(2, 0) = -rotationAxis(1)*sin(theta) + rotationAxis(0)*rotationAxis(2)*(1 - cos(theta));
			rotation(2, 1) = rotationAxis(1)*rotationAxis(2)*(1 - cos(theta)) + rotationAxis(0)*sin(theta);
			rotation(2, 2) = cos(theta) + pow(rotationAxis(2), 2) * (1 - cos(theta));

			PointCloud::Ptr tmp(new PointCloud());
			pcl::copyPointCloud(*input, *tmp);
			Eigen::Affine3f affine(rotation);
			pointProcess::remove_Centroid(tmp, mass_center);
			pcl::transformPointCloud(*tmp, *tmp, affine);
			outputList.push_back(tmp);
		}
		std::cout << "rotation x\n" << rotation << std::endl;
	}
	break;
	case Axis::y:
	{
		Eigen::Vector3f rotationAxis = middle_vector;// +mass_center;
		std::cout << "rotationAxis\n" << rotationAxis << std::endl;
		for (int angle_ = 0; angle_ < 180; angle_ += angle)
		{
			float square = sqrt(rotationAxis(0)*rotationAxis(0) + rotationAxis(1)*rotationAxis(1) + rotationAxis(2)*rotationAxis(2));
			rotationAxis(0) /= square;
			rotationAxis(1) /= square;
			rotationAxis(2) /= square;

			float theta = angle_ * CV_PI / 180 ;

			rotation(0, 0) = cos(theta) + rotationAxis(0) * rotationAxis(0) * (1 - cos(theta));
			rotation(0, 1) = (1 - cos(theta))*rotationAxis(0)*rotationAxis(1) - rotationAxis(2)*sin(theta);
			rotation(0, 2) = rotationAxis(1)*sin(theta) + rotationAxis(0)*rotationAxis(2)*(1 - cos(theta));
			rotation(1, 0) = rotationAxis(0)*rotationAxis(1)*(1 - cos(theta)) + rotationAxis(2)*sin(theta);
			rotation(1, 1) = cos(theta) + pow(rotationAxis(1), 2) * (1 - cos(theta));
			rotation(1, 2) = rotationAxis(1)*rotationAxis(2)*(1 - cos(theta)) - rotationAxis(0)*sin(theta);
			rotation(2, 0) = -rotationAxis(1)*sin(theta) + rotationAxis(0)*rotationAxis(2)*(1 - cos(theta));
			rotation(2, 1) = rotationAxis(1)*rotationAxis(2)*(1 - cos(theta)) + rotationAxis(0)*sin(theta);
			rotation(2, 2) = cos(theta) + pow(rotationAxis(2), 2) * (1 - cos(theta));

			PointCloud::Ptr tmp(new PointCloud());
			pcl::copyPointCloud(*input, *tmp);
			Eigen::Affine3f affine(rotation);
			pointProcess::remove_Centroid(tmp, mass_center);
			pcl::transformPointCloud(*tmp, *tmp, affine);
			outputList.push_back(tmp);
		}
		std::cout << "rotation y\n" << rotation << std::endl;
	}
	break;
	case Axis::z:
	{
		Eigen::Vector3f rotationAxis = minor_vector;// +mass_center;
		std::cout << "rotationAxis\n" << rotationAxis << std::endl;
		for (int angle_ = 0; angle_ < 180; angle_ += angle)
		{
			float square = sqrt(rotationAxis(0)*rotationAxis(0) + rotationAxis(1)*rotationAxis(1) + rotationAxis(2)*rotationAxis(2));
			rotationAxis(0) /= square;
			rotationAxis(1) /= square;
			rotationAxis(2) /= square;

			float theta = angle_ * CV_PI / 180;

			rotation(0, 0) = cos(theta) + rotationAxis(0) * rotationAxis(0) * (1 - cos(theta));
			rotation(0, 1) = (1 - cos(theta))*rotationAxis(0)*rotationAxis(1) - rotationAxis(2)*sin(theta);
			rotation(0, 2) = rotationAxis(1)*sin(theta) + rotationAxis(0)*rotationAxis(2)*(1 - cos(theta));
			rotation(1, 0) = rotationAxis(0)*rotationAxis(1)*(1 - cos(theta)) + rotationAxis(2)*sin(theta);
			rotation(1, 1) = cos(theta) + pow(rotationAxis(1), 2) * (1 - cos(theta));
			rotation(1, 2) = rotationAxis(1)*rotationAxis(2)*(1 - cos(theta)) - rotationAxis(0)*sin(theta);
			rotation(2, 0) = -rotationAxis(1)*sin(theta) + rotationAxis(0)*rotationAxis(2)*(1 - cos(theta));
			rotation(2, 1) = rotationAxis(1)*rotationAxis(2)*(1 - cos(theta)) + rotationAxis(0)*sin(theta);
			rotation(2, 2) = cos(theta) + pow(rotationAxis(2), 2) * (1 - cos(theta));

			PointCloud::Ptr tmp(new PointCloud());
			pcl::copyPointCloud(*input, *tmp);
			Eigen::Affine3f affine(rotation);
			pointProcess::remove_Centroid(tmp, mass_center);
			pcl::transformPointCloud(*tmp, *tmp, affine);
			outputList.push_back(tmp);
		}
		std::cout << "rotation z\n" << rotation << std::endl;
	}
	break;
	}
	pcl::PLYWriter writer;
	int beginIdx = filename.find_last_of("\\");
	int middleIdx = filename.find_last_of("_");
	int endIdx = filename.find_last_of(".");
	std::string name = filename.substr(beginIdx+1, middleIdx - beginIdx - 1);
	std::string label = filename.substr(middleIdx, endIdx - middleIdx );

	std::string xyz;
	if (axis == recognition3D::Axis::x)
		xyz = "x";
	else if (axis == recognition3D::Axis::y)
		xyz = "y";
	else
		xyz = "z";

	for (size_t num = 0; num < outputList.size(); ++num)
	{
		std::stringstream ss;
		ss << num ;
		
		writer.write(path +"\\"+ name+ "_"+ xyz + ss.str()+ label + ".ply", *outputList[num]);
	}
}

//cvfh转化为cvMat
cv::Mat recognition3D::cvfh_to_cvMat(FeatureCloud& cloud)
{
	int size = cloud.getCVFH_features()->points.size();
	int featureSize = cloud.getCVFH_features()->points[0].descriptorSize();

	cv::Mat descriptor;
	for (int i = 0; i < size; ++i)
	{
		cv::Mat tmpDesc = cv::Mat::zeros(1, featureSize, CV_32FC1);
		float* ptr = tmpDesc.ptr<float>(0);
		for (int j = 0; j < 308; ++j)
		{
			ptr[j] = cloud.getCVFH_features()->points[i].histogram[j];
		}
		descriptor.push_back(tmpDesc);
	}
	return descriptor;
}
//esf转化为cvMat
cv::Mat recognition3D::esf_to_cvMat(FeatureCloud& cloud)
{
	int size = cloud.getESF_features()->points.size();
	int featureSize = cloud.getESF_features()->points[0].descriptorSize();

	cv::Mat descriptor;
	for (int i = 0; i < size; ++i)
	{
		cv::Mat tmpDesc = cv::Mat::zeros(1, featureSize, CV_32FC1);
		float* ptr = tmpDesc.ptr<float>(0);
		for (int j = 0; j < 640; ++j)
		{
			ptr[j] = cloud.getESF_features()->points[i].histogram[j];
		}
		descriptor.push_back(tmpDesc);
		std::cout << " cloud.getESF_features() descriptor size : " << descriptor.rows << std::endl;
	}
	return descriptor;
}
//ourcvfh+esf 转化为cvMat
cv::Mat recognition3D::ourcvfh_and_esf_to_cvMat(FeatureCloud& cloud)
{
	int size = cloud.getESF_features()->points.size();
	int featureSize = cloud.getESF_features()->points[0].descriptorSize() + \
		cloud.getCVFH_features()->points[0].descriptorSize();

	cv::Mat descriptor;
	for (int i = 0; i < size; ++i)
	{
		cv::Mat tmpDesc = cv::Mat::zeros(1, featureSize, CV_32FC1);
		float* ptr = tmpDesc.ptr<float>(0);
		for (int j = 0; j < 640; ++j)
		{
			ptr[j] = cloud.getESF_features()->points[i].histogram[j];
		}
		for (int n = 640; n < featureSize; ++n)
		{
			ptr[n] = cloud.getCVFH_features()->points[i].histogram[n];
		}
		descriptor.push_back(tmpDesc);
	}
	return descriptor;
}
bool recognition3D::set_svm_train_model()
{
	if (objectLibrary.empty() || objectLib_labelVec.empty())
	{
		std::cout << "\nError: set_svm_train_model() , the objectLibrary or objectLibrary_label is empty!\n";
		return false;
	}

	int objectLibSize = objectLibrary.size();

	for (size_t i = 0; i < objectLibSize; ++i)
	{	
		bool vfh_flag = false;
		bool cvfh_flag = false;
		bool esf_flag = true;
		bool cvfh_esf_flag = false;

		//VFH特征
		if (vfh_flag)
		{
			cv::Mat label = cv::Mat::zeros(1, 1, CV_32SC1);
			cv::Mat descriptor = cv::Mat::zeros(1, 308, CV_32FC1);

			label.at<int>(0, 0) = objectLib_labelVec[i];
			descriptor = svmTrainer_.vector2Mat(objectLibrary[i].getVFH_features());

			objectLibrary_sample.push_back(descriptor);
			objectLibrary_label.push_back(label);
		}
		if (cvfh_flag)
		{			
			cv::Mat descriptor = cvfh_to_cvMat(objectLibrary[i]);
			int size = descriptor.rows;
			cv::Mat label = cv::Mat::zeros(size, 1, CV_32SC1);
			cv::Mat tmplabel = cv::Mat::zeros(1, 1, CV_32SC1);
			tmplabel.at<int>(0, 0) = objectLib_labelVec[i];
			for (int j = 0; j < size; ++j)
			{				
				tmplabel.copyTo(label.row(j));
			}
			objectLibrary_sample.push_back(descriptor);
			objectLibrary_label.push_back(label);
		}
		if (esf_flag)
		{
			cv::Mat descriptor = esf_to_cvMat(objectLibrary[i]);
			int size = descriptor.rows;
			cv::Mat label = cv::Mat::zeros(size, 1, CV_32SC1);
			cv::Mat tmplabel = cv::Mat::zeros(1, 1, CV_32SC1);
			tmplabel.at<int>(0, 0) = objectLib_labelVec[i];
			for (int j = 0; j < size; ++j)
			{
				tmplabel.copyTo(label.row(j));
			}
			objectLibrary_sample.push_back(descriptor);
			objectLibrary_label.push_back(label);
		}

		//OUR-CVFH + ESF
		if (cvfh_esf_flag)
		{
			cv::Mat descriptor = ourcvfh_and_esf_to_cvMat(objectLibrary[i]);
			int size = descriptor.rows;
			cv::Mat label = cv::Mat::zeros(size, 1, CV_32SC1);
			cv::Mat tmplabel = cv::Mat::zeros(1, 1, CV_32SC1);
			tmplabel.at<int>(0, 0) = objectLib_labelVec[i];
			for (int j = 0; j < size; ++j)
			{
				tmplabel.copyTo(label.row(j));
			}
			objectLibrary_sample.push_back(descriptor);
			objectLibrary_label.push_back(label);
		}
		

	}
	//std::cout << objectLibrary_sample << std::endl;
	//std::cout << objectLibrary_label << std::endl;
}

void recognition3D::svm_classifier_train(std::string savefilename)
{
	svmTrainer_.trainSVM(objectLibrary_sample, objectLibrary_label, savefilename);
	//svmTrainer_.trainRTrees(objectLibrary_sample, objectLibrary_label, savefilename);
	//svmTrainer_.trainKNN(objectLibrary_sample, objectLibrary_label, savefilename);
}
int recognition3D::svm_predict(PointCloud::Ptr input)
{
	std::cout << "\n>>>>>>>>>>>>>>>>>  Object to be classifiered  <<<<<<<<<<<<<<<<<<\n";
	double time1 = (double)cv::getTickCount();
	//更新分辨率
	float resolution = pointProcess::computeResolution(input);

	FeatureCloud tmpInput;
	tmpInput.setPointCloud(input);
	//Normal Calculation
	int normal_K = 30;
	float normal_R = resolution * 3;
	pointProcesser.computeSurfaceNormals(tmpInput, normal_K, normal_R);
	//设置关键点
	PointCloud::Ptr keypointCloudPtr(new PointCloud());
	pcl::copyPointCloud(*tmpInput.getPointCloud(), *keypointCloudPtr);
	Normals::Ptr keypointNormalPtr(new Normals());
	pcl::copyPointCloud(*tmpInput.getNormals(), *keypointNormalPtr);
	tmpInput.setKeypoints(keypointCloudPtr);
	tmpInput.setKeypointNormals(keypointNormalPtr);

	//改进的VFH
	if (false)
	{
		PointCloud::Ptr tmpCloud(new PointCloud);
		pcl::copyPointCloud(*tmpInput.getKeypoints(), *tmpCloud);
		resolution = pointProcesser.computeResolution(tmpCloud);
		pcl::PointIndices filterIndices;
		pointProcesser.Uniform_Filter(tmpInput.getKeypoints(), tmpCloud, resolution * 8, filterIndices);
		PointCloudNormal::Ptr normals_cloud(new PointCloudNormal());
		normals_cloud->width = static_cast<uint32_t> (tmpInput.getKeypoints()->points.size());
		normals_cloud->height = 1;
		normals_cloud->points.resize(normals_cloud->width);
		for (size_t i = 0; i < tmpInput.getKeypoints()->points.size(); ++i)
		{
			normals_cloud->points[i].x = tmpInput.getKeypoints()->points[i].x;
			normals_cloud->points[i].y = tmpInput.getKeypoints()->points[i].y;
			normals_cloud->points[i].z = tmpInput.getKeypoints()->points[i].z;
		}
		pcl::search::KdTree<pcl::PointNormal>::Ptr normals_tree(new pcl::search::KdTree<pcl::PointNormal>(false));
		normals_tree->setInputCloud(normals_cloud);
		pcl::NormalEstimationOMP<PointNormalT, PointNormalT> n3d;
		n3d.setRadiusSearch(resolution * 3);
		n3d.setSearchMethod(normals_tree);
		n3d.setInputCloud(normals_cloud);
		n3d.compute(*normals_cloud);

		pcl::PointIndices indices;
		for (int i = 0; i < normals_cloud->points.size(); ++i)
		{
			std::vector<int> indice;
			std::vector<float> dist;
			normals_tree->nearestKSearch(normals_cloud->points[i], 1, indice, dist);
			indices.indices.push_back(indice[0]);
		}
		indices.indices.erase(filterIndices.indices.begin(), filterIndices.indices.end());

		Eigen::Vector4f xyz_centroid;
		Eigen::Vector4f normal_centroid;
		pcl::PointCloud<pcl::VFHSignature308>::Ptr vfh(new pcl::PointCloud<pcl::VFHSignature308>);
		pcl::compute3DCentroid(*tmpCloud, xyz_centroid);
		pointProcesser.computeImproveVFHfeature(normals_cloud, indices, xyz_centroid, normal_centroid, *vfh);
		tmpInput.setVFH_features(vfh);
	}


	//Feature describe
	resolution = pointProcess::computeResolution(tmpInput.getKeypoints());
	float FPFH_radius = resolution * 4;
	float SHOT_radius = resolution * 4;
	//pointProcesser.computeFeatures_FPFH(tmpInput, FPFH_radius);
	//pointProcesser.computeFeatures_SHOT(tmpInput, SHOT_radius);
	//pointProcesser.computeFeatures_VFH(tmpInput);
	//pointProcesser.computeFeatures_OUR_CVFH(tmpInput);
	//pointProcesser.computeFeatures_CVFH(tmpInput);
	//pointProcesser.computeFeatures_ImprovedCVFH(tmpInput);
	pointProcesser.computeFeatures_ESF(tmpInput);

	//pcl::visualization::PCLPlotter* plotter = new pcl::visualization::PCLPlotter();
	//plotter->setShowLegend(true);
	//plotter->addFeatureHistogram<pcl::VFHSignature308>(*tmpInput.getCVFH_features(), "vfh", 0, "OUR-CVFH");
	//plotter->setWindowSize(600, 400);
	//plotter->spinOnce(10);
	
	//TODO:
	int feature_size = tmpInput.getESF_features()->points.size();
	int featureDimension = tmpInput.getESF_features()->points[0].descriptorSize();

	//int feature_size = tmpInput.getCVFH_features()->points.size();
	//int featureDimension = tmpInput.getCVFH_features()->points[0].descriptorSize();

	std::vector<int> results;
	for (int i = 0; i < feature_size; ++i)
	{
		cv::Mat descriptor = cv::Mat::zeros(1, featureDimension, CV_32FC1);
		float* ptr = descriptor.ptr < float >(0);
		for (int j = 0; j < featureDimension; ++j)
		{
			//TODO:
			//ptr[j] = tmpInput.getCVFH_features()->points[i].histogram[j];
			ptr[j] = tmpInput.getESF_features()->points[i].histogram[j];
		}
		//预测
		int result;
		//TODO:
		svmTrainer_.predict(svmTrainer::classifier::SVM, descriptor, result);
		results.push_back(result);
		//std::cout << "result : "<< result << std::endl;
	}
	
	std::map<int, int> myM;
	for (int i = 0; i < results.size(); ++i)
		++myM[results[i]];
	std::vector<std::pair<int, int>> tmpM;
	for (std::map<int, int>::iterator curr = myM.begin(); curr != myM.end(); ++curr)
	{
		tmpM.push_back(std::make_pair(curr->first, curr->second));
		//std::cout << curr->first << "  ,  " << curr->second << std::endl;
	}
		
	std::sort(tmpM.begin(), tmpM.end(), probcmp);

	if (tmpM.size() > 1 && tmpM[0].second == tmpM[1].second)
	{
		std::cout << "[predict failure] !!! " << std::endl;
		return -1;
	}
	
	double time2 = (double)cv::getTickCount();
	std::cout << "[predict time] : " << (time2 - time1) * 1000 / (cv::getTickFrequency()) << " ms" << std::endl;
	std::cout << "[predict result] : " << tmpM[0].first << std::endl << std::endl;

	return tmpM[0].first;
}

void recognition3D::load_svmClassifier(svmTrainer::classifier classifier_, std::string file)
{
	switch (classifier_)
	{
	case svmTrainer::classifier::SVM:
	{
		svmTrainer_.loadSVM(svmTrainer::classifier::SVM, file);
	}
	break;
	case svmTrainer::classifier::RTrees:
	{
		svmTrainer_.loadSVM(svmTrainer::classifier::RTrees, file);
	}
	break;
	}
	
}

//计算转换矩阵
float recognition3D::RotationTranslationCompute(
	int targetIndex,
	FeatureCloud& cloudsource,
	Eigen::Matrix4f &tranResult)
{
	if (objectLibrary.empty())
	{
		std::cout << "Error: the objectLibrary is empty()" << std::endl;
		return false;
	}
	if (targetIndex < 0 || targetIndex > 5)
	{
		std::cout << "Error: the targetIndex is wrong" << std::endl;
		return false;
	}
	FeatureCloud cloudtarget = objectLibrary[targetIndex];
	float resolution = pointProcess::computeResolution(cloudtarget.getKeypoints());
	// Correspondence Estimation
	pcl::Correspondences all_correspondences;//剔除前
	pcl::Correspondences tuple_inliers;//tuple原则剔除后
	pcl::Correspondences sac_inliers;//sac原则剔除后

	pointProcess::correspondence_estimation(cloudsource.getFPFH_features(),
		cloudtarget.getFPFH_features(),
		all_correspondences);
	std::cout << "correspondence_estimation size : " << all_correspondences.size() << std::endl;
	//约束去除错误点对
	float tupleScale = 0.9;
	int tuple_max_cnt_ = 1500;
	pointProcess::advancedMatching(cloudtarget.getKeypoints(), cloudsource.getKeypoints(),
		all_correspondences, tuple_inliers, tupleScale, tuple_max_cnt_);
	std::cout << "tuple_rejection size : " << tuple_inliers.size() << std::endl;

	pointProcess::correspondences_rejection(cloudsource.getKeypoints(), 
		cloudtarget.getKeypoints(),
		tuple_inliers, sac_inliers,
		60, resolution * 3);//0 1 为 50， 
	std::cout << "sac_rejection size : " << sac_inliers.size() << std::endl;
	if (false)
	{
		pointProcess::showPointCloudCorrespondences("all_correspondences", cloudtarget.getKeypoints(),
			cloudsource.getKeypoints(), all_correspondences, 200);
		pointProcess::showPointCloudCorrespondences("inliers", cloudtarget.getKeypoints(),
			cloudsource.getKeypoints(), sac_inliers, 200);
	}
	//根据匹配点对重新确立关键点
	FeatureCloud targetCloud_Keypoint, sourceCloud_Keypoint;
	PointCloud::Ptr targetKeypoint_(new PointCloud);
	PointCloud::Ptr sourceKeypoint_(new PointCloud);
	Normals::Ptr targetKeypointNormal_(new Normals);
	Normals::Ptr sourceKeypointNormal_(new Normals);
	FPFH_features::Ptr targetKeypointFPFH_(new FPFH_features);
	FPFH_features::Ptr sourceKeypointFPFH_(new FPFH_features);
	
	for (size_t i = 0; i < sac_inliers.size(); ++i)
	{
		PointT source = cloudsource.getKeypoints()->at(sac_inliers[i].index_query);
		PointT target = cloudtarget.getKeypoints()->at(sac_inliers[i].index_match);

		NormalT sourceNormal = cloudsource.getKeypointNormals()->at(sac_inliers[i].index_query);
		NormalT targetNormal = cloudtarget.getKeypointNormals()->at(sac_inliers[i].index_match);

		FPFH33_feature sourceFPFH = cloudsource.getFPFH_features()->at(sac_inliers[i].index_query);
		FPFH33_feature targetFPFH = cloudtarget.getFPFH_features()->at(sac_inliers[i].index_match);

		targetKeypoint_->points.push_back(target);
		sourceKeypoint_->points.push_back(source);

		targetKeypointNormal_->points.push_back(targetNormal);
		sourceKeypointNormal_->points.push_back(sourceNormal);

		targetKeypointFPFH_->points.push_back(targetFPFH);
		sourceKeypointFPFH_->points.push_back(sourceFPFH);
	}
	targetCloud_Keypoint.setKeypoints(targetKeypoint_);
	sourceCloud_Keypoint.setKeypoints(sourceKeypoint_);
	targetCloud_Keypoint.setKeypointNormals(targetKeypointNormal_);
	sourceCloud_Keypoint.setKeypointNormals(sourceKeypointNormal_);
	targetCloud_Keypoint.setFPFH_features(targetKeypointFPFH_);
	sourceCloud_Keypoint.setFPFH_features(sourceKeypointFPFH_);
	//Construct PointNormal建立法向量点云
	pointProcess::construct_PointNormal(cloudtarget, cloudsource);
	pointProcess::construct_PointNormal(targetCloud_Keypoint, sourceCloud_Keypoint);

	//PCL函数建立点云法向量
	//PointCloudNormal::Ptr pointNormal_src(new PointCloudNormal);
	//PointCloudNormal::Ptr pointNormal_tgt(new PointCloudNormal);
	//pcl::concatenateFields(*targetKeypoint_, *targetKeypointNormal_, *pointNormal_tgt);
	//pcl::concatenateFields(*sourceKeypoint_, *sourceKeypointNormal_, *pointNormal_src);
	//targetCloud_Keypoint.setPointCloudNormals(pointNormal_tgt);
	//sourceCloud_Keypoint.setPointCloudNormals(pointNormal_src);

	Eigen::Matrix4f tran = Eigen::Matrix4f::Identity();

	resolution = pointProcess::computeResolution(targetCloud_Keypoint.getKeypoints());
	float minsampleDistance = resolution * 2;
	int numofSample = sac_inliers.size() / 5;
	int correspondenceRandomness = 30;
	pointProcess::SAC_IA_Transform(sourceCloud_Keypoint, targetCloud_Keypoint, minsampleDistance,
		numofSample, correspondenceRandomness, tran);

	float transEps = 1e-10;//设置两次变化矩阵之间的差值（一般设置为1e-10即可）
	float maxCorresDist = 0.7;//设置对应点对之间的最大距离（此值对配准结果影响较大）
	float EuclFitEps = 0.0001;//设置收敛条件是均方误差和小于阈值,停止迭代；
	float outlThresh = resolution * 2.5;
	int maxIteration = 60;
	float scoreICP = pointProcess::iterative_closest_points("SVD", false, false,
		sourceCloud_Keypoint, targetCloud_Keypoint,
		transEps, maxCorresDist, EuclFitEps,
		outlThresh, maxIteration, tran);

	tranResult = tran;
	return scoreICP;
}