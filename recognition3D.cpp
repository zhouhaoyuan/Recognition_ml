#include "recognition3D.h"



recognition3D::recognition3D()
{
}


recognition3D::~recognition3D()
{
}

bool recognition3D::set_ObjectLibrary(std::vector< PointCloud::Ptr > &objectLib)
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
		}
		//更新分辨率
		resolution = pointProcess::computeResolution(tempCloudPtr);
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
		float FPFH_radius = resolution * 5;
		float SHOT_radius = resolution * 5;
		//pointProcesser.computeFeatures_FPFH(tempObj, FPFH_radius);
		//pointProcesser.computeFeatures_SHOT(tempObj, SHOT_radius);
	    //pointProcesser.computeFeatures_VFH(tempObj);
		//pointProcesser.computeFeatures_CVFH(tempObj);
		//pointProcesser.computeFeatures_ImprovedCVFH(tempObj);
		pointProcesser.computeFeatures_OUR_CVFH(tempObj);
		pointProcesser.computeFeatures_ESF(tempObj);
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
	float FPFH_radius = resolution * 5;
	float SHOT_radius = resolution * 5;
	//pointProcesser.computeFeatures_FPFH(tmpInput, FPFH_radius);
	//pointProcesser.computeFeatures_SHOT(tmpInput, SHOT_radius);
	//pointProcesser.computeFeatures_VFH(tmpInput);
	pointProcesser.computeFeatures_OUR_CVFH(tmpInput);
	//pointProcesser.computeFeatures_CVFH(tmpInput);
	//pointProcesser.computeFeatures_ImprovedCVFH(tmpInput);
	pointProcesser.computeFeatures_ESF(tmpInput);
	
	//TODO:
	int feature_size = tmpInput.getESF_features()->points.size();
	int featureDimension = tmpInput.getESF_features()->points[0].descriptorSize();// +\
		//+ tmpInput.getCVFH_features()->points[0].descriptorSize();

	std::vector<std::pair<int, double>> results;
	for (int i = 0; i < feature_size; ++i)
	{
		cv::Mat descriptor = cv::Mat::zeros(1, featureDimension, CV_32FC1);
		float* ptr = descriptor.ptr < float >(0);
		for (int j = 0; j < 640; ++j)
		{
			//TODO:
			ptr[j] = tmpInput.getESF_features()->points[i].histogram[j];
		}
		//for (int n = 640; n < featureDimension; ++n)
		//{
		//	//TODO:
		//	ptr[n] = tmpInput.getCVFH_features()->points[i].histogram[n];
		//}
		//预测
		std::pair<int, double> result;
		//TODO:
		svmTrainer_.predict(svmTrainer::classifier::SVM, descriptor, result);
		results.push_back(result);
	}
	std::sort(results.begin(), results.end(), compareConfidence);
		
	double time2 = (double)cv::getTickCount();
	std::cout << "predict time : " << (time2 - time1) * 1000 / (cv::getTickFrequency()) <<" ms \n"<< std::endl;

	//if (results[feature_size - 1].second > 0.6)
	std::cout << "[predict result] : " << results[feature_size - 1].first<<" , confidence : " << results[feature_size - 1].second << std::endl << std::endl;
		return results[feature_size - 1].first;
	//else
		//return -1;
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