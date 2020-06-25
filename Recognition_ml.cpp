#include "FeatureCloud.h"
#include "pointProcess.h"
#include "svmTrainer.h"
#include "recognition3D.h"
#include <iostream>
#include <utility>　　//pair的头文件

#include "ppf3d_public.h"

#include "cloudProcess.h"

//获取文件名
bool computePairNum(std::pair<double, std::string> pair1,
	std::pair<double, std::string> pair2)
{
	return pair1.first < pair2.first;
}
//文件名排序
void sort_filelists(std::vector<std::string>& filists)
{
	if (filists.empty())
		return;
	std::vector<std::pair<double, std::string> > filelists_pair;

	for (int i = 0; i < filists.size(); ++i) {

		std::string tmp_string = filists[i];
		int nposBEGIN = tmp_string.find_last_of("_");
		int nposEND = tmp_string.find_last_of(".");
		std::string tmp_num_string = tmp_string.substr(nposBEGIN + 1, nposEND);
		double tmp_num = atof(tmp_num_string.c_str());//atof()把字符串转换成浮点数
		std::pair<double, std::string> tmp_pair;
		tmp_pair.first = tmp_num;
		tmp_pair.second = tmp_string;
		filelists_pair.push_back(tmp_pair);
	}
	std::sort(filelists_pair.begin(), filelists_pair.end(), computePairNum);
	filists.clear();
	for (int i = 0; i < filelists_pair.size(); ++i)
	{
		filists.push_back(filelists_pair[i].second);
		std::cout << filists[i] << std::endl;
	}
}
//获取特定格式点云文件名
void getFiles(std::string path, std::string ext, std::vector<std::string>& files)
{
	//文件句柄    
	intptr_t  hFile = 0;
	//文件信息    
	struct _finddata_t fileinfo;
	std::string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之    
			//如果不是,加入列表    
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), ext, files);
			}
			else
			{
				std::string s = p.assign(path).append("\\").append(fileinfo.name);  //获取此文件的完整路径  
				char fileDrive[_MAX_DRIVE];
				char fileDir[_MAX_DIR];
				char fileName[_MAX_FNAME];
				char fileExt[_MAX_EXT];
				_splitpath(s.c_str(), fileDrive, fileDir, fileName, fileExt);  //将完整路径分解  
				if (strcmp(fileExt, ext.c_str()) == 0)  //筛选出符合后缀条件的文件  
				{
					std::string ss = p.assign(path).append("\\").append(fileinfo.name);
					files.push_back(ss);
				}

			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
};

//模拟点云拼接
void pointCloudRegistration(PointCloud::Ptr pointcloud)
{
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

	pointProcess pointCloudProcesser;
	pointCloudProcesser.SACSegmentation_model(pointcloud, coefficients, inliers, pcl::SACMODEL_PLANE);

	Eigen::Vector3f normalVec;
	normalVec.setZero();
	normalVec(0) = coefficients->values[0];
	normalVec(1) = coefficients->values[1];
	normalVec(2) = coefficients->values[2];

	std::cout << "the normalVec : \n" << normalVec << std::endl;

	Eigen::Vector3f Zaxis(0, 0, 1);
	Eigen::Matrix3f rotation = pointProcess::computeRotation(normalVec, Zaxis);

	Eigen::Affine3f affine_R(rotation);
	pcl::transformPointCloud(*pointcloud, *pointcloud, affine_R);

	Eigen::Vector3f centroid;
	pointProcess::computeCentroid(pointcloud, centroid);
	pointProcess::remove_Centroid(pointcloud, centroid);

	pcl::MomentOfInertiaEstimation <PointT> feature_extractor;
	feature_extractor.setInputCloud(pointcloud);
	feature_extractor.compute();
	std::vector <float> moment_of_inertia;
	std::vector <float> eccentricity;
	pcl::PointXYZ min_point_AABB;
	pcl::PointXYZ max_point_AABB;
	pcl::PointXYZ min_point_OBB;
	pcl::PointXYZ max_point_OBB;
	pcl::PointXYZ position_OBB;
	Eigen::Matrix3f rotational_matrix_OBB;
	float major_value, middle_value, minor_value;
	Eigen::Vector3f major_vector, middle_vector, minor_vector;
	Eigen::Vector3f mass_center;

	feature_extractor.getMomentOfInertia(moment_of_inertia);
	feature_extractor.getEccentricity(eccentricity);
	feature_extractor.getAABB(min_point_AABB, max_point_AABB);
	feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
	feature_extractor.getEigenValues(major_value, middle_value, minor_value);
	feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);
	feature_extractor.getMassCenter(mass_center);

	pcl::PLYWriter writer;

	std::vector<PointCloud::Ptr> y_cloudlist;
	int distance = max_point_AABB.y - min_point_AABB.y;
	int y_segmentNum = 3;
	int x_segmentNum = 3;//<10
	float perDist = (float)distance / y_segmentNum;
	for (unsigned int i = 0; i < y_segmentNum; i++)
	{
		PointCloud::Ptr tmpCloud(new PointCloud());
		pcl::PassThrough<PointT> passfilter;
		passfilter.setInputCloud(pointcloud);
		passfilter.setFilterFieldName("y");
		passfilter.setFilterLimits(min_point_AABB.y + perDist * i, min_point_AABB.y + perDist * (i + 1));
		std::vector<int> indices;
		passfilter.filter(indices);
		pcl::copyPointCloud(*pointcloud, indices, *tmpCloud);
		y_cloudlist.push_back(tmpCloud);

		std::stringstream ss;
		if (i == 0)
			ss << 0 << 0;
		else
			ss << i * 10;
		writer.write("F:/vs_program/PCLstudy/Recognition_SVM/cloud/cloud_" + ss.str() + ".ply", *tmpCloud);
	}

	distance = max_point_AABB.x - min_point_AABB.x;
	perDist = (float)distance / x_segmentNum;
	for (int n = 0; n < y_cloudlist.size(); ++n)
	{
		if (n % 2 == 0)
		{
			int count = 0;
			for (unsigned int i = 0; i < x_segmentNum; i++)
			{
				PointCloud::Ptr tmpCloud(new PointCloud());
				pcl::PassThrough<PointT> passfilter;
				passfilter.setInputCloud(y_cloudlist[n]);
				passfilter.setFilterFieldName("x");
				passfilter.setFilterLimits(min_point_AABB.x + perDist * i, min_point_AABB.x + perDist * (i + 1));
				std::vector<int> indices;
				passfilter.filter(indices);
				pcl::copyPointCloud(*y_cloudlist[n], indices, *tmpCloud);

				if (tmpCloud->points.empty())
					continue;

				std::stringstream ss;
				ss << n << count;
				count++;
				writer.write("F:/vs_program/PCLstudy/Recognition_SVM/cloud/cloud_" + ss.str() + ".ply", *tmpCloud);
			}
		}
		else
		{
			int count = 0;
			std::cout << std::endl;
			for (int i = x_segmentNum - 1; i >= 0; i--)
			{
				PointCloud::Ptr tmpCloud(new PointCloud());
				pcl::PassThrough<PointT> passfilter;
				passfilter.setInputCloud(y_cloudlist[n]);
				passfilter.setFilterFieldName("x");
				passfilter.setFilterLimits(min_point_AABB.x + perDist * i, min_point_AABB.x + perDist * (i + 1));
				std::vector<int> indices;
				passfilter.filter(indices);
				pcl::copyPointCloud(*y_cloudlist[n], indices, *tmpCloud);

				if (tmpCloud->points.empty())
					continue;

				std::stringstream ss;
				ss << n << count;
				count++;
				writer.write("F:/vs_program/PCLstudy/Recognition_SVM/cloud/cloud_" + ss.str() + ".ply", *tmpCloud);

			}
		}
	}

	pcl::visualization::PCLVisualizer viewer("viewer");
	viewer.setBackgroundColor(0, 0, 0);
	viewer.addCoordinateSystem(0.5);
	//viewer.initCameraParameters();
	pcl::visualization::PointCloudColorHandlerGenericField<PointT> color(pointcloud, "z"); // 按照z字段进行渲染
	viewer.addPointCloud(pointcloud, color, "pointcloud");

	//Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
	//Eigen::Quaternionf quat(rotational_matrix_OBB);
	//viewer.addCube(position, quat, max_point_OBB.x - min_point_OBB.x, max_point_OBB.y - min_point_OBB.y, max_point_OBB.z - min_point_OBB.z, "OBB");
	//viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
	//	pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "OBB");

	//viewer.addCube(min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y,
	//	min_point_AABB.z, max_point_AABB.z, 1.0, 1.0, 0.0, "AABB");//AABB盒子
	//viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
	//	pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "AABB");

	pcl::PointXYZ center(mass_center(0), mass_center(1), mass_center(2));
	pcl::PointXYZ x_axis(major_vector(0) + mass_center(0), major_vector(1) + mass_center(1), major_vector(2) + mass_center(2));
	pcl::PointXYZ y_axis(middle_vector(0) + mass_center(0), middle_vector(1) + mass_center(1), middle_vector(2) + mass_center(2));
	pcl::PointXYZ z_axis(minor_vector(0) + mass_center(0), minor_vector(1) + mass_center(1), minor_vector(2) + mass_center(2));
	viewer.addLine(center, x_axis, 1.0f, 0.0f, 0.0f, "major eigen vector");
	viewer.addLine(center, y_axis, 0.0f, 1.0f, 0.0f, "middle eigen vector");
	viewer.addLine(center, z_axis, 0.0f, 0.0f, 1.0f, "minor eigen vector");

	while (!viewer.wasStopped())
	{
		viewer.spinOnce(1000);
		viewer.removeAllPointClouds();

		std::vector<std::string> files;
		std::string format = ".ply";
		std::string path = "F:/vs_program/PCLstudy/Recognition_SVM/cloud";
		getFiles(path, format, files);
		sort_filelists(files);
		pcl::PLYReader reader;

		PointCloud::Ptr result(new PointCloud());
		for (int i = 0; i <= files.size(); ++i)
		{
			if (i < files.size())
			{
				if (i == 0)
					viewer.spinOnce(5000);
				//boost::this_thread::sleep(boost::posix_time::milliseconds(5000));   //随时间
				PointCloud::Ptr cloud(new PointCloud);
				reader.read<PointT>(files[i], *cloud);
				*result += *cloud;

				std::stringstream name;
				name << i;
				pcl::visualization::PointCloudColorHandlerGenericField<PointT> fildColor(cloud, "z"); // 按照z字段进行渲染
				viewer.addPointCloud(cloud, fildColor, name.str());
				viewer.spinOnce(5000);
				//boost::this_thread::sleep(boost::posix_time::milliseconds(5000));   //随时间
			}
			else
			{
				viewer.removeAllPointClouds();
				pcl::visualization::PointCloudColorHandlerGenericField<PointT> fildColor(result, "z"); // 按照z字段进行渲染
				viewer.addPointCloud(result, fildColor, "result");
				viewer.spinOnce(4000);
				//boost::this_thread::sleep(boost::posix_time::milliseconds(5000));   //随时间
			}
		}

	}
}
void calibrate_by_XY(std::vector<PointCloud::Ptr>& pointlist, Eigen::Matrix4f& rotation)
{
	if (pointlist.size() < 3)
	{
		std::cout << "Error : the pointlist is not enough !\n";
		return;
	}
	pointProcess pointCloudProcesser;
	/***************************** x 轴*********************/
	pcl::ModelCoefficients::Ptr coefficients_sphere1(new pcl::ModelCoefficients());
	pcl::PointIndices::Ptr inliers_sphere1(new pcl::PointIndices());
	pcl::ModelCoefficients::Ptr coefficients_sphere2(new pcl::ModelCoefficients());
	pcl::PointIndices::Ptr inliers_sphere2(new pcl::PointIndices());
	pointCloudProcesser.SACSegmentation_model(pointlist[0], coefficients_sphere1, inliers_sphere1, pcl::SACMODEL_SPHERE);
	pointCloudProcesser.SACSegmentation_model(pointlist[1], coefficients_sphere2, inliers_sphere2, pcl::SACMODEL_SPHERE);

	Eigen::Vector3f sphere_centroid_1, sphere_centroid_2;
	sphere_centroid_1(0) = coefficients_sphere1->values[0];
	sphere_centroid_1(1) = coefficients_sphere1->values[1];
	sphere_centroid_1(2) = coefficients_sphere1->values[2];
	sphere_centroid_2(0) = coefficients_sphere2->values[0];
	sphere_centroid_2(1) = coefficients_sphere2->values[1];
	sphere_centroid_2(2) = coefficients_sphere2->values[2];

	Eigen::Vector3f Xaxis(1, 0, 0);
	Eigen::Vector3f X_tran;
	X_tran.setZero();
	X_tran = sphere_centroid_2 - sphere_centroid_1;
	float dist = X_tran.norm();
	X_tran /= dist;
	std::cout << "Xaxis : \n" << Xaxis << std::endl;
	std::cout << "X_tran : \n" << X_tran << std::endl;

	Eigen::Matrix3f rotation_X = pointProcess::computeRotation(X_tran, Xaxis);

	Eigen::Affine3f affine_RX(rotation_X);
	PointCloud::Ptr cloud_rotated(new PointCloud());
	pcl::transformPointCloud(*pointlist[0], *cloud_rotated, affine_RX);

	/***************************** y 轴*********************/
	pcl::ModelCoefficients::Ptr coefficients_sphere3(new pcl::ModelCoefficients());
	pcl::PointIndices::Ptr inliers_sphere3(new pcl::PointIndices());

	pointCloudProcesser.SACSegmentation_model(pointlist[2], coefficients_sphere3, inliers_sphere3, pcl::SACMODEL_SPHERE);

	Eigen::Vector3f sphere_centroid_3;
	sphere_centroid_3(0) = coefficients_sphere3->values[0];
	sphere_centroid_3(1) = coefficients_sphere3->values[1];
	sphere_centroid_3(2) = coefficients_sphere3->values[2];

	pcl::transformPoint(sphere_centroid_1, sphere_centroid_1, affine_RX);
	pcl::transformPoint(sphere_centroid_3, sphere_centroid_3, affine_RX);
	std::cout << "sphere_centroid_1 : \n" << sphere_centroid_1 << std::endl;
	std::cout << "sphere_centroid_3 : \n" << sphere_centroid_3 << std::endl;

	Eigen::Vector3f Yaxis(0, 1, 0);
	Eigen::Vector3f Y_tran;
	Y_tran.setZero();
	Y_tran = sphere_centroid_3 - sphere_centroid_1;
	float distY = Y_tran.norm();
	Y_tran /= distY;
	std::cout << "Yaxis : \n" << Yaxis << std::endl;
	std::cout << "Y_tran : \n" << Y_tran << std::endl;
	float theta_Y = acos(Yaxis.dot(Y_tran) / (Yaxis.norm()*Y_tran.norm()));
	std::cout << "theta_Y : " << theta_Y << std::endl;

	Eigen::Matrix3f Y_rotation;
	Y_rotation.setIdentity();
	Y_rotation(1, 1) = (float)cos(theta_Y);
	Y_rotation(1, 2) = (float)sin(theta_Y);
	Y_rotation(2, 1) = (float)-sin(theta_Y);
	Y_rotation(2, 2) = (float)cos(theta_Y);
	std::cout << "the Y_rotation : \n" << Y_rotation << std::endl;

	Eigen::Affine3f Y_rotatin_(Y_rotation);
	PointCloud::Ptr cloud_rotated_2(new PointCloud());
	pcl::transformPointCloud(*cloud_rotated, *cloud_rotated_2, Y_rotatin_);

	/**************** 标定结果 ************/
	Eigen::Matrix3f rotation_result = Y_rotation * rotation_X;
	rotation = Eigen::Matrix4f::Identity();
	rotation.block<3, 3>(0, 0) = rotation_result;

	Eigen::Affine3f tran_R(rotation);

	PointCloud::Ptr cloud_0(new PointCloud());
	PointCloud::Ptr cloud_1(new PointCloud());
	PointCloud::Ptr cloud_2(new PointCloud());
	pcl::copyPointCloud(*pointlist[0], *cloud_0);
	pcl::copyPointCloud(*pointlist[1], *cloud_1);
	pcl::copyPointCloud(*pointlist[2], *cloud_2);
	pcl::transformPointCloud(*cloud_0, *cloud_0, tran_R);
	pcl::transformPointCloud(*cloud_1, *cloud_1, tran_R);
	pcl::transformPointCloud(*cloud_2, *cloud_2, tran_R);

	//***************************显示

	Eigen::Vector3f centroid;
	pointProcess::computeCentroid(cloud_rotated_2, centroid);
	pointProcess::remove_Centroid(cloud_rotated_2, centroid);

	pcl::MomentOfInertiaEstimation <PointT> feature_extractor;
	feature_extractor.setInputCloud(cloud_rotated_2);
	feature_extractor.compute();
	pcl::PointXYZ min_point_AABB;
	pcl::PointXYZ max_point_AABB;
	feature_extractor.getAABB(min_point_AABB, max_point_AABB);

	Eigen::Matrix4f baseMatrix;
	baseMatrix.setIdentity();
	baseMatrix.block<3, 1>(0, 3) = -centroid;
	Eigen::Affine3f baseTran(baseMatrix);
	pcl::transformPointCloud(*pointlist[0], *pointlist[0], baseTran);
	pcl::transformPointCloud(*pointlist[1], *pointlist[1], baseTran);
	pcl::transformPointCloud(*pointlist[2], *pointlist[2], baseTran);
	pcl::transformPointCloud(*cloud_0, *cloud_0, baseTran);
	pcl::transformPointCloud(*cloud_1, *cloud_1, baseTran);
	pcl::transformPointCloud(*cloud_2, *cloud_2, baseTran);

	pcl::visualization::PCLVisualizer viewer("viewer");
	viewer.setBackgroundColor(0, 0, 0);
	viewer.addCoordinateSystem(50.0);
	pcl::visualization::PointCloudColorHandlerGenericField<PointT> color(pointlist[0], "z");
	viewer.addPointCloud(pointlist[0], color, "pointlist0");
	viewer.addPointCloud(pointlist[1], color, "*pointlist1");
	viewer.addPointCloud(pointlist[2], color, "*pointlist2");
	viewer.addPointCloud(cloud_0, color, "cloud_0");
	viewer.addPointCloud(cloud_1, color, "*cloud_1");
	viewer.addPointCloud(cloud_2, color, "*cloud_2");

	viewer.addCube(min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y,
		min_point_AABB.z, max_point_AABB.z, 1.0, 1.0, 0.0, "AABB");//AABB盒子

	viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
		pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "AABB");

	while (!viewer.wasStopped())
	{
		viewer.spinOnce(10);
	}

}
void calibrate_by_plane_X(std::vector<PointCloud::Ptr>& pointlist, Eigen::Matrix4f& rotation_)
{
	if (pointlist.size() < 3)
	{
		std::cout << "Error : the pointlist is not enough !\n";
		return;
	}
	pointProcess pointCloudProcesser;

	/************** z轴 ***************/
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
	pointCloudProcesser.SACSegmentation_model(pointlist[0], coefficients, inliers, pcl::SACMODEL_PLANE);

	Eigen::Vector3f normalVec;
	normalVec.setZero();
	normalVec(0) = coefficients->values[0];
	normalVec(1) = coefficients->values[1];
	normalVec(2) = coefficients->values[2];

	std::cout << "the normalVec : \n" << normalVec << std::endl;

	Eigen::Vector3f Zaxis(0, 0, 1);
	Eigen::Matrix3f rotation = pointProcess::computeRotation(normalVec, Zaxis);

	Eigen::Affine3f affine_R(rotation);
	PointCloud::Ptr cloud_rotated(new PointCloud());
	pcl::transformPointCloud(*pointlist[0], *cloud_rotated, affine_R);

	/***************************** x 轴*********************/
	pcl::ModelCoefficients::Ptr coefficients_sphere1(new pcl::ModelCoefficients());
	pcl::PointIndices::Ptr inliers_sphere1(new pcl::PointIndices());
	pcl::ModelCoefficients::Ptr coefficients_sphere2(new pcl::ModelCoefficients());
	pcl::PointIndices::Ptr inliers_sphere2(new pcl::PointIndices());
	pointCloudProcesser.SACSegmentation_model(pointlist[1], coefficients_sphere1, inliers_sphere1, pcl::SACMODEL_SPHERE);
	pointCloudProcesser.SACSegmentation_model(pointlist[2], coefficients_sphere2, inliers_sphere2, pcl::SACMODEL_SPHERE);

	Eigen::Vector3f sphere_centroid_1, sphere_centroid_2;
	sphere_centroid_1(0) = coefficients_sphere1->values[0];
	sphere_centroid_1(1) = coefficients_sphere1->values[1];
	sphere_centroid_1(2) = coefficients_sphere1->values[2];
	sphere_centroid_2(0) = coefficients_sphere2->values[0];
	sphere_centroid_2(1) = coefficients_sphere2->values[1];
	sphere_centroid_2(2) = coefficients_sphere2->values[2];

	pcl::transformPoint(sphere_centroid_1, sphere_centroid_1, affine_R);
	pcl::transformPoint(sphere_centroid_2, sphere_centroid_2, affine_R);
	std::cout << "sphere_centroid_1 : \n" << sphere_centroid_1 << "sphere_centroid_2 : \n" << sphere_centroid_2 << std::endl;
	Eigen::Vector3f Xaxis(1, 0, 0);
	Eigen::Vector3f X_tran;
	X_tran.setZero();
	X_tran = sphere_centroid_2 - sphere_centroid_1;
	X_tran(1) = 0;
	X_tran(2) = 0;
	float dist = X_tran.norm();
	X_tran /= dist;
	std::cout << "Xaxis : \n" << Xaxis << std::endl;
	std::cout << "X_tran : \n" << X_tran << std::endl;

	float theta_X = acos(Xaxis.dot(X_tran) / (Xaxis.norm()*X_tran.norm()));
	std::cout << "theta_X : " << theta_X << std::endl;
	Eigen::Matrix3f X_rotation;
	X_rotation.setIdentity();
	X_rotation(0, 0) = (float)cos(theta_X);
	X_rotation(0, 1) = (float)sin(theta_X);
	X_rotation(1, 0) = (float)-sin(theta_X);
	X_rotation(1, 1) = (float)cos(theta_X);
	std::cout << "the X_rotation : \n" << X_rotation << std::endl;

	Eigen::Affine3f affine_RX(X_rotation);
	PointCloud::Ptr cloud_rotated_2(new PointCloud());
	pcl::transformPointCloud(*cloud_rotated, *cloud_rotated_2, affine_RX);

	/**************** 标定结果 ************/
	Eigen::Matrix3f rotation_result = X_rotation * rotation;
	Eigen::Matrix4f result = Eigen::Matrix4f::Identity();
	result.block<3, 3>(0, 0) = rotation_result;
	rotation_ = result;

	Eigen::Affine3f tran_R(rotation_result);

	PointCloud::Ptr cloud_0(new PointCloud());
	PointCloud::Ptr cloud_1(new PointCloud());
	PointCloud::Ptr cloud_2(new PointCloud());
	pcl::copyPointCloud(*pointlist[0], *cloud_0);
	pcl::copyPointCloud(*pointlist[1], *cloud_1);
	pcl::copyPointCloud(*pointlist[2], *cloud_2);
	pcl::transformPointCloud(*cloud_0, *cloud_0, tran_R);
	pcl::transformPointCloud(*cloud_1, *cloud_1, tran_R);
	pcl::transformPointCloud(*cloud_2, *cloud_2, tran_R);

	//***************************显示

	Eigen::Vector3f centroid;
	pointProcess::computeCentroid(cloud_rotated_2, centroid);
	pointProcess::remove_Centroid(cloud_rotated_2, centroid);

	pcl::MomentOfInertiaEstimation <PointT> feature_extractor;
	feature_extractor.setInputCloud(cloud_rotated_2);
	feature_extractor.compute();

	pcl::PointXYZ min_point_AABB;
	pcl::PointXYZ max_point_AABB;

	feature_extractor.getAABB(min_point_AABB, max_point_AABB);

	Eigen::Matrix4f baseMatrix;
	baseMatrix.setIdentity();
	baseMatrix.block<3, 1>(0, 3) = -centroid;
	Eigen::Affine3f baseTran(baseMatrix);
	pcl::transformPointCloud(*pointlist[0], *pointlist[0], baseTran);
	pcl::transformPointCloud(*pointlist[1], *pointlist[1], baseTran);
	pcl::transformPointCloud(*pointlist[2], *pointlist[2], baseTran);

	pcl::transformPointCloud(*cloud_0, *cloud_0, baseTran);
	pcl::transformPointCloud(*cloud_1, *cloud_1, baseTran);
	pcl::transformPointCloud(*cloud_2, *cloud_2, baseTran);

	pcl::visualization::PCLVisualizer viewer("viewer");
	viewer.setBackgroundColor(0, 0, 0);
	viewer.addCoordinateSystem(50.0);
	pcl::visualization::PointCloudColorHandlerGenericField<PointT> color(pointlist[0], "z");
	viewer.addPointCloud(pointlist[0], color, "pointlist0");
	viewer.addPointCloud(pointlist[1], color, "*pointlist1");
	viewer.addPointCloud(pointlist[2], color, "*pointlist2");

	viewer.addPointCloud(cloud_0, color, "cloud_0");
	viewer.addPointCloud(cloud_1, color, "*cloud_1");
	viewer.addPointCloud(cloud_2, color, "*cloud_2");

	viewer.addCube(min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y,
		min_point_AABB.z, max_point_AABB.z, 1.0, 1.0, 0.0, "AABB");//AABB盒子

	viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
		pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "AABB");


	while (!viewer.wasStopped())
	{
		viewer.spinOnce(10);
	}

}

int main()
{
#if false
	//模拟点云拼接
	//************************//

	if (false)
	{
		//点云切割
		std::string scenePath = "F:\\vs_program\\PCLstudy\\Recognition_SVM\\1\\niudun.ply";
		pcl::PLYReader reader;
		PointCloud::Ptr pointcloud(new PointCloud());
		reader.read<PointT>(scenePath, *pointcloud);

		pointProcess::StatisticalOutlierRemoval_Filter(pointcloud, pointcloud);
		float resolution = pointProcess::computeResolution(pointcloud);
		pointProcess::VoxelGrid_Filter(pointcloud, pointcloud, resolution * 2);

		pointCloudRegistration(pointcloud);
}

	if (false)
	{
		//调平
		std::string Path = "F:\\vs_program\\PCLstudy\\Recognition_SVM\\calibrateDataSet";//calibrateDataSet
		std::string fileSuffix = ".ply";
		std::vector<std::string> files;
		getFiles(Path, fileSuffix, files);
		sort_filelists(files);
		pcl::PLYReader reader;

		std::vector<PointCloud::Ptr> pointlist;
		for (int i = 0; i < files.size(); ++i)
		{
			PointCloud::Ptr pointcloud(new PointCloud());
			reader.read<PointT>(files[i], *pointcloud);

			pointProcess::StatisticalOutlierRemoval_Filter(pointcloud, pointcloud);
			float resolution = pointProcess::computeResolution(pointcloud);
			pointProcess::VoxelGrid_Filter(pointcloud, pointcloud, resolution * 3);

			pointlist.push_back(pointcloud);
		}
		Eigen::Matrix4f tran;
		calibrate_by_plane_X(pointlist, tran);
		calibrate_by_XY(pointlist, tran);
	}
#endif
	//旋转创建样本
#if false
	std::string uwaPath = "F:\\vs_program\\PCLstudy\\Recognition_SVM\\UWAdataSet\\model";
	std::string stanfordPath = "F:\\vs_program\\PCLstudy\\Recognition_SVM\\UWAdataSet\\model";
	std::string increaseDataPath = "F:\\vs_program\\PCLstudy\\Recognition_SVM\\UWAdataSet\\modelData";
	std::vector<std::string> files;
	std::string format = ".ply";
	pcl::PLYReader reader;
	getFiles(uwaPath, format, files);
	for (int num = 0; num < files.size(); ++num)
	{
		PointCloud::Ptr cloud(new PointCloud);
		reader.read<PointT>(files[num], *cloud);

		std::vector<PointCloud::Ptr> cloudlist;
		recognition3D::rotateModelToDataSet(cloud, 2, recognition3D::Axis::z, cloudlist, increaseDataPath, files[num]);
	}
#endif
	//SVM训练
#if false
	//std::string uwa_objectLibPath = "E:\\vsProgram\\Recognition_ml\\UWAdataSet\\objectLib_SVM";
	//std::string uwa_scenePath = "E:\\vsProgram\\PCLstudy\\Recognition_SVM\\test";
	std::string stf_objectLibPath = "E:\\vsProgram\\Recognition_ml\\DataSet_UWA\\objectLib_SVM";
	std::string stf_scenePath = "E:\\vsProgram\\Recognition_ml\\DataSet_UWA\\scene_SVM";
	std::string stf_modelPath = "E:\\vsProgram\\Recognition_ml\\DataSet_UWA\\model";
	//训练模型
	bool train_flag = false;
	recognition3D recognitionTool;

	std::vector<std::string> files;
	std::string format = ".ply";
	pcl::PLYReader reader;

	if (train_flag)
	{
		getFiles(stf_objectLibPath, format, files);
		std::vector<PointCloud::Ptr> objLib;
		for (int i = 0; i < files.size(); ++i)
		{
			PointCloud::Ptr cloud(new PointCloud);
			reader.read<PointT>(files[i], *cloud);
			objLib.push_back(cloud);
		}

		bool setLib = recognitionTool.set_ObjectLibrary(objLib);
		bool setLabel = recognitionTool.set_ObjectLibrary_label(files);

		if (!setLib)
		{
			std::cout << "\nError: the objectLibrary is not set up successfully!\n";
		}
		else if (!setLabel)
		{
			std::cout << "\nError: the objectLib_labelVec is not set up successfully!\n";
		}
		else
		{
			recognitionTool.set_svm_train_model();
			recognitionTool.svm_classifier_train("STF_efh");
			//recognitionTool.svm_classifier_train("UWA");
		}
	}

	//加载模型       //  STF_ourcvfh_svm.xml  // STF_efh_svm.xml
	recognitionTool.load_svmClassifier(svmTrainer::classifier::SVM, "STF_efh_svm.xml");
	getFiles(stf_modelPath, format, files);
	sort_filelists(files);
	std::vector<PointCloud::Ptr> objLib;
	for (int i = 0; i < files.size(); ++i)
	{
		PointCloud::Ptr cloud(new PointCloud);
		reader.read<PointT>(files[i], *cloud);
		objLib.push_back(cloud);
	}
	if (!recognitionTool.set_ObjectLibrary(objLib, true))
		return -1;
	//载入场景
	std::vector<std::string> scenefiles;
	getFiles(stf_scenePath, format, scenefiles);
	
	std::vector<PointCloud::Ptr> sceneClouds;
	for (int i = 0; i < scenefiles.size(); ++i)
	{
		PointCloud::Ptr cloud(new PointCloud);
		reader.read<PointT>(scenefiles[i], *cloud);
		sceneClouds.push_back(cloud);
	}

	PointCloud::Ptr scenePointcloud(new PointCloud());
	pcl::copyPointCloud(*sceneClouds[0], *scenePointcloud);
	pointProcess pointProcesser_;
	pointProcess::StatisticalOutlierRemoval_Filter(scenePointcloud, scenePointcloud);
	float resolution = pointProcess::computeResolution(scenePointcloud);
	if (false)
	{
		pointProcess::VoxelGrid_Filter(scenePointcloud, scenePointcloud, resolution * 6);
		resolution = pointProcess::computeResolution(scenePointcloud);
	}
	//点云分割聚类
	//平面分割
	//int nr_points = scenePointcloud->points.size();
	//while (scenePointcloud->points.size() > 0.3 * nr_points)
	//{
	//	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
	//	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
	//	int maxSACsegmentaion = 100;
	//	float distThreshold = 3 * resolution;
	//	pointProcesser_.SACSegmentation_model(scenePointcloud, coefficients, inliers,
	//		pcl::SACMODEL_PLANE, maxSACsegmentaion, distThreshold);

	//	pointProcesser_.extractIndicesPoints(scenePointcloud, scenePointcloud, inliers, true);
	//}

	FeatureCloud tmpInput;
	tmpInput.setPointCloud(scenePointcloud);
	//Normal Calculation
	int normal_K = 30;
	float normal_R = resolution * 3;
	pointProcesser_.computeSurfaceNormals(tmpInput, normal_K);
	//剔除法向量无效的点
	pointProcesser_.removeNANfromNormal(tmpInput);

	std::vector<PointCloud::Ptr> cloudClusterList;
	std::vector<pcl::PointIndices> cloudClusterIndices;
	//欧氏距离
	if (true)
	{
		float clusterTolerance = resolution * 5;
		int minClusterSize = 2000;
		int maxClusterSize = 500000;
		pointProcesser_.EuclideanClusterExtraction(tmpInput.getPointCloud(), cloudClusterList, cloudClusterIndices,
			clusterTolerance, minClusterSize, maxClusterSize);
	}
	//区域生长
	if (false)
	{
		int minClusterSize = 1000;
		int maxClusterSize = 100000;
		int numofNeighbour = 30;
		float smoothThreshold = 50.0 / 180.0 * M_PI;
		float curvatureThreshold = 0.5;
		pointProcesser_.RegionGrowingClusterExtraction(tmpInput.getPointCloud(),
			cloudClusterList, cloudClusterIndices,
			tmpInput.getNormals(),
			minClusterSize, maxClusterSize, numofNeighbour, smoothThreshold, curvatureThreshold);
	}

	////预测

	std::vector<FeatureCloud> FeatureCloudList;
	std::vector<int> predictResult;
	FeatureCloudList.reserve(cloudClusterList.size());
	predictResult.reserve(cloudClusterList.size());
	std::cout << std::endl;
	for (size_t i = 0; i < cloudClusterList.size(); ++i)
	{
		std::cout << std::endl << std::endl << "***********提取分割区域***********\n";
		FeatureCloud tmpFC;
		tmpFC.setKeypoints(cloudClusterList[i]);

		pcl::PointIndices::Ptr tmpIndice(new pcl::PointIndices());
		tmpIndice->indices.insert(tmpIndice->indices.begin(), cloudClusterIndices[i].indices.begin(),
			cloudClusterIndices[i].indices.end());
		pointProcess::extractIndicesNormals(tmpInput.getNormals(), tmpFC.getKeypointNormals()
			, tmpIndice,false);

		float resolution = pointProcess::computeResolution(tmpFC.getKeypoints());
		float FPFH_radius = resolution * 4;
		pointProcess::computeFeatures_FPFH(tmpFC, FPFH_radius);
		FeatureCloudList.push_back(tmpFC);

		int result = recognitionTool.svm_predict(cloudClusterList[i]);
		predictResult.push_back(result);
	}

	pcl::visualization::PCLVisualizer viewer("viewer");
	pcl::visualization::PointCloudColorHandlerCustom<PointT> color(scenePointcloud, 200,200,200);
	viewer.addPointCloud(scenePointcloud, color, "cloud");

	pcl::MomentOfInertiaEstimation <PointT> feature_extractor;

	srand(time(NULL));
	for (size_t num = 0; num < FeatureCloudList.size(); ++num)
	{
		std::cout << std::endl << std::endl << "***********假设验证***********\n";
		std::stringstream ss;
		ss << num;	

		if (1)
		{		
			Eigen::Matrix4f tranRes;
			float score = recognitionTool.RotationTranslationCompute(predictResult[num], FeatureCloudList[num], tranRes);
			Eigen::Matrix3f rotation_ = tranRes.block<3, 3>(0, 0).transpose();
			Eigen::Matrix4f tranModel = Eigen::Matrix4f::Identity();
			tranModel.block<3, 3>(0, 0) = rotation_;
			tranModel.block<3, 1>(0, 3) = -tranRes.block<3, 1>(0, 3);

			std::cout << " [ predictResult ] " << predictResult[num] << std::endl;
			std::cout << " [ The ICP score ] " << score << std::endl;
			std::cout << std::endl << tranModel << std::endl;
			std::cout << " [ The ICP score 2 ] " << 0.331673 << std::endl;

			feature_extractor.setInputCloud(FeatureCloudList[num].getKeypoints());
			feature_extractor.compute();
			pcl::PointXYZ min_point_AABB;
			pcl::PointXYZ max_point_AABB;
			pcl::PointXYZ min_point_OBB;
			pcl::PointXYZ max_point_OBB;
			feature_extractor.getAABB(min_point_AABB, max_point_AABB);

			pcl::PointXYZ position;
			Eigen::Matrix3f rotation;
			feature_extractor.getOBB(min_point_OBB, max_point_OBB, position, rotation);

			Eigen::Vector3f _position(position.x, position.y, position.z);
			Eigen::Quaternionf quat(rotation);

			//viewer.addCube(_position, quat, max_point_OBB.x - min_point_OBB.x,
			//	max_point_OBB.y - min_point_OBB.y, max_point_OBB.z - min_point_OBB.z, "OBB_" + ss.str());//OBB
			//viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
			//	pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "OBB_" + ss.str());

			viewer.addCube(min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y,
				min_point_AABB.z, max_point_AABB.z, 1.0, 1.0, 0.0, "AABB_" + ss.str());//AABB盒子			
			viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
				pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "AABB_" + ss.str());

			//注意设置addText的大小
			switch (predictResult[num])
			{
			case 0:
			{
				std::stringstream name;
				name << 0;
				pcl::visualization::PointCloudColorHandlerCustom<PointT> color0(cloudClusterList[num], 255, 0, 0);
				viewer.addPointCloud(cloudClusterList[num], color0, "cloud_" + ss.str());
				//viewer.addText3D("Armadillo_" + name.str(), min_point_AABB, 0.02, 255, 0, 0, "Armadillo_" + ss.str());

				pcl::visualization::PointCloudColorHandlerCustom<PointT> colorModel(objLib[0], 125, 125, 125);
				pcl::transformPointCloud(*objLib[0], *objLib[0], tranModel);
				viewer.addPointCloud(objLib[0], colorModel, "model_" + ss.str());
			}
			break;
			case 1:
			{
				std::stringstream name;
				name << 1;
				pcl::visualization::PointCloudColorHandlerCustom<PointT> color1(cloudClusterList[num], 0, 255, 0);
				viewer.addPointCloud(cloudClusterList[num], color1, "cloud_" + ss.str());
				//viewer.addText3D("bun_" + name.str(), min_point_AABB, 0.02, 0, 255, 0, "bun_" + ss.str());

				pcl::visualization::PointCloudColorHandlerCustom<PointT> colorModel(objLib[1], 125, 125, 125);
				pcl::transformPointCloud(*objLib[1], *objLib[1], tranModel);
				viewer.addPointCloud(objLib[1], colorModel, "model_" + ss.str());
			}
			break;
			case 2:
			{
				std::stringstream name;
				name << 2;
				pcl::visualization::PointCloudColorHandlerCustom<PointT> color2(cloudClusterList[num], 0, 0, 255);
				viewer.addPointCloud(cloudClusterList[num], color2, "cloud_" + ss.str());
				//viewer.addText3D("dragon_vrip_" + name.str(), min_point_AABB, 0.02, 0, 0, 255, "dragon_vrip_" + ss.str());

				
				pcl::visualization::PointCloudColorHandlerCustom<PointT> colorModel(objLib[2], 125, 125, 125);
				pcl::transformPointCloud(*objLib[2], *objLib[2], tranModel);
				//viewer.addPointCloud(objLib[2], colorModel, "model_" + ss.str());
			}
			break;
			case 3:
			{
				std::stringstream name;
				name << 3;
				pcl::visualization::PointCloudColorHandlerCustom<PointT> color3(cloudClusterList[num], 255, 255, 0);
				viewer.addPointCloud(cloudClusterList[num], color3, "cloud_" + ss.str());
				//viewer.addText3D("happy_vrip_" + name.str(), min_point_AABB, 0.02, 255, 255, 0, "happy_vrip_" + ss.str());

				pcl::visualization::PointCloudColorHandlerCustom<PointT> colorModel(objLib[3], 125, 125, 125);
				pcl::transformPointCloud(*objLib[3], *objLib[3], tranRes.reverse());
				//viewer.addPointCloud(objLib[3], colorModel, "model_" + ss.str());
			}
			break;
			case 4:
			{
				std::stringstream name;
				name << 4;
				pcl::visualization::PointCloudColorHandlerCustom<PointT> color4(cloudClusterList[num], 255, 0, 255);
				viewer.addPointCloud(cloudClusterList[num], color4, "cloud_" + ss.str());
				//viewer.addText3D("statuette_" + name.str(), min_point_AABB, 0.02, 255, 0, 255, "statuette_" + ss.str());

				pcl::visualization::PointCloudColorHandlerCustom<PointT> colorModel(objLib[4], 125, 125, 125);
				pcl::transformPointCloud(*objLib[4], *objLib[4], tranRes.reverse());
				//viewer.addPointCloud(objLib[4], colorModel, "model_" + ss.str());
			}
			break;
			case 5:
			{
				std::stringstream name;
				name << 5;
				pcl::visualization::PointCloudColorHandlerCustom<PointT> color5(cloudClusterList[num], 0, 255, 255);
				viewer.addPointCloud(cloudClusterList[num], color5, "cloud_" + ss.str());
				//viewer.addText3D("dragon_" + name.str(), min_point_AABB, 0.02, 0, 255, 0, "dragon_" + ss.str());

				pcl::visualization::PointCloudColorHandlerCustom<PointT> colorModel(objLib[5], 125, 125, 125);
				pcl::transformPointCloud(*objLib[5], *objLib[5], tranRes.reverse());
				//viewer.addPointCloud(objLib[5], colorModel, "model_" + ss.str());
			}
			break;
			default:
				break;
			}
		}

		////显示聚类
		//int r = (int)(255 * rand() / (1.0 + RAND_MAX)) ;
		//int g = (int)(255 * rand() / (1.0 + RAND_MAX));
		//int b = (int)(255 * rand() / (1.0 + RAND_MAX));

		//pcl::visualization::PointCloudColorHandlerCustom<PointT> color(cloudClusterList[num], r, g, b);
		//viewer.addPointCloud(cloudClusterList[num], color, ss.str());
	}

	while (!viewer.wasStopped())
	{
		viewer.spinOnce(10);
	}

#endif

	//PPF
#if false
	//std::string modelFileName = "E:\\vsProgram\\Recognition_ml\\DataSet_UWA\\model\\chicken_1.ply";
	//std::string sceneFileName = "E:\\vsProgram\\Recognition_ml\\DataSet_UWA\\scene\\rs10_0.ply";
	//std::string SaveFileName = "E:\\vsProgram\\Recognition_ml\\DataSet_UWA\\scene\\result_rs10_0.ply";

	std::string modelFileName = "E:\\qtProgram\\pointCloudScanViewer\\ScanData\\DataSet_shigao\\david_model\\david-head.ply";
	std::string sceneFileName = "E:\\qtProgram\\pointCloudScanViewer\\ScanData\\DataSet_shigao\\Scene\\Complex_scene (6).ply";
	std::string SaveFileName = "E:\\qtProgram\\pointCloudScanViewer\\ScanData\\DataSet_shigao\\result.ply";

	pcl::PLYReader reader;

	PointCloud::Ptr modelPoints(new PointCloud());
	PointCloud::Ptr scenePoints(new PointCloud());
	reader.read<PointT>(modelFileName, *modelPoints);
	reader.read<PointT>(sceneFileName, *scenePoints);

	float resolution = pointProcess::computeResolution(scenePoints);
	pointProcess::VoxelGrid_Filter(scenePoints, scenePoints, resolution * 4);
	pointProcess::VoxelGrid_Filter(modelPoints, modelPoints, resolution * 4);
	pointProcess::StatisticalOutlierRemoval_Filter(scenePoints, scenePoints);
	pointProcess::StatisticalOutlierRemoval_Filter(modelPoints, modelPoints);
	
	resolution = pointProcess::computeResolution(scenePoints);

	Normals::Ptr model_normals(new Normals());
	Normals::Ptr scene_normals(new Normals());
	pointProcess::computeNormals(modelPoints, model_normals, 30, resolution * 4);
	pointProcess::computeNormals(scenePoints, scene_normals, 30, resolution * 4);

	std::vector<Eigen::Vector3f> M_pc, M_nr;
	std::vector<Eigen::Vector3f> S_pc, S_nr;

	ppf3d_public detector(0.05, 0.04);

	for (int i = 0; i < modelPoints->points.size(); ++i)
	{
		Eigen::Vector3f tmpP = modelPoints->points[i].getVector3fMap();
		Eigen::Vector3f tmpN = model_normals->points[i].getNormalVector3fMap();
		M_pc.push_back(tmpP);
		M_nr.push_back(tmpN);
	}

	for (int i = 0; i < scenePoints->points.size(); ++i)
	{
		Eigen::Vector3f tmpP = scenePoints->points[i].getVector3fMap();
		Eigen::Vector3f tmpN = scene_normals->points[i].getNormalVector3fMap();
		S_pc.push_back(tmpP);
		S_nr.push_back(tmpN);
	}

	// Now train the model
	std::cout << "M2E Training..." << std::endl;
	clock_t start, end;
	start = clock();
	detector.trainM2E(M_pc, M_nr);
	end = clock();
	std::cout << "\nM2E Training complete in "
		<< (end - start) / CLOCKS_PER_SEC << " s \n";

	// Match the model to the scene and get the pose
	std::cout << std::endl << "M2E Starting matching..." << std::endl;

	std::vector<Pose_3DM2E> resultsM2E;
	start = clock();
	detector.matchM2E(S_pc, S_nr, resultsM2E, 1.0 / 20.0, 0.02);
	end = clock();
	std::cout << std::endl << "M2E Matching complete in " 
		<< (end - start) / CLOCKS_PER_SEC << " s \n";

	std::vector<Eigen::Vector3f> ptm_save;
	for (int i = 0; i < M_pc.size(); ++i)
	{
		Eigen::Vector4f v_tmp;
		v_tmp << M_pc[i], 1;

		Eigen::Vector4f  pt_after_m = resultsM2E[0].pose*v_tmp;
		ptm_save.push_back(pt_after_m.head(3));
	}
	detector.savePLY(SaveFileName, ptm_save);

	/******************   界面显示  ********************/

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_model(new pcl::PointCloud<pcl::PointXYZ>());	
	pcl::io::loadPLYFile<pcl::PointXYZ>(modelFileName, *cloud_model);

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_scene(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPLYFile<pcl::PointXYZ>(sceneFileName, *cloud_scene);

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_match(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPLYFile<pcl::PointXYZ>(SaveFileName, *cloud_match);

	Eigen::Matrix4f tranRes = Eigen::Matrix4f::Identity();
	std::wcout << std::endl;
	//float score = pointProcess::ICP_pointclouds(cloud_scene, cloud_match, tranRes);
	std::cout << std::endl<< "[ ICP score ] " << 0.152488 << std::endl;
	pcl::transformPointCloud(*cloud_match, *cloud_match, tranRes);

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D viewer"));

	int v1(0), v2(0);

	//viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer->setBackgroundColor(0,0,0);
	//viewer->addPointCloud<pcl::PointXYZ>(cloud_model, "cloud model", v1);

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_scene(cloud_scene, 255, 255, 255);
	viewer->addPointCloud<pcl::PointXYZ>(cloud_scene, color_scene,"cloud scene");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_match(cloud_match, 0, 255, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloud_match, color_match, "cloud match");

	pcl::MomentOfInertiaEstimation <PointT> feature_extractor;
	feature_extractor.setInputCloud(cloud_match);
	feature_extractor.compute();
	pcl::PointXYZ min_point_AABB;
	pcl::PointXYZ max_point_AABB;
	pcl::PointXYZ min_point_OBB;
	pcl::PointXYZ max_point_OBB;
	feature_extractor.getAABB(min_point_AABB, max_point_AABB);

	pcl::PointXYZ position;
	Eigen::Matrix3f rotation;
	feature_extractor.getOBB(min_point_OBB, max_point_OBB, position, rotation);

	Eigen::Vector3f mass_center = Eigen::Vector3f::Zero();
	feature_extractor.getMassCenter(mass_center);

	Eigen::Vector3f major_vector, middle_vector, minor_vector;
	feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);
	
	Eigen::Vector3f _position(position.x, position.y, position.z);
	Eigen::Quaternionf quat(rotation);

	viewer->addCube(min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y,
		min_point_AABB.z, max_point_AABB.z, 1.0, 1.0, 0.0, "AABB");//AABB盒子			
	viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
		pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "AABB");
	//viewer->addText3D("Armadillo_0", min_point_AABB, 0.02, 255, 0, 0, "test");

	pcl::PointXYZ center(mass_center(0), mass_center(1), mass_center(2));
	pcl::PointXYZ x_axis(major_vector(0) + mass_center(0), major_vector(1) + mass_center(1), major_vector(2) + mass_center(2));
	pcl::PointXYZ y_axis(middle_vector(0) + mass_center(0), middle_vector(1) + mass_center(1), middle_vector(2) + mass_center(2));
	pcl::PointXYZ z_axis(minor_vector(0) + mass_center(0), minor_vector(1) + mass_center(1), minor_vector(2) + mass_center(2));

	Eigen::Vector3f x_aV = x_axis.getVector3fMap() ;
	Eigen::Vector3f y_aV = y_axis.getVector3fMap() ;
	Eigen::Vector3f z_aV = z_axis.getVector3fMap() ;

	PointT x_a(PointT(x_aV(0), x_aV(1), x_aV(2)));
	PointT y_a(PointT(y_aV(0), y_aV(1), y_aV(2)));
	PointT z_a(PointT(z_aV(0), z_aV(1), z_aV(2)));

	viewer->addLine(center, x_a, 1.0f, 0.0f, 0.0f, "major eigen vector");
	viewer->addLine(center, y_a, 0.0f, 1.0f, 0.0f, "middle eigen vector");
	viewer->addLine(center, z_a, 0.0f, 0.0f, 1.0f, "minor eigen vector");

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
#endif

	std::string FileName = "E:\\vsProgram\\point cloud algorithm\\Recognition_ml\\huhaohui\\Plane.ply";
	pcl::PLYReader reader;
	//pcl::PLYWriter writer;
	PointCloud::Ptr pcloud(new PointCloud());
	//PointCloud::Ptr filter_pcloud(new PointCloud());
	reader.read<PointT>(FileName, *pcloud);

	/*******************************************************************/
	//尝试滤波自编算法
	/*std::vector<Eigen::Vector3f> M_pc;
	for (int i = 0; i < pcloud->points.size(); ++i)
	{
		Eigen::Vector3f tmpP = pcloud->points[i].getVector3fMap();
		M_pc.push_back(tmpP);
	}
	mCP::cloudProcess cloudProcessing;
	float resolution = cloudProcessing.computeResolution(M_pc);
	std::vector<int> removeIndices;
	cloudProcessing.statisticalFilter(M_pc, M_pc, 30, 1.0, removeIndices);
	float leafsize = resolution * 2;
	Eigen::Vector3f leafS(leafsize, leafsize, leafsize);
	//cloudProcessing.voxelFilter(M_pc, M_pc, leafS);
	cloudProcessing.uniformSampling(M_pc, M_pc, leafS);
	resolution = cloudProcessing.computeResolution(M_pc);
	mCP::Features normals;
	cloudProcessing.computeNormals(M_pc, normals, resolution * 3, 11);*/
	/*******************************************************************/

	FeatureCloud tempObj;

	std::vector<int> mapping;
	pcl::removeNaNFromPointCloud(*pcloud, *pcloud, mapping);
	//Statisstical filter
	pointProcess::StatisticalOutlierRemoval_Filter(pcloud, pcloud, 30, 1.0);
	//Resolution Calculation
	float resolution = 0.0;
	resolution = pointProcess::computeResolution(pcloud);
	//降采样获取关键点
	if (true)
	{
		float leafSize = resolution * 4;
		pointProcess::VoxelGrid_Filter(pcloud, pcloud, leafSize);
		//更新分辨率
		resolution = pointProcess::computeResolution(pcloud);
	}
	//设置点云
	tempObj.setPointCloud(pcloud);
	//Normal Calculation
	int normal_K = 30;
	float normal_R = resolution * 3;
	//pointProcess::computeSurfaceNormals(tempObj, normal_K, normal_R);

	pointProcess pointCloudProcesser;
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
	pointCloudProcesser.SACSegmentation_model(pcloud, coefficients, inliers, pcl::SACMODEL_PLANE);

	Eigen::Vector3f normalVec;
	normalVec.setZero();
	normalVec(0) = coefficients->values[0];
	normalVec(1) = coefficients->values[1];
	normalVec(2) = coefficients->values[2];

	std::cout << "the normalVec : \n" << normalVec << std::endl;
	
	/*********************计算包围盒********************/
	/*
	pcl::MomentOfInertiaEstimation <PointT> feature_extractor;
	feature_extractor.setInputCloud(pcloud);
	feature_extractor.compute();
	pcl::PointXYZ min_point_AABB;
	pcl::PointXYZ max_point_AABB;
	pcl::PointXYZ min_point_OBB;
	pcl::PointXYZ max_point_OBB;
	feature_extractor.getAABB(min_point_AABB, max_point_AABB);

	pcl::PointXYZ position;
	Eigen::Matrix3f rotation;
	feature_extractor.getOBB(min_point_OBB, max_point_OBB, position, rotation);

	Eigen::Vector3f mass_center = Eigen::Vector3f::Zero();
	feature_extractor.getMassCenter(mass_center);

	Eigen::Vector3f major_vector, middle_vector, minor_vector;
	feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);

	Eigen::Vector3f _position(position.x, position.y, position.z);
	Eigen::Quaternionf quat(rotation);
	*/
	/*************************************************/

	//设置关键点
	//PointCloud::Ptr keypointCloudPtr(new PointCloud());
	//pcl::copyPointCloud(*tempObj.getPointCloud(), *keypointCloudPtr);
	//Normals::Ptr keypointNormalPtr(new Normals());
	//pcl::copyPointCloud(*tempObj.getNormals(), *keypointNormalPtr);
	//tempObj.setKeypoints(keypointCloudPtr);
	//tempObj.setKeypointNormals(keypointNormalPtr);
	//pointProcess::computeFeatures_OUR_CVFH(tempObj);

	/****************************************************/
	//显示直方图
	/*
	pcl::visualization::PCLPlotter* plotter = new pcl::visualization::PCLPlotter();
	plotter->setShowLegend(true);
	plotter->addFeatureHistogram<pcl::VFHSignature308>(*tempObj.getCVFH_features(), "vfh", 0, "OUR-CVFH");
	plotter->setWindowSize(600, 400);
	plotter->plot();*/
	/************************************************************/

	pcl::visualization::PCLVisualizer viewer;
	//int v1(0), v2(1);
	//viewer.createViewPort(0, 0, 0.5, 1, v1);
	//viewer.createViewPort(0.5, 0, 1, 1, v2);

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> singleColor(pcloud, 255, 255, 255);
	Eigen::Vector3f mean;
	pointProcess::computeCentroid(pcloud, mean);
	//pointProcess::remove_Centroid(pcloud, mean);
	viewer.addPointCloud(pcloud, singleColor, "cloud");
	//viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(tempObj.getPointCloud(), tempObj.getNormals(), 80, 2.5f, "normals");

	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> singleColor(pcloud, 0, 255, 0);
	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> singleColor2(filterCloud, 0, 255, 0);
	//viewer.addPointCloud(pcloud, singleColor, "cloud", v1);
	//viewer.addPointCloud(filterCloud, singleColor2, "cloud2", v2);


	//viewer.addCube(_position, quat, max_point_OBB.x - min_point_OBB.x,
	//	max_point_OBB.y - min_point_OBB.y, max_point_OBB.z - min_point_OBB.z, "OBB_");//OBB
	//viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
	//	pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "OBB_");

	//pcl::visualization::PointCloudColorHandlerCustom<PointT> colorP(pcloud, 250, 250, 250);
	////pcl::visualization::PointCloudColorHandlerCustom<PointT> colorfilter(filter_pcloud, 250, 250, 250);

	//viewer.addPointCloud(pcloud, colorP, "pcloud");
	////viewer.addPointCloud(filter_pcloud, colorfilter, "filter", v2);

	//pcl::PointXYZ center(mass_center(0), mass_center(1), mass_center(2));
	//pcl::PointXYZ x_axis(major_vector(0) + mass_center(0), major_vector(1) + mass_center(1), major_vector(2) + mass_center(2));
	//pcl::PointXYZ y_axis(middle_vector(0) + mass_center(0), middle_vector(1) + mass_center(1), middle_vector(2) + mass_center(2));
	//pcl::PointXYZ z_axis(minor_vector(0) + mass_center(0), minor_vector(1) + mass_center(1), minor_vector(2) + mass_center(2));

	//Eigen::Vector3f x_aV = x_axis.getVector3fMap() * 1.2;
	//Eigen::Vector3f y_aV = y_axis.getVector3fMap() ;
	//Eigen::Vector3f z_aV = z_axis.getVector3fMap() ;

	//PointT x_a(PointT(x_aV(0), x_aV(1), x_aV(2)));
	//PointT y_a(PointT(y_aV(0), y_aV(1), y_aV(2)));
	//PointT z_a(PointT(z_aV(0), z_aV(1), z_aV(2)));


	viewer.addLine(PointT(mean.x(),mean.y(),mean.z()), PointT(normalVec.x()*5+mean.x(), normalVec.y()*5 + mean.y(), normalVec.z()*5+mean.z()), 1.0f, 0.0f, 0.0f, "major eigen vector");
	//viewer.addLine(center, y_a, 0.0f, 1.0f, 0.0f, "middle eigen vector");
	//viewer.addLine(center, z_a, 0.0f, 0.0f, 1.0f, "minor eigen vector");
	viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "major eigen vector");


while (!viewer.wasStopped())
{
	viewer.spinOnce(100);
	boost::this_thread::sleep(boost::posix_time::microseconds(100000));
}

	system("pause");
	return 0;
}