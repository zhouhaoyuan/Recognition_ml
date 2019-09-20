#include "pointProcess.h"



pointProcess::pointProcess()
{
}


pointProcess::~pointProcess()
{
}

void pointProcess::computeCentroid(const PointCloud::Ptr pointInput, Eigen::Vector3f &mean)
{
	if (pointInput->points.empty())
	{
		std::cout << "Error: computeMean() , the input is null\n";
		return;
	}
	mean.setZero();
	Eigen::Vector4d centroid;
	pcl::compute3DCentroid(*pointInput, centroid);
	mean(0) = (float)centroid(0);
	mean(1) = (float)centroid(1);
	mean(2) = (float)centroid(2);
}

void pointProcess::remove_Centroid(PointCloud::Ptr pointInput, Eigen::Vector3f& mean)
{
	int npti = pointInput->size();
	for (size_t i = 0; i < npti; ++i)
	{
		pointInput->points[i].x -= mean(0);
		pointInput->points[i].y -= mean(1);
		pointInput->points[i].z -= mean(2);
	}
}

float pointProcess::computeResolution(PointCloud::Ptr pInput)
{
	if (pInput->points.empty())
	{
		std::cout << "Error: computeResolution() , the input is null\n";
		return -1;
	}
	float resolution = 0.0;
	int n_points = 0;
	std::vector<int> indices(2);//搜索点云中已存在的点就是 2 ，第一个为目标点
	std::vector<float> distances(2);

	pcl::search::KdTree<PointT> tree;
	tree.setInputCloud(pInput);

	int npSize = pInput->points.size();
	for (int i = 0; i < npSize; ++i)
	{
		if (!pcl_isfinite(pInput->points[i].x) )
		{
			continue;
		}
		int nres = tree.nearestKSearch(i, 2, indices, distances);
		if (nres == 2)
		{
			resolution += sqrt(distances[1]);
			++n_points;
		}
	}
	if (n_points != 0)
	{
		resolution /= n_points;
	}
	std::cout << "The pointCloud resolution : " << resolution << endl;

	return resolution;
}
//计算两向量的旋转矩阵
Eigen::Matrix3f pointProcess::computeRotation(Eigen::Vector3f &a, Eigen::Vector3f &b)
{
	//罗德里格斯
	Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();

	float theta = acos( a.dot(b) / (a.norm() * b.norm()));

	Eigen::Vector3f rotationAxis; 
	//rotationAxis = a.cross(b);
	rotationAxis(0) = a(1) * b(2) - a(2) * b(1);
	rotationAxis(1) = a(2) * b(0) - a(0) * b(2);
	rotationAxis(2) = a(0) * b(1) - a(1) * b(0);

	float square = sqrt(rotationAxis(0)*rotationAxis(0) + rotationAxis(1)*rotationAxis(1) + rotationAxis(2)*rotationAxis(2));
	//square = rotationAxis.norm();
	//square = sqrt(rotationAxis.cwiseAbs2().sum());
	rotationAxis(0) /= square;
	rotationAxis(1) /= square;
	rotationAxis(2) /= square;

	rotation(0, 0) = cos(theta) + rotationAxis(0) * rotationAxis(0) * (1 - cos(theta));
	rotation(0, 1) = (1 - cos(theta))*rotationAxis(0)*rotationAxis(1) - rotationAxis(2)*sin(theta);
	rotation(0, 2) = rotationAxis(1)*sin(theta) + rotationAxis(0)*rotationAxis(2)*(1 - cos(theta));
	rotation(1, 0) = rotationAxis(0)*rotationAxis(1)*(1 - cos(theta)) + rotationAxis(2)*sin(theta);
	rotation(1, 1) = cos(theta) + pow(rotationAxis(1), 2) * (1 - cos(theta));
	rotation(1, 2) = rotationAxis(1)*rotationAxis(2)*(1 - cos(theta)) - rotationAxis(0)*sin(theta);
	rotation(2, 0) = -rotationAxis(1)*sin(theta) + rotationAxis(0)*rotationAxis(2)*(1 - cos(theta));
	rotation(2, 1) = rotationAxis(1)*rotationAxis(2)*(1 - cos(theta)) + rotationAxis(0)*sin(theta);
	rotation(2, 2) = cos(theta) + pow(rotationAxis(2), 2) * (1 - cos(theta));

	std::cout << "the rotation matrix : \n" << rotation << std::endl;
	return rotation;
}
void pointProcess::removeNANfromNormal(FeatureCloud &cloud)
{
	PointCloud::Ptr nanremoved_(new PointCloud());
	std::vector<int> index;

	pcl::removeNaNNormalsFromPointCloud(*cloud.getNormals(), *cloud.getNormals(), index);
	pcl::copyPointCloud(*cloud.getPointCloud(), index, *nanremoved_);
	cloud.setPointCloud(nanremoved_);
}
void pointProcess::Uniform_Filter(PointCloud::Ptr input, PointCloud::Ptr output,
	float uniform_Radius , pcl::PointIndices filterIndices)
{
	clock_t start, end;
	start = clock();

	int num = input->size();
	pcl::UniformSampling<PointT> uniform_sampling;
	uniform_sampling.setInputCloud(input);
	uniform_sampling.setRadiusSearch(uniform_Radius);
	uniform_sampling.filter(*output);
    uniform_sampling.getRemovedIndices(filterIndices);

	std::cout << "Uniform_Filter, Input points: " << num << "; Output points: " << output->size() << std::endl;

	end = clock();
	std::cout << "StatisticalOutlierRemoval_Filter has finished in "
		<< (end - start) / CLOCKS_PER_SEC << " s \n";
}
void pointProcess::VoxelGrid_Filter(PointCloud::Ptr input, PointCloud::Ptr output, float leafsize)
{
	if (input->points.empty())
	{
		std::cout << "Error: VoxelGrid_Filter() , the input is null\n";
		return;
	}
	clock_t start, end;
	start = clock();

	int num = input->size();
	pcl::VoxelGrid<PointT> voxelgrid_filter;//对体素网格中所有点求均值,以期望均值点代替原始点集,更精确
	voxelgrid_filter.setLeafSize(leafsize, leafsize, leafsize);
	voxelgrid_filter.setInputCloud(input);
	voxelgrid_filter.filter(*output);

	end = clock();
	std::cout << "VoxelGrid_Filter(), Input points: " << num
		<< "; Output points: " << output->size() << " ; " << (end - start) / CLOCKS_PER_SEC << " s \n";
}

void pointProcess::StatisticalOutlierRemoval_Filter(
	PointCloud::Ptr input,
	PointCloud::Ptr output,
	int K ,
	float stddevMulThresh)
{
	if (input->points.empty())
	{
		std::cout << "Error: VoxelGrid_Filter() , the input is null\n";
		return;
	}
	clock_t start, end;
	start = clock();

	int num = input->size();
	pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
	statistical_filter.setMeanK(K);
	statistical_filter.setStddevMulThresh(stddevMulThresh);
	statistical_filter.setInputCloud(input);
	statistical_filter.filter(*output);

	end = clock();
	std::cout << "StatisticalOutlierRemoval_Filter(), Input points: " << num
		<< "; Output points: " << output->size() << " ; " << (end - start) / CLOCKS_PER_SEC << " s \n";
}

void pointProcess::PassThrough_Filter(
	PointCloud::Ptr input,
	PointCloud::Ptr output,
	std::string axis,
	float upBound,
	float downBound,
	bool negative)
{
	if (input->points.empty())
	{
		std::cout << "Error: VoxelGrid_Filter() , the input is null\n";
		return;
	}

	clock_t start, end;
	start = clock();

	int num = input->size();
	pcl::PassThrough<PointT> pass;
	pass.setInputCloud(input);
	pass.setFilterFieldName(axis);
	pass.setFilterLimits(downBound, upBound);
	pass.setFilterLimitsNegative(negative);//设置不在范围内的点保留还是去除
	pass.filter(*output);

	end = clock();
	std::cout << "PassThrough_Filter(), Input points: " << num
	<< "; Output points: " << output->size() << " ; " << (end - start) / CLOCKS_PER_SEC << " s \n";
}
//双边滤波
void pointProcess::Bilateral_Filter(
	PointCloud::Ptr input,
	PointCloud::Ptr output,
	double sigmas,
	double sigmar
)
{
	if (input->points.empty())
	{
		std::cout << "Error: VoxelGrid_Filter() , the input is null\n";
		return;
	}
	clock_t start, end;
	start = clock();

	int num = input->size();
	pcl::FastBilateralFilterOMP<PointT> bf;
	bf.setNumberOfThreads(4);
	bf.setInputCloud(input);
	bf.setSigmaS(sigmas);
	bf.setSigmaR(sigmar);
	bf.filter(*output);
		
	end = clock();
	std::cout << "PassThrough_Filter(), Input points: " << num
		<< "; Output points: " << output->size() << " ; " << (end - start) / CLOCKS_PER_SEC << " s \n";
}

void pointProcess::computeSurfaceNormals(
	FeatureCloud& cloud,
	int K,
	float radius,
	int numofthreads)
{
	clock_t start, end;
	start = clock();

	Normals::Ptr normals_(new Normals());
	pcl::NormalEstimationOMP<PointT, NormalT> norm_est;
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());

	norm_est.setNumberOfThreads(numofthreads);
	norm_est.setSearchMethod(tree);
	if (radius != 0) {

		norm_est.setRadiusSearch(radius);
	}
	else {
		norm_est.setKSearch(K);
	}
	norm_est.setInputCloud(cloud.getPointCloud());
	norm_est.compute(*normals_);
	cloud.setNormals(normals_);
	removeNANfromNormal(cloud);

	end = clock();
	std::cout << "computeSurfaceNormals() has finished in "
		<< (end - start) / CLOCKS_PER_SEC << " s \n";
}
Eigen::Vector4f pointProcess::computeWeightedNormal(PointCloud::Ptr inputCloud, Eigen::Vector4f& point)
{
	clock_t begin = clock();

	float resolution = computeResolution(inputCloud);
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
	tree->setInputCloud(inputCloud);
	
	PointT point_(point(0), point(1), point(2));
	std::vector<int> indices;
	std::vector<float> distances;
	int num = tree->nearestKSearch(point_, inputCloud->points.size() - 1, indices, distances);
	std::sort(distances.begin(), distances.end());
	float R_max = distances[distances.size() - 1];

	Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
	Eigen::Matrix<float, 1, 9, Eigen::RowMajor> accu = Eigen::Matrix<float, 1, 9, Eigen::RowMajor>::Zero();
	float weightPAll = 0.0;

	std::vector<int> tmpIndices;
	std::vector<float> tmpDistances;

	#pragma omp parallel for shared (accu, weightPAll) private (tmpIndices, tmpDistances) num_threads(4)
	for (int i = 0; i < indices.size(); ++i)
	{
		float weightP = 1 / (num * (R_max - (inputCloud->points[indices[i]].getVector3fMap() - point_.getVector3fMap()).norm()));
		
		weightPAll += weightP;
		//cov += weightP * (inputCloud->points[indices[i]].getVector3fMap() - point_.getVector3fMap())
		//	*(inputCloud->points[indices[i]].getVector3fMap() - centroidGeometry).transpose();

		accu[0] += weightP * inputCloud->points[indices[i]].x * inputCloud->points[indices[i]].x;
		accu[1] += weightP * inputCloud->points[indices[i]].x * inputCloud->points[indices[i]].y;
		accu[2] += weightP * inputCloud->points[indices[i]].x * inputCloud->points[indices[i]].z;
		accu[3] += weightP * inputCloud->points[indices[i]].y * inputCloud->points[indices[i]].y; // 4
		accu[4] += weightP * inputCloud->points[indices[i]].y * inputCloud->points[indices[i]].z; // 5
		accu[5] += weightP * inputCloud->points[indices[i]].z * inputCloud->points[indices[i]].z; // 8
		accu[6] += weightP * inputCloud->points[indices[i]].x;
		accu[7] += weightP * inputCloud->points[indices[i]].y;
		accu[8] += weightP * inputCloud->points[indices[i]].z;
	}
	
	accu /= static_cast<float> (indices.size());
	accu /= weightPAll;

	cov.coeffRef(0) = accu[0] - accu[6] * accu[6];
	cov.coeffRef(1) = accu[1] - accu[6] * accu[7];
	cov.coeffRef(2) = accu[2] - accu[6] * accu[8];
	cov.coeffRef(4) = accu[3] - accu[7] * accu[7];
	cov.coeffRef(5) = accu[4] - accu[7] * accu[8];
	cov.coeffRef(8) = accu[5] - accu[8] * accu[8];
	cov.coeffRef(3) = cov.coeff(1);
	cov.coeffRef(6) = cov.coeff(2);
	cov.coeffRef(7) = cov.coeff(5);

	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(cov, Eigen::ComputeEigenvectors);
	Eigen::Matrix3f eigenVecs = eigensolver.eigenvectors();
	Eigen::Vector3f eigenVals = eigensolver.eigenvalues();
	Eigen::Vector3f::Index minIdx;
	eigenVals.minCoeff(&minIdx);
	Eigen::Vector4f eigenVector = Eigen::Vector4f::Zero();
	eigenVector(0) = eigenVecs(0, minIdx);
	eigenVector(1) = eigenVecs(1, minIdx);
	eigenVector(2) = eigenVecs(2, minIdx);
	//std::cout << "eigenVals : \n" << eigenVals << std::endl;
	//std::cout << "eigenVecs : \n" << eigenVecs << std::endl;
	clock_t end = clock();
	std::cout << "computeWeightedNormal() consumed : " << float(end - begin) / CLOCKS_PER_SEC << " s \n";

	return eigenVector;
}
//FPFH计算
void pointProcess::computeFeatures_FPFH(FeatureCloud &cloud, float R)
{
	clock_t start, end;
	start = clock();

	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_features_(new pcl::PointCloud<pcl::FPFHSignature33>());
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr nanremoved_(new pcl::PointCloud<pcl::FPFHSignature33>());
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
	pcl::FPFHEstimationOMP<PointT, NormalT, pcl::FPFHSignature33> fpfh;
	fpfh.setNumberOfThreads(4);
	fpfh.setSearchSurface(cloud.getKeypoints());
	fpfh.setInputCloud(cloud.getKeypoints());
	fpfh.setInputNormals(cloud.getKeypointNormals());
	fpfh.setSearchMethod(tree);
	fpfh.setRadiusSearch(R);

	fpfh.compute(*fpfh_features_);//3个bin，每个bin有11间隔
	removeNANfromFPFH(fpfh_features_, nanremoved_, cloud);

	end = clock();
	std::cout << "computeFeatures_FPFH() has finished in "
		<< (end - start) / CLOCKS_PER_SEC << " s \n";
}
//SHOT计算
void pointProcess::computeFeatures_SHOT(FeatureCloud &cloud, float R)
{
	clock_t start, end;
	start = clock();

	pcl::PointCloud<pcl::SHOT352>::Ptr shot_features_(new pcl::PointCloud<pcl::SHOT352>());
	pcl::PointCloud<pcl::SHOT352>::Ptr nanremoved_(new pcl::PointCloud<pcl::SHOT352>());
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
	pcl::SHOTEstimationOMP<PointT, NormalT, pcl::SHOT352> shot_est;
	shot_est.setNumberOfThreads(4);
	shot_est.setSearchSurface(cloud.getKeypoints());
	shot_est.setInputNormals(cloud.getKeypointNormals());
	shot_est.setInputCloud(cloud.getKeypoints());
	shot_est.setSearchMethod(tree);
	shot_est.setRadiusSearch(R); 

	shot_est.compute(*shot_features_);
	
	removeNANfromSHOT(shot_features_, nanremoved_, cloud);

	end = clock();
	std::cout << "computeFeatures_SHOT() has finished in "
		<< (end - start) / CLOCKS_PER_SEC << " s \n";
}
void pointProcess::computeFeatures_VFH(FeatureCloud &cloud)
{
	clock_t start, end;
	start = clock();

	pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhPtr(new pcl::PointCloud<pcl::VFHSignature308>());
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());

	pcl::VFHEstimation< PointT, NormalT, pcl::VFHSignature308> vfh;
	vfh.setInputCloud(cloud.getKeypoints());
	vfh.setInputNormals(cloud.getKeypointNormals());
	vfh.setSearchMethod(tree);
	vfh.setNormalizeBins(true);
	vfh.setNormalizeDistance(false);
	vfh.compute(*vfhPtr);

	cloud.setVFH_features(vfhPtr);
	end = clock();
	std::cout << "computeFeatures_VFH() has finished in "
		<< (end - start) / CLOCKS_PER_SEC << " s \n";
}
void pointProcess::computeFeatures_CVFH(FeatureCloud &cloud)
{
	clock_t start, end;
	start = clock();

	float resolution = pointProcess::computeResolution(cloud.getPointCloud());
	//求曲率阈值
	std::vector<int> indices_out;
	std::vector<int> indices_in;
	float curvatureThreshold = 0.0;
	std::vector<float> curvatureVec;
	curvatureVec.reserve(cloud.getKeypointNormals()->points.size());
	for (size_t num = 0; num < cloud.getKeypointNormals()->points.size(); ++num)
	{
		curvatureVec.push_back(cloud.getKeypointNormals()->points[num].curvature);
	}
	std::sort(curvatureVec.begin(), curvatureVec.end());
	curvatureThreshold = curvatureVec[cloud.getKeypointNormals()->points.size() - 1] * 0.5;

	pcl::PointCloud<pcl::VFHSignature308>::Ptr cvfhPtr(new pcl::PointCloud<pcl::VFHSignature308>());
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());

	pcl::CVFHEstimation< PointT, NormalT, pcl::VFHSignature308> cvfh;
	cvfh.setInputCloud(cloud.getKeypoints());
	cvfh.setInputNormals(cloud.getKeypointNormals());
	cvfh.setSearchMethod(tree);
	cvfh.setNormalizeBins(true);
	cvfh.setClusterTolerance( resolution * 2.5);
	cvfh.setCurvatureThreshold(curvatureThreshold);//曲率阈值，去除法向量用
	cvfh.setEPSAngleThreshold(50.0 / 180 * M_PI);//相邻法向量最大偏差，聚类标准
	cvfh.compute(*cvfhPtr);

	cloud.setCVFH_features(cvfhPtr);
	end = clock();
	std::cout <<"vfh size : "<<cvfhPtr->points.size() << " , computeFeatures_CVFH() has finished in "
		<< (end - start) / CLOCKS_PER_SEC << " s \n";
}
// 计算SPFH
void pointProcess::computePointSPFHSignature(PointCloudNormal::Ptr cloud,
	pcl::PointIndices indices,
	Eigen::Vector4f& centroid_p,
	Eigen::Vector4f& centroid_normal,
	Eigen::VectorXf& hist_f1_,
	Eigen::VectorXf& hist_f2_,
	Eigen::VectorXf& hist_f3_,
	Eigen::VectorXf& hist_f4_)
{
	clock_t begin = clock();
	bool normalize_distances_ = true;// Normalize the shape distribution component of VFH
	bool normalize_bins_ = true;//Normalize bins by the number the total number of points
	bool size_component_ = false;//Activate or deactivate the size component of VFH 
	double distance_normalization_factor = 1.0;
	if (normalize_distances_)
	{
		Eigen::Vector4f max_pt;
		pcl::getMaxDistance(*cloud, centroid_p, max_pt);
		max_pt[3] = 0;
		distance_normalization_factor = (centroid_p - max_pt).norm();
	}
	// Factorization constant
	float hist_incr;
	if (normalize_bins_)
		hist_incr = 100.0f / static_cast<float> (cloud->points.size() - 1);
	else
		hist_incr = 1.0f;

	float hist_incr_size_component;
	if (size_component_)
		hist_incr_size_component = hist_incr;
	else
		hist_incr_size_component = 0.0;
	// Iterate over all the points in the neighborhood
	int nr_bins_f1_ = 45, nr_bins_f2_ = 45,
		nr_bins_f3_ = 45, nr_bins_f4_ = 45;

	Eigen::Vector4f pfh_tuple;
	hist_f1_.setZero(nr_bins_f1_);
	hist_f2_.setZero(nr_bins_f2_);
	hist_f3_.setZero(nr_bins_f3_);
	hist_f4_.setZero(nr_bins_f4_);

	PointCloud::Ptr pointCloud(new PointCloud);
	for (size_t i = 0; i < indices.indices.size(); ++i)
	{
		PointT point;
		point.x = cloud->points[indices.indices[i]].x;
		point.y = cloud->points[indices.indices[i]].y;
		point.z = cloud->points[indices.indices[i]].z;
		pointCloud->points.push_back(point);
	}

	pcl::search::KdTree<PointT> tree;
	tree.setInputCloud(pointCloud);
	float resolution = computeResolution(pointCloud);

	float d_pi_ = 1.0 / (2.0 * M_PI);
	for (size_t idx = 0; idx < indices.indices.size(); ++idx)
	{
		// Compute the pair P to NNi
		Eigen::Vector4f point = cloud->points[indices.indices[idx]].getVector4fMap();
		
		//重心加权法矢
		std::vector<int> tmpIndices;
		std::vector<float> tmpDist;
		tree.radiusSearch(pointCloud->points[idx], resolution * 3, tmpIndices, tmpDist);
		PointCloud::Ptr tmpCloud(new PointCloud);
		pcl::copyPointCloud(*pointCloud, tmpIndices, *tmpCloud);
		////TODO：
		Eigen::Vector4f point_normal = computeWeightedNormal(tmpCloud, point);
		//无加权法矢
		NormalT normal;
		normal.normal_x = cloud->points[indices.indices[idx]].normal_x;
		normal.normal_y = cloud->points[indices.indices[idx]].normal_y;
		normal.normal_z = cloud->points[indices.indices[idx]].normal_z;
		/////TODO：
		//Eigen::Vector4f point_normal(normal.normal_x, normal.normal_y, normal.normal_z, 0);

		if (!pcl::computePairFeatures(centroid_p, centroid_normal, point,
			point_normal, pfh_tuple[0], pfh_tuple[1],
			pfh_tuple[2], pfh_tuple[3]))
			continue;

		// Normalize the f1, f2, f3, f4 features and push them in the histogram
		int h_index = static_cast<int> (floor(nr_bins_f1_ * ((pfh_tuple[0] + M_PI) * d_pi_)));
		if (h_index < 0)
			h_index = 0;
		if (h_index >= nr_bins_f1_)
			h_index = nr_bins_f1_ - 1;
		hist_f1_(h_index) += hist_incr;

		h_index = static_cast<int> (floor(nr_bins_f2_ * ((pfh_tuple[1] + 1.0) * 0.5)));
		if (h_index < 0)
			h_index = 0;
		if (h_index >= nr_bins_f2_)
			h_index = nr_bins_f2_ - 1;
		hist_f2_(h_index) += hist_incr;

		h_index = static_cast<int> (floor(nr_bins_f3_ * ((pfh_tuple[2] + 1.0) * 0.5)));
		if (h_index < 0)
			h_index = 0;
		if (h_index >= nr_bins_f3_)
			h_index = nr_bins_f3_ - 1;
		hist_f3_(h_index) += hist_incr;

		if (normalize_distances_)
			h_index = static_cast<int> (floor(nr_bins_f4_ * (pfh_tuple[3] / distance_normalization_factor)));
		else
			h_index = static_cast<int> (pcl_round(pfh_tuple[3] * 100));

		if (h_index < 0)
			h_index = 0;
		if (h_index >= nr_bins_f4_)
			h_index = nr_bins_f4_ - 1;
		hist_f4_(h_index) += hist_incr_size_component;
	}

	clock_t end = clock();
	std::cout << "computePointSPFHSignature() consumed : " << float(end - begin) / CLOCKS_PER_SEC << " s \n";
}
// 计算改进的VFH
void pointProcess::computeImproveVFHfeature(PointCloudNormal::Ptr pointcloudnormal,
	pcl::PointIndices indices,
	Eigen::Vector4f& xyz_centroid,
	Eigen::Vector4f& normal_centroid,
	pcl::PointCloud<pcl::VFHSignature308>& result)
{
	clock_t t1 = clock();

	PointCloud::Ptr cloud(new PointCloud);
	for (size_t i = 0; i < indices.indices.size(); ++i)
	{
		PointT point;
		point.x = pointcloudnormal->points[indices.indices[i]].x;
		point.y = pointcloudnormal->points[indices.indices[i]].y;
		point.z = pointcloudnormal->points[indices.indices[i]].z;
		cloud->points.push_back(point);
	}

	// Compute the direction of view from the viewpoint to the centroid
	Eigen::Vector4f viewpoint(0, 0, 0, 0);
	Eigen::Vector4f d_vp_p = viewpoint - xyz_centroid;
	d_vp_p.normalize();
	//重心加权法矢
	Eigen::Vector4f centroid_n = computeWeightedNormal(cloud, xyz_centroid);

	Eigen::VectorXf hist_f1_;
	Eigen::VectorXf hist_f2_;
	Eigen::VectorXf hist_f3_;
	Eigen::VectorXf hist_f4_;
	computePointSPFHSignature(pointcloudnormal, indices, xyz_centroid, centroid_n,
		hist_f1_, hist_f2_, hist_f3_, hist_f4_);

	result.points.resize(1);
	result.width = 1;
	result.height = 1;

	for (int d = 0; d < hist_f1_.size(); ++d)
		result.points[0].histogram[d + 0] = hist_f1_[d];

	size_t data_size = hist_f1_.size();
	for (int d = 0; d < hist_f2_.size(); ++d)
		result.points[0].histogram[d + data_size] = hist_f2_[d];

	data_size += hist_f2_.size();
	for (int d = 0; d < hist_f3_.size(); ++d)
		result.points[0].histogram[d + data_size] = hist_f3_[d];

	data_size += hist_f3_.size();
	for (int d = 0; d < hist_f4_.size(); ++d)
		result.points[0].histogram[d + data_size] = hist_f4_[d];

	// ---[ Step 2 : obtain the viewpoint component
	Eigen::VectorXf hist_vp_;
	int nr_bins_vp_ = 128;
	hist_vp_.setZero(nr_bins_vp_);
	bool normalize_bins_ = true;
	double hist_incr;
	if (normalize_bins_)
		hist_incr = 100.0 / static_cast<double> (cloud->points.size());
	else
		hist_incr = 1.0;

	for (size_t i = 0; i < indices.indices.size(); ++i)
	{
		Eigen::Vector4f normal(pointcloudnormal->points[indices.indices[i]].normal[0],
			pointcloudnormal->points[indices.indices[i]].normal[1],
			pointcloudnormal->points[indices.indices[i]].normal[2], 0);
		// Normalize
		double alpha = (normal.dot(d_vp_p) + 1.0) * 0.5;
		int fi = static_cast<int> (floor(alpha * static_cast<double> (hist_vp_.size())));
		if (fi < 0)
			fi = 0;
		if (fi >(static_cast<int> (hist_vp_.size()) - 1))
			fi = static_cast<int> (hist_vp_.size()) - 1;
		// Bin into the histogram
		hist_vp_[fi] += static_cast<float> (hist_incr);
	}
	data_size += hist_f4_.size();
	// Copy the resultant signature
	for (int d = 0; d < hist_vp_.size(); ++d)
		result.points[0].histogram[d + data_size] = hist_vp_[d];

	clock_t t2 = clock();
	std::cout << "computeImproveVFHfeature() consumed : " << float(t2 - t1) / CLOCKS_PER_SEC << " s \n";
}
//改进的CVFH全局特征描述子
void pointProcess::computeFeatures_ImprovedCVFH(FeatureCloud& cloud)
{
	clock_t start, end;
	start = clock();
	//分辨率
	float resolution = pointProcess::computeResolution(cloud.getKeypoints());

	std::vector<int> filterIndices;
	pcl::removeNaNNormalsFromPointCloud(*cloud.getKeypointNormals(), *cloud.getKeypointNormals(), filterIndices);
	pcl::copyPointCloud(*cloud.getKeypoints(), filterIndices, *cloud.getKeypoints());

	pcl::CVFHEstimation< PointT, NormalT, pcl::VFHSignature308> cvfh;
	//Step 0: remove normals with high curvature
	std::vector<int> indices_out;
	std::vector<int> indices_in;
	float curvatureThreshold = 0.0;
	std::vector<float> curvatureVec;
	curvatureVec.reserve(cloud.getKeypointNormals()->points.size());

	for (size_t num = 0; num < cloud.getKeypointNormals()->points.size(); ++num)
	{
		curvatureVec.push_back(cloud.getKeypointNormals()->points[num].curvature);
	}
	std::sort(curvatureVec.begin(), curvatureVec.end());
	curvatureThreshold = curvatureVec[cloud.getKeypointNormals()->points.size() - 1] * 0.5;

	//过滤高曲率点云
	cvfh.filterNormalsWithHighCurvature(*cloud.getKeypointNormals(), filterIndices, indices_out, indices_in, curvatureThreshold);
	std::cout << "filterNormalsWithHighCurvature() indices_in size : " << indices_in.size() << std::endl;
	std::cout << "filterNormalsWithHighCurvature() indices_out size : " << indices_out.size() << std::endl;

	PointCloud::Ptr filtered_cloud(new PointCloud());
	pcl::copyPointCloud(*cloud.getKeypoints(), indices_in, *filtered_cloud);
	//更新分辨率
	resolution = computeResolution(filtered_cloud);
	//生成带法向量的点云
	PointCloudNormal::Ptr normals_filtered_cloud(new PointCloudNormal());
	normals_filtered_cloud->width = static_cast<uint32_t> (indices_in.size());
	normals_filtered_cloud->height = 1;
	normals_filtered_cloud->points.resize(normals_filtered_cloud->width);
	for (size_t i = 0; i < indices_in.size(); ++i)
	{
		normals_filtered_cloud->points[i].x = cloud.getKeypoints()->points[indices_in[i]].x;
		normals_filtered_cloud->points[i].y = cloud.getKeypoints()->points[indices_in[i]].y;
		normals_filtered_cloud->points[i].z = cloud.getKeypoints()->points[indices_in[i]].z;
	}

	std::vector<pcl::PointIndices> clusters;
	//设置聚类点云最小值
	int min_points_ = cloud.getKeypoints()->points.size() / 10;
	if (normals_filtered_cloud->points.size() >= min_points_)
	{
		//recompute normals and use them for clustering
		pcl::search::KdTree<pcl::PointNormal>::Ptr normals_tree_filtered(new pcl::search::KdTree<pcl::PointNormal>(false));
		normals_tree_filtered->setInputCloud(normals_filtered_cloud);

		pcl::NormalEstimationOMP<PointNormalT, PointNormalT> n3d;
		n3d.setRadiusSearch(resolution * 3);
		n3d.setSearchMethod(normals_tree_filtered);
		n3d.setInputCloud(normals_filtered_cloud);
		n3d.compute(*normals_filtered_cloud);

		pcl::search::KdTree<pcl::PointNormal>::Ptr normals_tree(new pcl::search::KdTree<pcl::PointNormal>(false));
		normals_tree->setInputCloud(normals_filtered_cloud);

		float cluster_tolerance_ = resolution * 2.5;
		float eps_angle_threshold_ = 50.0/180.0*M_PI;
		int min_points_ = 500;
		int max_points_ = 10000;

		extractEuclideanClustersSmooth(*normals_filtered_cloud,
			*normals_filtered_cloud,
			cluster_tolerance_,
			normals_tree,
			clusters,
			eps_angle_threshold_,
			min_points_,
			max_points_);

	}

	pcl::PointCloud<pcl::VFHSignature308>::Ptr improveCVFHs(new pcl::PointCloud<pcl::VFHSignature308>());

	if (clusters.size() > 0)
	{
		for (size_t i = 0; i < clusters.size(); ++i)
		{
			PointCloud::Ptr tmpCloud(new PointCloud);
			pcl::copyPointCloud(*normals_filtered_cloud, clusters[i], *tmpCloud);
			resolution = computeResolution(tmpCloud);
			//均匀采样
			pcl::PointIndices filterIndices;
			Uniform_Filter(tmpCloud, tmpCloud, resolution * 2, filterIndices);
			clusters[i].indices.erase(filterIndices.indices.begin(), filterIndices.indices.end());

			Eigen::Vector4f avg_normal = Eigen::Vector4f::Zero();
			Eigen::Vector4f avg_centroid = Eigen::Vector4f::Zero();

			for (size_t j = 0; j < clusters[i].indices.size(); ++j)
			{
				avg_normal += normals_filtered_cloud->points[clusters[i].indices[j]].getNormalVector4fMap();
				avg_centroid += normals_filtered_cloud->points[clusters[i].indices[j]].getVector4fMap();
			}
			avg_normal /= static_cast<float>(clusters[i].indices.size());
			avg_normal.normalize();
			avg_centroid /= static_cast<float>(clusters[i].indices.size());

			pcl::PointCloud<pcl::VFHSignature308> vfh_signature;
			computeImproveVFHfeature(normals_filtered_cloud, clusters[i], avg_centroid, avg_normal, vfh_signature);
			improveCVFHs->points.push_back(vfh_signature.points[0]);
		}
		cloud.setCVFH_features(improveCVFHs);
	}
	else {
		std::cout << "\nError: clusters is empty!\n";
		cloud.setCVFH_features(improveCVFHs);
	}
	
}
//OUR_CVFH全局特征描述子
void pointProcess::computeFeatures_OUR_CVFH(FeatureCloud &cloud)
{
	clock_t start, end;
	start = clock();

	float resolution = pointProcess::computeResolution(cloud.getKeypoints());
	//求曲率阈值，选中间值
	std::vector<int> indices_out;
	std::vector<int> indices_in;
	float curvatureThreshold = 0.0;
	std::vector<float> curvatureVec;
	curvatureVec.reserve(cloud.getKeypointNormals()->points.size());
	for (size_t num = 0; num < cloud.getKeypointNormals()->points.size(); ++num)
	{
		curvatureVec.push_back(cloud.getKeypointNormals()->points[num].curvature);
	}
	std::sort(curvatureVec.begin(), curvatureVec.end());
	curvatureThreshold = curvatureVec[cloud.getKeypointNormals()->points.size() - 1] * 0.5;

	pcl::PointCloud<pcl::VFHSignature308>::Ptr our_cvfhPtr(new pcl::PointCloud<pcl::VFHSignature308>());
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());

	pcl::OURCVFHEstimation< PointT, NormalT, pcl::VFHSignature308> our_cvfh;
	our_cvfh.setInputCloud(cloud.getKeypoints());
	our_cvfh.setInputNormals(cloud.getKeypointNormals());
	our_cvfh.setSearchMethod(tree);
	our_cvfh.setNormalizeBins(true);
	our_cvfh.setClusterTolerance(resolution * 2.5);
	our_cvfh.setCurvatureThreshold(curvatureThreshold);//曲率阈值，去除法向量用
	our_cvfh.setEPSAngleThreshold( 30.0 / 180.0 * M_PI );//相邻法向量最大偏差，聚类标准
	// Set the minimum axis ratio between the SGURF axes. At the disambiguation phase,
	// this will decide if additional Reference Frames need to be created, if ambiguous.
	our_cvfh.setAxisRatio(0.7);//用于跟消歧因子比较 ，其为【0,1】，1为完全歧义
	our_cvfh.compute(*our_cvfhPtr);

	cloud.setCVFH_features(our_cvfhPtr);
	end = clock();
	std::cout << "vfh size : " << our_cvfhPtr->points.size() << " , computeFeatures_CVFH() has finished in "
		<< (end - start) / CLOCKS_PER_SEC << " s \n";
}
//esf全局特征描述子
void pointProcess::computeFeatures_ESF(FeatureCloud &cloud)
{
	clock_t start, end;
	start = clock();
	//用于存储ESF描述符的对象。
	pcl::PointCloud < pcl::ESFSignature640 > ::Ptr  esfPtr(new pcl::PointCloud < pcl::ESFSignature640 > );
	pcl::ESFEstimation<PointT, pcl::ESFSignature640 > esf;
	esf.setInputCloud(cloud.getKeypoints());
	
	esf.compute(*esfPtr);

	cloud.setESF_features(esfPtr);

	end = clock();
	std::cout << "esf size : " << esfPtr->points.size() << " , computeFeatures_ESF() has finished in "
		<< (end - start) / CLOCKS_PER_SEC << " s \n";
}
//FPFH去除NAN点
void pointProcess::removeNANfromFPFH(
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr feature_descriptor,
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr nanremoved,
	FeatureCloud& cloud)
{
	std::vector<int> index;
	// Reserve enough space for the indices
	index.resize(feature_descriptor->points.size());
	size_t j = 0;

	for (size_t i = 0; i < feature_descriptor->points.size(); ++i)
	{
		if (!pcl_isfinite(feature_descriptor->points[i].histogram[0]))
			continue;
		nanremoved->points.push_back( feature_descriptor->points[i]);
		index[i] = static_cast<int>(i);
		j++;
	}
	if (j != feature_descriptor->points.size())
	{
		// Resize to the correct size
		nanremoved->points.resize(j);
		index.resize(j);
	}
	nanremoved->width = nanremoved->points.size();
	nanremoved->height = 1;
	nanremoved->is_dense = true;

	cloud.setFPFH_features(nanremoved);
	pcl::copyPointCloud(*cloud.getKeypoints(),index, *cloud.getKeypoints());
	pcl::copyPointCloud(*cloud.getKeypointNormals(), index, *cloud.getKeypointNormals());
}
//SHOT去除NAN点
void pointProcess::removeNANfromSHOT(
	pcl::PointCloud<pcl::SHOT352>::Ptr feature_descriptor,
	pcl::PointCloud<pcl::SHOT352>::Ptr nanremoved,
	FeatureCloud& cloud)
{
	std::vector<int> index;
	// Reserve enough space for the indices
	index.resize(feature_descriptor->points.size());
	size_t j = 0;

	for (size_t i = 0; i < feature_descriptor->points.size(); ++i)
	{
		if (!pcl_isfinite(feature_descriptor->points[i].descriptor[0]))
			continue;
		nanremoved->points.push_back ( feature_descriptor->points[i]);
		index[i] = static_cast<int>(i);
		j++;
	}
	if (j != feature_descriptor->points.size())
	{
		// Resize to the correct size
		nanremoved->points.resize(j);
		index.resize(j);
	}
	nanremoved->width = nanremoved->points.size();
	nanremoved->height = 1;
	nanremoved->is_dense = true;

	cloud.setSHOT_features(nanremoved);
	pcl::copyPointCloud(*cloud.getKeypoints(), index, *cloud.getKeypoints());
	pcl::copyPointCloud(*cloud.getKeypointNormals(), index, *cloud.getKeypointNormals());
}

// 基于RANSAC的形状提取
bool pointProcess::SACSegmentation_model(PointCloud::Ptr pointInput,
	pcl::ModelCoefficients::Ptr coefficients,
	pcl::PointIndices::Ptr inliers,
	pcl::SacModel modeltype ,
	int maxIteration ,
	float distancethreshold )
{
	clock_t start, end;
	start = clock();
	if (pointInput->points.empty())
	{
		std::cout << "Error: SACSegmentation_model() , the pointInput is null\n";
		return false;
	}
	//Create the segmentation object
	pcl::SACSegmentation<PointT> seg;
	seg.setOptimizeCoefficients(true);
	seg.setModelType(modeltype);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(maxIteration);
	seg.setDistanceThreshold(distancethreshold);

	seg.setInputCloud(pointInput);
	seg.segment(*inliers, *coefficients);

	if (inliers->indices.size() == 0)
	{
		PCL_ERROR("Could not estimate a model for the given dataset.");
		return false;
	}
	end = clock();
	std::cout << "SACSegmentation_model() has finished in "
		<< float(end - start) / CLOCKS_PER_SEC << " s" << std::endl;
}
//提取或去除索引的点云
void pointProcess::extractIndicesPoints(PointCloud::Ptr pointInput,
	PointCloud::Ptr pointOutput,
	pcl::PointIndices::Ptr inliers,
	bool extractNegative)
{
	pcl::ExtractIndices<PointT> extract;
	extract.setInputCloud(pointInput);
	extract.setIndices(inliers);
	//true 为提取索引， false 为剔除索引
	extract.setNegative(extractNegative);
	extract.filter(*pointOutput);
	if (!extractNegative)
		std::cout << "Extract the rest-component: "
		<< pointOutput->points.size() << std::endl;
	else
		std::cout << "Extract the indice-component: "
		<< pointOutput->points.size() << std::endl;

}
//基于欧氏距离的聚类
void pointProcess::EuclideanClusterExtraction(PointCloud::Ptr pointInput,
	std::vector<PointCloud::Ptr>& cloudListOutput,
	float clusterTolerance ,
	int minClusterSize ,
	int maxClusterSize )
{
	if (pointInput == nullptr || pointInput->points.empty())
	{
		std::cout << "Error: EuclideanClusterExtraction() , pointInput is empty!\n";
		return;
	}

	clock_t start, end;
	start = clock();
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
	std::vector<pcl::PointIndices> cluster_indices;

	pcl::EuclideanClusterExtraction<PointT> ec;
	ec.setInputCloud(pointInput);
	ec.setSearchMethod(tree);
	ec.setClusterTolerance(clusterTolerance);
	ec.setMinClusterSize(minClusterSize);
	ec.setMaxClusterSize(maxClusterSize);
	ec.extract(cluster_indices);

	int num = 1;
	std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin();
	for (; it != cluster_indices.end(); ++it)
	{
		PointCloud::Ptr cloud_cluster(new PointCloud());
		pcl::copyPointCloud(*pointInput, *it, *cloud_cluster);
		cloudListOutput.push_back(cloud_cluster);
		std::cout << "PointCloud representing the Cluster: " << num << " , "
			<< cloud_cluster->points.size() << " data points." << std::endl;
		num++;
	}

	end = clock();
	std::cout << "EuclideanClusterExtraction() has finished in "
		<< (float)(end - start) / CLOCKS_PER_SEC << " s " << std::endl;
}
//基于区域生长的聚类
void pointProcess::RegionGrowingClusterExtraction(PointCloud::Ptr pointInput,
	std::vector<PointCloud::Ptr>& cloudListOutput,
	Normals::Ptr normalInput,
	int minClusterSize,
	int maxClusterSize,
	int numofNeighbour ,
	float smoothThreshold ,
	float curvatureThreshold)
{
	clock_t start, end;
	start = clock();
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
	pcl::RegionGrowing<PointT, NormalT> reg;
	reg.setMinClusterSize(minClusterSize);
	reg.setMaxClusterSize(maxClusterSize);
	reg.setSearchMethod(tree);
	reg.setNumberOfNeighbours(numofNeighbour);
	reg.setInputCloud(pointInput);
	reg.setInputNormals(normalInput);
	reg.setSmoothnessThreshold(smoothThreshold);

	std::vector<pcl::PointIndices> cluster_indices;
	reg.extract(cluster_indices);
	std::cout << "Number of clusters is equal to " << cluster_indices.size() << std::endl;

	int j = 1;
	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{
		PointCloud::Ptr cloud_cluster(new PointCloud);
		for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
		{
			cloud_cluster->points.push_back(pointInput->points[*pit]);
		}
		cloud_cluster->width = cloud_cluster->points.size();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		std::cout << "PointCloud representing the Cluster: " << j << " , " << cloud_cluster->points.size() << " data points." << std::endl;
		j++;
		cloudListOutput.push_back(cloud_cluster);

	}
	end = clock();
	std::cout << "RegionGrowingClusterExtraction has finished in "
		<< float(end - start) / CLOCKS_PER_SEC << " s \n";
}
//翻版区域生长
void pointProcess::extractEuclideanClustersSmooth(
	const pcl::PointCloud<pcl::PointNormal> &cloud,
	const pcl::PointCloud<pcl::PointNormal> &normals,
	float tolerance,
	const pcl::search::Search<pcl::PointNormal>::Ptr &tree,
	std::vector<pcl::PointIndices> &clusters,
	double eps_angle,
	unsigned int min_pts_per_cluster,
	unsigned int max_pts_per_cluster)
{
	clock_t begin = clock();

	if (tree->getInputCloud()->points.size() != cloud.points.size())
	{
		PCL_ERROR("[pcl::extractEuclideanClusters] Tree built for a different point cloud dataset (%lu) than the input cloud (%lu)!\n", tree->getInputCloud()->points.size(), cloud.points.size());
		return;
	}
	if (cloud.points.size() != normals.points.size())
	{
		PCL_ERROR("[pcl::extractEuclideanClusters] Number of points in the input point cloud (%lu) different than normals (%lu)!\n", cloud.points.size(), normals.points.size());
		return;
	}
	// Create a bool vector of processed point indices, and initialize it to false
	std::vector<bool> processed(cloud.points.size(), false);
	std::vector<int> nn_indices;
	std::vector<float> nn_distances;
	// Process all points in the indices vector
	for (int i = 0; i < static_cast<int> (cloud.points.size()); ++i)
	{
		if (processed[i])
			continue;

		std::vector<unsigned int> seed_queue;
		int sq_idx = 0;
		seed_queue.push_back(i);

		processed[i] = true;

		while (sq_idx < static_cast<int> (seed_queue.size()))
		{
			// Search for sq_idx,找不到就下一个种子点
			if (!tree->radiusSearch(seed_queue[sq_idx], tolerance, nn_indices, nn_distances))
			{
				sq_idx++;
				continue;
			}

			for (size_t j = 1; j < nn_indices.size(); ++j) // nn_indices[0] should be sq_idx
			{
				if (processed[nn_indices[j]]) // Has this point been processed before ?
					continue;

				//processed[nn_indices[j]] = true;
				// [-1;1]

				double dot_p = normals.points[seed_queue[sq_idx]].normal[0] * normals.points[nn_indices[j]].normal[0]
					+ normals.points[seed_queue[sq_idx]].normal[1] * normals.points[nn_indices[j]].normal[1]
					+ normals.points[seed_queue[sq_idx]].normal[2] * normals.points[nn_indices[j]].normal[2];

				if (fabs(acos(dot_p)) < eps_angle)
				{
					processed[nn_indices[j]] = true;
					seed_queue.push_back(nn_indices[j]);
				}
			}

			sq_idx++;
		}

		// If this queue is satisfactory, add to the clusters
		if (seed_queue.size() >= min_pts_per_cluster && seed_queue.size() <= max_pts_per_cluster)
		{
			pcl::PointIndices r;
			r.indices.resize(seed_queue.size());
			for (size_t j = 0; j < seed_queue.size(); ++j)
				r.indices[j] = seed_queue[j];

			std::sort(r.indices.begin(), r.indices.end());
			//std::unique()返回的是不重复元素的末尾指针，即重复元素的头指针
			r.indices.erase(std::unique(r.indices.begin(), r.indices.end()), r.indices.end());//删除重复元素

			r.header = cloud.header;
			clusters.push_back(r); // We could avoid a copy by working directly in the vector
		}
	}

	clock_t end = clock();
	std::cout << "extractEuclideanClustersSmooth() consumed : " << float(end - begin) / CLOCKS_PER_SEC
		<< " s , cluster size : " << clusters.size() << std::endl;
}
// 最小包围盒
void pointProcess::OrientedBoundingBox(PointCloud::Ptr pointInput,
	Eigen::Vector3f &whd,//长宽高
	Eigen::Vector3f &bboxT,//平移
	Eigen::Quaternionf &bboxQ,//旋转
	float scalar,
	PointT& pcX, PointT& pcY, PointT& pcZ, PointT& initialoriginalPoint)
{
	if (pointInput == nullptr || pointInput->points.empty())
	{
		std::cout << "\nError: OrientedBoundingBox() --- the pointInput is empty!\n";
		return;
	}
	
	PointCloud::Ptr cloudInput(new PointCloud());
	pcl::copyPointCloud(*pointInput, *cloudInput);
	//重心
	Eigen::Vector4f pcaCentroid;
	pcl::compute3DCentroid(*cloudInput, pcaCentroid);
	//协方差矩阵
	Eigen::Matrix3f covariance;
	pcl::computeCovarianceMatrixNormalized(*cloudInput, pcaCentroid, covariance);
	//特征向量
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
	Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
	Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();
	//校正主方向间垂直
	eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));
	eigenVectorsPCA.col(0) = eigenVectorsPCA.col(1).cross(eigenVectorsPCA.col(2));
	eigenVectorsPCA.col(1) = eigenVectorsPCA.col(2).cross(eigenVectorsPCA.col(0));

	// 另一种计算点云协方差矩阵特征值和特征向量的方式:通过pcl中的pca接口，
	//如下，这种情况得到的特征向量相似特征向量
	/*
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPCAprojection(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PCA<pcl::PointXYZ> pca;
	pca.setInputCloud(cloudSegmented);
	pca.project(*cloudSegmented, *cloudPCAprojection);
	std::cerr << std::endl << "EigenVectors: " << pca.getEigenVectors() << std::endl;//计算特征向量
	std::cerr << std::endl << "EigenValues: " << pca.getEigenValues() << std::endl;//计算特征值
	*/
	Eigen::Matrix4f tm = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f tm_inv = Eigen::Matrix4f::Identity();
	tm.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();//R
	tm.block<3, 1>(0, 3) =  - 1.0f * (eigenVectorsPCA.transpose()) * (pcaCentroid.head(3));//-R*T
	tm_inv = tm.inverse();

	PointCloud::Ptr transformedCloud(new PointCloud());
	pcl::transformPointCloud(*cloudInput, *transformedCloud, tm);

	PointT min_p1, max_p1;
	Eigen::Vector3f c1, c;
	pcl::getMinMax3D(*transformedCloud, min_p1, max_p1);
	c1 = 0.5f * (min_p1.getVector3fMap() + max_p1.getVector3fMap());//两点的对称中心,形心

	Eigen::Affine3f tm_inv_aff(tm_inv);
	pcl::transformPoint(c1, c, tm_inv_aff);//原点云的对称中心,形心

	Eigen::Vector3f whd1;
	whd1 = max_p1.getVector3fMap() - min_p1.getVector3fMap();
	whd = whd1;
	scalar = (whd1(0) + whd1(1) + whd1(2)) / 3;//点云平均尺度

	const Eigen::Quaternionf bboxQ1(Eigen::Quaternionf::Identity());
	const Eigen::Vector3f bboxT1(c1);

	bboxQ = tm_inv.block<3, 3>(0, 0);
	bboxT = c;

	//变换到原点的点云主方向
	//PointT originalPoint;
	//originalPoint.x = 0.0;
	//originalPoint.y = 0.0;
	//originalPoint.z = 0.0;
	//Eigen::Vector3f px, py, pz;
	//Eigen::Affine3f tm_aff(tm);
	//pcl::transformVector(eigenVectorsPCA.col(0), px, tm_aff);
	//pcl::transformVector(eigenVectorsPCA.col(1), py, tm_aff);
	//pcl::transformVector(eigenVectorsPCA.col(2), pz, tm_aff);

	//PointT pcaX, pcaY, pcaZ;
	//pcaX.x = scalar * px(0);
	//pcaX.y = scalar * px(1);
	//pcaX.z = scalar * px(2);

	//pcaY.x = scalar * py(0);
	//pcaY.y = scalar * py(1);
	//pcaY.z = scalar * py(2);

	//pcaZ.x = scalar * pz(0);
	//pcaZ.y = scalar * pz(1);
	//pcaZ.z = scalar * pz(2);

	//原点云方向包围盒坐标轴方向，三个特征向量的方向
	initialoriginalPoint.x = pcaCentroid(0);
	initialoriginalPoint.y = pcaCentroid(1);
	initialoriginalPoint.z = pcaCentroid(2);

	pcX.x = scalar * eigenVectorsPCA(0, 0) + initialoriginalPoint.x;//加回平移量
	pcX.y = scalar * eigenVectorsPCA(1, 0) + initialoriginalPoint.y;
	pcX.z = scalar * eigenVectorsPCA(2, 0) + initialoriginalPoint.z;

	pcY.x = scalar * eigenVectorsPCA(0, 1) + initialoriginalPoint.x;
	pcY.y = scalar * eigenVectorsPCA(1, 1) + initialoriginalPoint.y;
	pcY.z = scalar * eigenVectorsPCA(2, 1) + initialoriginalPoint.z;

	pcZ.x = scalar * eigenVectorsPCA(0, 2) + initialoriginalPoint.x;
	pcZ.y = scalar * eigenVectorsPCA(1, 2) + initialoriginalPoint.y;
	pcZ.z = scalar * eigenVectorsPCA(2, 2) + initialoriginalPoint.z;
}