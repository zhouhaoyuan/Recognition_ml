#pragma once

#include "FeatureCloud.h"



class pointProcess
{
public:
	pointProcess();
	~pointProcess();
	//质心
	static void computeCentroid(const PointCloud::Ptr pointInput, Eigen::Vector3f &mean);
	//求重心及减去重心
	static void remove_Centroid(PointCloud::Ptr point, Eigen::Vector3f& mean);
	//分辨率
	static float computeResolution(PointCloud::Ptr pInput);
	// 最小包围盒
	static void OrientedBoundingBox(PointCloud::Ptr pointInput,
		Eigen::Vector3f &whd,
		Eigen::Vector3f &bboxT,
		Eigen::Quaternionf &bboxQ,
		float scalar,
		PointT& pcX, PointT& pcY, PointT& pcZ, PointT& initialoriginalPoint);
	//计算两向量的旋转矩阵
	static Eigen::Matrix3f computeRotation(Eigen::Vector3f &a, Eigen::Vector3f &b);	//体素滤波
	//均匀滤波
	void Uniform_Filter(PointCloud::Ptr input,PointCloud::Ptr output
		,float uniform_Radius , pcl::PointIndices filterIndices);
	static void VoxelGrid_Filter(PointCloud::Ptr input, PointCloud::Ptr output,
		float leafsize = 1.0);
	//基于统计学滤波
	static void StatisticalOutlierRemoval_Filter(
		PointCloud::Ptr input,
		PointCloud::Ptr output,
		int K = 30,
		float stddevMulThresh = 1.0);
	//直通滤波
	static void PassThrough_Filter(
		PointCloud::Ptr input,
		PointCloud::Ptr output,
		std::string axis,
		float upBound,
		float downBound,
		bool negative = false);
	//双边滤波
	static void Bilateral_Filter(
		PointCloud::Ptr input,
		PointCloud::Ptr output,
		double sigmas,
		double sigmar
	);
	//法向量计算
	void computeSurfaceNormals(
		FeatureCloud& cloud,
		int K = 50,
		float radius = 0,
		int numofthreads = 4);
	// 法向量去除NAN点
	void removeNANfromNormal(FeatureCloud &cloud);
	//加权法向量计算
	Eigen::Vector4f computeWeightedNormal(PointCloud::Ptr inputCloud, Eigen::Vector4f& point);
	//FPFH计算
	void computeFeatures_FPFH(FeatureCloud &cloud, float R);
	//FPFH去除NAN点
	void removeNANfromFPFH(
		pcl::PointCloud<pcl::FPFHSignature33>::Ptr feature_descriptor,
		pcl::PointCloud<pcl::FPFHSignature33>::Ptr nanremoved,
		FeatureCloud& cloud);
	//SHOT计算
	void computeFeatures_SHOT(FeatureCloud &cloud, float R);
	//SHOT去除NAN点
	void removeNANfromSHOT(
		pcl::PointCloud<pcl::SHOT352>::Ptr feature_descriptor,
		pcl::PointCloud<pcl::SHOT352>::Ptr nanremoved,
		FeatureCloud& cloud);
	//VFH全局特征描述子
	void computeFeatures_VFH(FeatureCloud &cloud);
	//CVFH全局特征描述子
	void computeFeatures_CVFH(FeatureCloud& cloud);
	// 计算SPFH
	void computePointSPFHSignature(PointCloudNormal::Ptr cloud,
		pcl::PointIndices indices,
		Eigen::Vector4f& centroid_p,
		Eigen::Vector4f& centroid_normal,
		Eigen::VectorXf& hist_f1_,
		Eigen::VectorXf& hist_f2_,
		Eigen::VectorXf& hist_f3_,
		Eigen::VectorXf& hist_f4_);
	// 计算改进的VFH
	void computeImproveVFHfeature(PointCloudNormal::Ptr pointcloudnormal,
		pcl::PointIndices indices,
		Eigen::Vector4f& xyz_centroid,
		Eigen::Vector4f& normal_centroid,
		pcl::PointCloud<pcl::VFHSignature308>& result);
	//改进的CVFH全局特征描述子
	void computeFeatures_ImprovedCVFH(FeatureCloud& cloud);

	//OUR_CVFH全局特征描述子
	void computeFeatures_OUR_CVFH(FeatureCloud &cloud);

	//esf全局特征描述子
	void computeFeatures_ESF(FeatureCloud &cloud);

	// 基于RANSAC的形状提取
	bool SACSegmentation_model(PointCloud::Ptr pointInput,
		pcl::ModelCoefficients::Ptr coefficients,
		pcl::PointIndices::Ptr inliers,
		pcl::SacModel modeltype = pcl::SACMODEL_PLANE,
		int maxIteration = 100,
		float distancethreshold = 1.0);
	//提取或去除索引的点云
	void extractIndicesPoints(PointCloud::Ptr pointInput,
		PointCloud::Ptr pointOutput,
		pcl::PointIndices::Ptr inliers,
		bool extractNegative);
	//基于欧氏距离的聚类
	void EuclideanClusterExtraction(PointCloud::Ptr pointInput,
		std::vector<PointCloud::Ptr>& cloudListOutput,
		float clusterTolerance = 0.02,
		int minClusterSize = 100,
		int maxClusterSize = 1000);
	//基于区域生长的聚类
	void RegionGrowingClusterExtraction(PointCloud::Ptr pointInput,
		std::vector<PointCloud::Ptr>& cloudListOutput,
		Normals::Ptr normalInput,
		int minClusterSize,
		int maxClusterSize,
		int numofNeighbour = 30,
		float smoothThreshold = 3.0 / 180.0 * M_PI,
		float curvatureThreshold = 1.0);
	//翻版区域生长
	void extractEuclideanClustersSmooth(
		const pcl::PointCloud<pcl::PointNormal> &cloud,
		const pcl::PointCloud<pcl::PointNormal> &normals,
		float tolerance,
		const pcl::search::Search<pcl::PointNormal>::Ptr &tree,
		std::vector<pcl::PointIndices> &clusters,
		double eps_angle,
		unsigned int min_pts_per_cluster,
		unsigned int max_pts_per_cluster);
};

