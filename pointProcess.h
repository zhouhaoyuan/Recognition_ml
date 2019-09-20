#pragma once

#include "FeatureCloud.h"



class pointProcess
{
public:
	pointProcess();
	~pointProcess();
	//����
	static void computeCentroid(const PointCloud::Ptr pointInput, Eigen::Vector3f &mean);
	//�����ļ���ȥ����
	static void remove_Centroid(PointCloud::Ptr point, Eigen::Vector3f& mean);
	//�ֱ���
	static float computeResolution(PointCloud::Ptr pInput);
	// ��С��Χ��
	static void OrientedBoundingBox(PointCloud::Ptr pointInput,
		Eigen::Vector3f &whd,
		Eigen::Vector3f &bboxT,
		Eigen::Quaternionf &bboxQ,
		float scalar,
		PointT& pcX, PointT& pcY, PointT& pcZ, PointT& initialoriginalPoint);
	//��������������ת����
	static Eigen::Matrix3f computeRotation(Eigen::Vector3f &a, Eigen::Vector3f &b);	//�����˲�
	//�����˲�
	void Uniform_Filter(PointCloud::Ptr input,PointCloud::Ptr output
		,float uniform_Radius , pcl::PointIndices filterIndices);
	static void VoxelGrid_Filter(PointCloud::Ptr input, PointCloud::Ptr output,
		float leafsize = 1.0);
	//����ͳ��ѧ�˲�
	static void StatisticalOutlierRemoval_Filter(
		PointCloud::Ptr input,
		PointCloud::Ptr output,
		int K = 30,
		float stddevMulThresh = 1.0);
	//ֱͨ�˲�
	static void PassThrough_Filter(
		PointCloud::Ptr input,
		PointCloud::Ptr output,
		std::string axis,
		float upBound,
		float downBound,
		bool negative = false);
	//˫���˲�
	static void Bilateral_Filter(
		PointCloud::Ptr input,
		PointCloud::Ptr output,
		double sigmas,
		double sigmar
	);
	//����������
	void computeSurfaceNormals(
		FeatureCloud& cloud,
		int K = 50,
		float radius = 0,
		int numofthreads = 4);
	// ������ȥ��NAN��
	void removeNANfromNormal(FeatureCloud &cloud);
	//��Ȩ����������
	Eigen::Vector4f computeWeightedNormal(PointCloud::Ptr inputCloud, Eigen::Vector4f& point);
	//FPFH����
	void computeFeatures_FPFH(FeatureCloud &cloud, float R);
	//FPFHȥ��NAN��
	void removeNANfromFPFH(
		pcl::PointCloud<pcl::FPFHSignature33>::Ptr feature_descriptor,
		pcl::PointCloud<pcl::FPFHSignature33>::Ptr nanremoved,
		FeatureCloud& cloud);
	//SHOT����
	void computeFeatures_SHOT(FeatureCloud &cloud, float R);
	//SHOTȥ��NAN��
	void removeNANfromSHOT(
		pcl::PointCloud<pcl::SHOT352>::Ptr feature_descriptor,
		pcl::PointCloud<pcl::SHOT352>::Ptr nanremoved,
		FeatureCloud& cloud);
	//VFHȫ������������
	void computeFeatures_VFH(FeatureCloud &cloud);
	//CVFHȫ������������
	void computeFeatures_CVFH(FeatureCloud& cloud);
	// ����SPFH
	void computePointSPFHSignature(PointCloudNormal::Ptr cloud,
		pcl::PointIndices indices,
		Eigen::Vector4f& centroid_p,
		Eigen::Vector4f& centroid_normal,
		Eigen::VectorXf& hist_f1_,
		Eigen::VectorXf& hist_f2_,
		Eigen::VectorXf& hist_f3_,
		Eigen::VectorXf& hist_f4_);
	// ����Ľ���VFH
	void computeImproveVFHfeature(PointCloudNormal::Ptr pointcloudnormal,
		pcl::PointIndices indices,
		Eigen::Vector4f& xyz_centroid,
		Eigen::Vector4f& normal_centroid,
		pcl::PointCloud<pcl::VFHSignature308>& result);
	//�Ľ���CVFHȫ������������
	void computeFeatures_ImprovedCVFH(FeatureCloud& cloud);

	//OUR_CVFHȫ������������
	void computeFeatures_OUR_CVFH(FeatureCloud &cloud);

	//esfȫ������������
	void computeFeatures_ESF(FeatureCloud &cloud);

	// ����RANSAC����״��ȡ
	bool SACSegmentation_model(PointCloud::Ptr pointInput,
		pcl::ModelCoefficients::Ptr coefficients,
		pcl::PointIndices::Ptr inliers,
		pcl::SacModel modeltype = pcl::SACMODEL_PLANE,
		int maxIteration = 100,
		float distancethreshold = 1.0);
	//��ȡ��ȥ�������ĵ���
	void extractIndicesPoints(PointCloud::Ptr pointInput,
		PointCloud::Ptr pointOutput,
		pcl::PointIndices::Ptr inliers,
		bool extractNegative);
	//����ŷ�Ͼ���ľ���
	void EuclideanClusterExtraction(PointCloud::Ptr pointInput,
		std::vector<PointCloud::Ptr>& cloudListOutput,
		float clusterTolerance = 0.02,
		int minClusterSize = 100,
		int maxClusterSize = 1000);
	//�������������ľ���
	void RegionGrowingClusterExtraction(PointCloud::Ptr pointInput,
		std::vector<PointCloud::Ptr>& cloudListOutput,
		Normals::Ptr normalInput,
		int minClusterSize,
		int maxClusterSize,
		int numofNeighbour = 30,
		float smoothThreshold = 3.0 / 180.0 * M_PI,
		float curvatureThreshold = 1.0);
	//������������
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

