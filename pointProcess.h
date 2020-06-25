#pragma once

#include "FeatureCloud.h"



class pointProcess
{
public:
	pointProcess();
	~pointProcess();
	//����
	static void computeCentroid(const PointCloud::Ptr pointInput, Eigen::Vector3f &mean);
	//��ȥ����
	static void remove_Centroid(PointCloud::Ptr point, Eigen::Vector3f& mean);
	//��һ��
	static void normalizePoints(PointCloud::Ptr pointCloud, Eigen::Vector3f& mean,
		float global_scale);
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
	static void Uniform_Filter(PointCloud::Ptr input,PointCloud::Ptr output
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
	//
	static PointCloudNormal MLSreSampling(pcl::PointCloud<PointT>::Ptr inputP);
	//����������
	static void computeNormals(
		PointCloud::Ptr input,
		Normals::Ptr output,
		int K = 50,
		float radius = 0,
		int numofthreads = 4);
	//����������
	static void computeSurfaceNormals(
		FeatureCloud& cloud,
		int K = 50,
		float radius = 0,
		int numofthreads = 4);
	// ������ȥ��NAN��
	static void removeNANfromNormal(FeatureCloud &cloud);
	//��Ȩ����������
	static Eigen::Vector4f computeWeightedNormal(PointCloud::Ptr inputCloud, Eigen::Vector4f& point);
	//FPFH����
	static void computeFeatures_FPFH(FeatureCloud &cloud, float R);
	//FPFHȥ��NAN��
	static void removeNANfromFPFH(
		pcl::PointCloud<pcl::FPFHSignature33>::Ptr feature_descriptor,
		pcl::PointCloud<pcl::FPFHSignature33>::Ptr nanremoved,
		FeatureCloud& cloud);
	//SHOT����
	static void computeFeatures_SHOT(FeatureCloud &cloud, float R);
	//SHOTȥ��NAN��
	static void removeNANfromSHOT(
		pcl::PointCloud<pcl::SHOT352>::Ptr feature_descriptor,
		pcl::PointCloud<pcl::SHOT352>::Ptr nanremoved,
		FeatureCloud& cloud);
	//VFHȫ������������
	static void computeFeatures_VFH(FeatureCloud &cloud);
	//CVFHȫ������������
	static void computeFeatures_CVFH(FeatureCloud& cloud);
	// ����SPFH
	static void computePointSPFHSignature(PointCloudNormal::Ptr cloud,
		pcl::PointIndices indices,
		Eigen::Vector4f& centroid_p,
		Eigen::Vector4f& centroid_normal,
		Eigen::VectorXf& hist_f1_,
		Eigen::VectorXf& hist_f2_,
		Eigen::VectorXf& hist_f3_,
		Eigen::VectorXf& hist_f4_);
	// ����Ľ���VFH
	static void computeImproveVFHfeature(PointCloudNormal::Ptr pointcloudnormal,
		pcl::PointIndices indices,
		Eigen::Vector4f& xyz_centroid,
		Eigen::Vector4f& normal_centroid,
		pcl::PointCloud<pcl::VFHSignature308>& result);
	//�Ľ���CVFHȫ������������
	static void computeFeatures_ImprovedCVFH(FeatureCloud& cloud);

	//OUR_CVFHȫ������������
	static void computeFeatures_OUR_CVFH(FeatureCloud &cloud);

	//esfȫ������������
	static void computeFeatures_ESF(FeatureCloud &cloud);

	// ����RANSAC����״��ȡ
	static bool SACSegmentation_model(PointCloud::Ptr pointInput,
		pcl::ModelCoefficients::Ptr coefficients,
		pcl::PointIndices::Ptr inliers,
		pcl::SacModel modeltype = pcl::SACMODEL_PLANE,
		int maxIteration = 100,
		float distancethreshold = 1.0);
	//��ȡ��ȥ�������ĵ���
	static void extractIndicesPoints(PointCloud::Ptr pointInput,
		PointCloud::Ptr pointOutput,
		pcl::PointIndices::Ptr inliers,
		bool extractNegative = false);
	//��ȡ��ȥ�������ķ���
	static void extractIndicesNormals(Normals::Ptr pointInput,
		Normals::Ptr pointOutput,
		pcl::PointIndices::Ptr inliers,
		bool extractNegative = false);
	//����ŷ�Ͼ���ľ���
	static void EuclideanClusterExtraction(PointCloud::Ptr pointInput,
		std::vector<PointCloud::Ptr>& cloudListOutput,
		std::vector<pcl::PointIndices>& indices,
		float clusterTolerance = 0.02,
		int minClusterSize = 100,
		int maxClusterSize = 1000);
	//�������������ľ���
	static void RegionGrowingClusterExtraction(PointCloud::Ptr pointInput,
		std::vector<PointCloud::Ptr>& cloudListOutput,
		std::vector<pcl::PointIndices>& indices,
		Normals::Ptr normalInput,
		int minClusterSize,
		int maxClusterSize,
		int numofNeighbour = 30,
		float smoothThreshold = 3.0 / 180.0 * M_PI,
		float curvatureThreshold = 1.0);
	//������������
	static void extractEuclideanClustersSmooth(
		const pcl::PointCloud<pcl::PointNormal> &cloud,
		const pcl::PointCloud<pcl::PointNormal> &normals,
		float tolerance,
		const pcl::search::Search<pcl::PointNormal>::Ptr &tree,
		std::vector<pcl::PointIndices> &clusters,
		double eps_angle,
		unsigned int min_pts_per_cluster,
		unsigned int max_pts_per_cluster);

	//kdTree���������(PointT)
	static void getNearestIndices(
		const PointCloud::Ptr cloudIn,
		const PointCloud::Ptr cloudQuery,
		PointCloud::Ptr cloudResult,
		pcl::PointIndices::Ptr indicesPtr);
	//��Ӧ��Թ���
	static void correspondence_estimation(
		FPFH_features::Ptr source_cloud,
		FPFH_features::Ptr target_cloud,
		pcl::Correspondences &all_corres);
	//��Ӧ����޳�
	static void correspondences_rejection(
		const PointCloud::Ptr source_cloud,
		const PointCloud::Ptr target_cloud,
		pcl::Correspondences &correspondences,
		pcl::Correspondences &inliers,
		int MaximumIterations, float Inlierthreshold);
	//��Ӧ����޳��� �Զ���Լ����
	static void advancedMatching(PointCloud::Ptr target, PointCloud::Ptr source,
		pcl::Correspondences &correspondences,
		pcl::Correspondences &inliers,
		float tupleScale,
		int tuple_max_cnt_);

	//SAC-IA
	static void SAC_IA_Transform(FeatureCloud &source_cloud,
		FeatureCloud &target_cloud,
		float minsampleDistance,
		int numofSample,
		int correspondenceRandomness,
		Eigen::Matrix4f& final_transformation);

	//ICP
	static float iterative_closest_points(std::string solver,
		bool flag_reciprocal, bool flag_ransac,
		FeatureCloud &source_cloud, FeatureCloud &target_cloud,
		float transEps, float corresDist, float EuclFitEps,
		float outlThresh, int maxIteration,
		Eigen::Matrix4f &final_transformation);

	//ICP
	static float ICP_pointclouds(
		PointCloud::Ptr cloudTarget,
		PointCloud::Ptr cloudSource,
		Eigen::Matrix4f& tranRes);

	//��ʾ���ƶ�Ӧ���
	static void showPointCloudCorrespondences(std::string viewerName,
		PointCloud::Ptr cloudTarget_,
		PointCloud::Ptr cloudSource_,
		pcl::Correspondences &corr_, int showThreshold);

	//��������������
	static void construct_PointNormal(FeatureCloud& targetCloud,
		FeatureCloud& sourceCloud);

};
