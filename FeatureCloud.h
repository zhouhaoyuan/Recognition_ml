#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>

#include <pcl/common/common.h>
#include <pcl/common/angles.h>
#include <pcl/common/transforms.h>
//kdTree
#include <pcl/search/kdtree.h>
//filters
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/fast_bilateral_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>//提取滤波器
#include <pcl/filters/project_inliers.h>//投影滤波类头文件
// key points
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/keypoints/harris_3d.h>
//feature
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/features/vfh.h>
#include <pcl/features/cvfh.h>
#include <pcl/features/pfh_tools.h>
#include <pcl/features/our_cvfh.h>
#include  <pcl/features/esf.h>

//segmentation
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/segmentation/supervoxel_clustering.h>

#include <pcl/features/moment_of_inertia_estimation.h>


//visualizer
#include <pcl/visualization/pcl_visualizer.h>
//#include "vtkAutoInit.h" 
//VTK_MODULE_INIT(vtkRenderingOpenGL);

typedef pcl::PointXYZ PointT;
typedef pcl::Normal NormalT;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<NormalT> Normals;
typedef pcl::PointCloud<PointNormalT> PointCloudNormal;

class FeatureCloud
{
public:
	FeatureCloud();
	~FeatureCloud();

	//set point
	void setPointCloud(PointCloud::Ptr cloud) { PointCloudPtr = cloud; }
	//set keypoint
	void setKeypoints(PointCloud::Ptr cloud) { keypointsPtr = cloud; }
	//set point normals
	void setNormals(Normals::Ptr normals) { NormalsPtr = normals; }
	//set keypoint normals
	void setKeypointNormals(Normals::Ptr normals) { keypointNormalsPtr = normals; }
	//set pointcloudnormals
	void setPointCloudNormals(PointCloudNormal::Ptr pointcloudnormals) { PointCloudNormalPtr = pointcloudnormals; }
	//set vfh
	void setVFH_features(pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhsPtr) { VFH308_featuresPtr = vfhsPtr; }
	//set cvfh
	void setCVFH_features(pcl::PointCloud<pcl::VFHSignature308>::Ptr CvfhsPtr) { CVFH_featuresPtr = CvfhsPtr; }
	//set fpfh
	void setFPFH_features(pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhsPtr) { FPFH33_featuresPtr = fpfhsPtr; }
	//set shot
	void setSHOT_features(pcl::PointCloud<pcl::SHOT352>::Ptr shotsPtr) { SHOT352_featuresPtr = shotsPtr; }
	//set ESF
	void setESF_features(pcl::PointCloud<pcl::ESFSignature640>::Ptr esfPtr) { ESF_featuresPtr = esfPtr; }

	//get point
	PointCloud::Ptr getPointCloud() { return  PointCloudPtr; }
	//get keypoint
	PointCloud::Ptr getKeypoints() { return  keypointsPtr; }
	//get normal
	Normals::Ptr getNormals() { return  NormalsPtr; }
	//get keypoint normals
	Normals::Ptr getKeypointNormals() { return keypointNormalsPtr; }
	//get pointcloudnormal
	PointCloudNormal::Ptr getPointCloudNormal() { return  PointCloudNormalPtr; }
	//get vfh
	pcl::PointCloud<pcl::VFHSignature308>::Ptr getVFH_features() { return  VFH308_featuresPtr; }
	//get cvfh
	pcl::PointCloud<pcl::VFHSignature308>::Ptr getCVFH_features() { return  CVFH_featuresPtr; }
	//get fpfh
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr getFPFH_features() { return FPFH33_featuresPtr; }
	//get shot
	pcl::PointCloud<pcl::SHOT352>::Ptr getSHOT_features() { return SHOT352_featuresPtr; }
	//get esf
	pcl::PointCloud<pcl::ESFSignature640>::Ptr getESF_features() { return ESF_featuresPtr; }

private:
	PointCloud::Ptr PointCloudPtr;
	PointCloud::Ptr keypointsPtr;
	Normals::Ptr NormalsPtr;
	Normals::Ptr keypointNormalsPtr;
	PointCloudNormal::Ptr PointCloudNormalPtr;

	pcl::PointCloud<pcl::FPFHSignature33>::Ptr FPFH33_featuresPtr;
	pcl::PointCloud<pcl::SHOT352>::Ptr SHOT352_featuresPtr;
	pcl::PointCloud<pcl::VFHSignature308>::Ptr VFH308_featuresPtr;
	pcl::PointCloud<pcl::VFHSignature308>::Ptr CVFH_featuresPtr;
	pcl::PointCloud<pcl::ESFSignature640>::Ptr ESF_featuresPtr;

};

