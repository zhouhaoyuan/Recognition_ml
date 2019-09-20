#include "FeatureCloud.h"



FeatureCloud::FeatureCloud()
{
    PointCloudPtr.reset(new PointCloud());
	keypointsPtr.reset(new PointCloud());
	NormalsPtr.reset(new Normals());
	keypointNormalsPtr.reset(new Normals());
	PointCloudNormalPtr.reset(new PointCloudNormal());
	VFH308_featuresPtr.reset(new pcl::PointCloud<pcl::VFHSignature308>());
	CVFH_featuresPtr.reset(new pcl::PointCloud<pcl::VFHSignature308>());
	SHOT352_featuresPtr.reset(new pcl::PointCloud<pcl::SHOT352>());
	FPFH33_featuresPtr.reset(new pcl::PointCloud<pcl::FPFHSignature33>());
}


FeatureCloud::~FeatureCloud()
{
}
