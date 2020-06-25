#include "cloudProcess.h"

using namespace mCP;

LARGE_INTEGER m_freq, m_timeStart, m_timeOff;

void Error(std::string s) {
	std::cout <<"Error ："<< s << std::endl;
}
void myTimeStart() {

	QueryPerformanceFrequency(&m_freq);
	QueryPerformanceCounter(&m_timeStart);
}
std::string myTimeOff() {

	QueryPerformanceCounter(&m_timeOff);
	double elapsedTime = (double)(m_timeOff.QuadPart - m_timeStart.QuadPart) / m_freq.QuadPart;
	std::stringstream ss;
	ss << elapsedTime * 1000;
	return ss.str() + "ms";
}

bool compareFirstIdx(std::pair<int, int> a, std::pair<int, int> b) {
	return a.first < b.first;
}

float mCP::cloudProcess::computeResolution(Points & input)
{
	myTimeStart();
	if (input.empty())
	{
		Error("computeResolution() , the input is null\n");
		return -1;
	}
	float resolution = 0.0;
	int n_points = 0;
	std::vector<int> indices(2);
	std::vector<float> dists(2);

	KDTree tree(flann::KDTreeSingleIndexParams(15));
	BuildKDTree<Eigen::Vector3f>(input, &tree);

	for (int i = 0; i < input.size(); ++i) {

		if (!std::isfinite<float>(input[i](0)) ||
			!std::isfinite<float>(input[i](1)) ||
			!std::isfinite<float>(input[i](2))) {
			continue;
		}
		SearchKDTree<Eigen::Vector3f>(&tree, input[i], indices, dists, 2);
		if (indices.size() == 2) {
			resolution += sqrt(dists[1]);
			++n_points;
		}
	}
	if (n_points != 0)
		resolution /= n_points;

	std::cout << "The pointCloud resolution : " << resolution<< " , "<< myTimeOff() << std::endl;
	return resolution;
}

void mCP::cloudProcess::getMinMax3D(Points & input, Eigen::Vector3f& minP, Eigen::Vector3f& maxP)
{
	Eigen::Array3f min_p, max_p;
	min_p.setConstant(FLT_MAX);
	max_p.setConstant(-FLT_MAX);

	for (size_t i = 0; i < input.size(); ++i)
	{
		// Check if the point is invalid
		if (!std::isfinite<float>(input[i](0)) ||
			!std::isfinite<float>(input[i](1)) ||
			!std::isfinite<float>(input[i](2))) {
			continue;
		}
		Eigen::Array3f pt = input[i].array();
		min_p = min_p.min<Eigen::Array3f>(pt);
		max_p = max_p.max<Eigen::Array3f>(pt);
	}
	minP(0) = min_p(0); minP(1) = min_p(1); minP(2) = min_p(2);
	maxP(0) = max_p(0); maxP(1) = max_p(1); maxP(2) = max_p(2);
}

unsigned int mCP::cloudProcess::computeMeanAndCovarianceMatrix(const Points& input, const std::vector<int> &indices, Eigen::Matrix<float, 4, 1>& centroid, Eigen::Matrix<float, 3, 3>& covariance_matrix)
{
	if (input.empty())
		return 0;

	centroid.setZero();
	covariance_matrix.setZero();
	Eigen::Matrix<float, 1, 9, Eigen::RowMajor> accu = Eigen::Matrix<float, 1, 9, Eigen::RowMajor>::Zero();
	int pointCount = 0;
	for (std::vector<int>::const_iterator it = indices.begin(); it != indices.end(); ++it)
	{
		if(!std::isfinite<float>(input[*it](0)) ||
		   !std::isfinite<float>(input[*it](1)) ||
		   !std::isfinite<float>(input[*it](2)))
		{
			continue;
		}
		accu[0] += input[*it].x() * input[*it].x();
		accu[1] += input[*it].x() * input[*it].y();
		accu[2] += input[*it].x() * input[*it].z();
		accu[3] += input[*it].y() * input[*it].y();
		accu[4] += input[*it].y() * input[*it].z();
		accu[5] += input[*it].z() * input[*it].z();
		accu[6] += input[*it].x();
		accu[7] += input[*it].y();
		accu[8] += input[*it].z();
		++pointCount;
	}
	accu /= static_cast<float> (pointCount);
	if (pointCount != 0)
	{
		centroid[0] = accu[6]; centroid[1] = accu[7]; centroid[2] = accu[8];
		centroid[3] = 1;
		covariance_matrix.coeffRef(0) = accu[0] - accu[6] * accu[6];
		covariance_matrix.coeffRef(1) = accu[1] - accu[6] * accu[7];
		covariance_matrix.coeffRef(2) = accu[2] - accu[6] * accu[8];
		covariance_matrix.coeffRef(4) = accu[3] - accu[7] * accu[7];
		covariance_matrix.coeffRef(5) = accu[4] - accu[7] * accu[8];
		covariance_matrix.coeffRef(8) = accu[5] - accu[8] * accu[8];
		covariance_matrix.coeffRef(3) = covariance_matrix.coeff(1);
		covariance_matrix.coeffRef(6) = covariance_matrix.coeff(2);
		covariance_matrix.coeffRef(7) = covariance_matrix.coeff(5);
	}
	return(static_cast<unsigned int> (pointCount));
}

void mCP::cloudProcess::solvePlaneParameters(const Eigen::Matrix3f & covariance_matrix, const Eigen::Vector4f & point, Eigen::Vector4f & plane_parameters, float & curvature)
{
	EIGEN_ALIGN16 Eigen::Vector3f::Scalar eigen_value;
	EIGEN_ALIGN16 Eigen::Vector3f eigen_vector;
	eigen33(covariance_matrix, eigen_value, eigen_vector);

	plane_parameters[0] = eigen_vector[0];
	plane_parameters[1] = eigen_vector[1];
	plane_parameters[2] = eigen_vector[2];

	// Compute the curvature surface change
	float eig_sum = covariance_matrix.coeff(0) + covariance_matrix.coeff(4) + covariance_matrix.coeff(8);
	if (eig_sum != 0)
		curvature = fabsf(eigen_value / eig_sum);
	else
		curvature = 0;

	plane_parameters[3] = 0;
	// Hessian form (D = nc . p_plane (centroid here) + p)
	plane_parameters[3] = -1 * plane_parameters.dot(point);
}


void mCP::cloudProcess::statisticalFilter(Points & input, Points & output, int k, float stddevMulThresh, std::vector<int>& removeIndices, bool negative)
{
	if (input.empty())
	{
		Error("statisticalFilter() , the input is empty");
		return;
	}
	myTimeStart();

	KDTree tree(flann::KDTreeSingleIndexParams(15));
	BuildKDTree<Eigen::Vector3f>(input, &tree);

	int beforeSize = input.size();
	std::vector<int> nn_indices(k);
	std::vector<float> nn_dists(k);
	std::vector<float> distances(input.size());
	std::vector<int> indices(input.size());
	removeIndices.resize(input.size());
	output.resize(input.size());
	
	int validnum = 0;
	for (int i = 0; i < input.size(); ++i) {

		if (!std::isfinite<float>(input[i](0)) ||
			!std::isfinite<float>(input[i](1)) ||
			!std::isfinite<float>(input[i](2))) {
			distances[i] = 0;
			continue;
		}
		SearchKDTree<Eigen::Vector3f>(&tree, input[i], nn_indices, nn_dists, k);
		
		double dist_sum = 0.0;
		for (int j = 0; j < k; ++j) {
			dist_sum += sqrt(nn_dists[j]);
		}
		distances[i] = static_cast<float>(dist_sum / k);
		++validnum;
	}

	double sum = 0, sq_sum = 0;
	for (int i = 0; i < distances.size(); ++i) {
		sum += distances[i];
		sq_sum += distances[i] * distances[i];
	}
	double mean = sum / static_cast<double>(validnum);
	double variance = (sq_sum - sum * sum / static_cast<double>(validnum)) / (static_cast<double>(validnum) - 1);
	double stddev = sqrt(variance);
	double dist_threshold = mean + stddevMulThresh * stddev;

	int rii = 0, oii = 0;
	for (int i = 0; i < input.size(); ++i) {
		if ( (!negative && distances[i] > dist_threshold) || 
			 (negative && distances[i] <= dist_threshold) ||
			fabs(distances[i]) <= 1e-6  ) {

			removeIndices[rii++] = i;
			continue;
		}
		indices[oii++] = i;
	}
	indices.resize(oii);
	removeIndices.resize(rii);

	for (int i = 0; i < indices.size(); ++i) {
		output[i] = input[indices[i]];
	}
	output.resize(indices.size());
	std::cout << "statisticalFilter() , before: " << beforeSize << " , after: " << output .size()<<" , " << myTimeOff() << std::endl;
}

void mCP::cloudProcess::voxelFilter(Points & input, Points & output, Eigen::Vector3f & leafSize, int min_points_per_voxel)
{
	if (input.empty())
	{
		Error("voxelFilter() , the input is empty");
		return;
	}
	myTimeStart();
	int beforeSize = input.size();

	Eigen::Vector3f min_p, max_p;
	getMinMax3D(input, min_p, max_p);

	Eigen::Array3f inverse_leaf_size_ = Eigen::Array3f::Ones() / leafSize.array();
	std::cout << "inverse_leaf_size_\n"<< inverse_leaf_size_ << std::endl;
	// 检查 leaf size 是否太小
	int64_t dx = static_cast<int64_t>((max_p[0] - min_p[0]) * inverse_leaf_size_[0]) + 1;
	int64_t dy = static_cast<int64_t>((max_p[1] - min_p[1]) * inverse_leaf_size_[1]) + 1;
	int64_t dz = static_cast<int64_t>((max_p[2] - min_p[2]) * inverse_leaf_size_[2]) + 1;

	if ((dx*dy*dz) > static_cast<int64_t>(INT_MAX))
	{
		Error("[voxelFilter] Leaf size is too small for the input dataset. Integer indices would overflow.");
		output = input;
		return;
	}
	// 栅格最大最小值，各轴分段数，索引用系数
	Eigen::Vector3i min_b_, max_b_, div_b_, divb_mul_;
	min_b_[0] = static_cast<int> (floor(min_p[0] * inverse_leaf_size_[0]));
	max_b_[0] = static_cast<int> (floor(max_p[0] * inverse_leaf_size_[0]));
	min_b_[1] = static_cast<int> (floor(min_p[1] * inverse_leaf_size_[1]));
	max_b_[1] = static_cast<int> (floor(max_p[1] * inverse_leaf_size_[1]));
	min_b_[2] = static_cast<int> (floor(min_p[2] * inverse_leaf_size_[2]));
	max_b_[2] = static_cast<int> (floor(max_p[2] * inverse_leaf_size_[2]));

	div_b_ = max_b_ - min_b_ + Eigen::Vector3i::Ones();
	divb_mul_ = Eigen::Vector3i(1, div_b_[0], div_b_[0] * div_b_[1]);

	// 同时存储栅格索引和点云索引
	std::vector<std::pair<int, int> > index_vector;
	index_vector.reserve(input.size());

	for (int i = 0; i < input.size(); ++i) {

		if (!std::isfinite<float>(input[i](0)) ||
			!std::isfinite<float>(input[i](1)) ||
			!std::isfinite<float>(input[i](2))) {
			continue;
		}
		int ijk0 = static_cast<int> (floor(input[i](0) * inverse_leaf_size_[0]) - static_cast<float> (min_b_[0]));
		int ijk1 = static_cast<int> (floor(input[i](1) * inverse_leaf_size_[1]) - static_cast<float> (min_b_[1]));
		int ijk2 = static_cast<int> (floor(input[i](2) * inverse_leaf_size_[2]) - static_cast<float> (min_b_[2]));

		// Compute the centroid leaf index
		int idx = ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 * divb_mul_[2];
		index_vector.push_back(std::make_pair(static_cast<unsigned int> (idx), i));
	}
	std::sort(index_vector.begin(), index_vector.end(), compareFirstIdx);

	unsigned int total = 0;
	unsigned int index = 0;
	std::vector<std::pair<unsigned int, unsigned int> > first_and_last_indices_vector;
	while (index < index_vector.size())
	{
		int i = index + 1;
		while (i < index_vector.size() && index_vector[i].first == index_vector[index].first)
			++i;
		if (i - index >= min_points_per_voxel)
		{
			++total;
			first_and_last_indices_vector.push_back(std::make_pair(index, i));
		}
		index = i;
	}
	std::cout << "first_and_last_indices_vector : " << first_and_last_indices_vector.size() << std::endl;

	output.resize(first_and_last_indices_vector.size());
	int num = 0;
	for (unsigned int j = 0; j < first_and_last_indices_vector.size(); ++j) {

		Eigen::Vector3f tmp = Eigen::Vector3f::Zero();
		unsigned int first = first_and_last_indices_vector[j].first;
		unsigned int last = first_and_last_indices_vector[j].second;
		for (unsigned int idx = first; idx < last; ++idx) {
			tmp += input[index_vector[idx].second];
		}
		tmp /= static_cast<float> (last - first);
		output[num++] = tmp;
	}
	std::cout << "voxelFilter() , before: " << beforeSize << " , after: " << output.size() << " , " << myTimeOff() << std::endl;
}

void mCP::cloudProcess::uniformSampling(Points & input, Points & output, Eigen::Vector3f & leafSize)
{
	if (input.empty())
	{
		Error("voxelFilter() , the input is empty");
		return;
	}
	myTimeStart();
	int beforeSize = input.size();

	Eigen::Vector3f min_p, max_p;
	getMinMax3D(input, min_p, max_p);

	Eigen::Array3f inverse_leaf_size_ = Eigen::Array3f::Ones() / leafSize.array();
	// 检查 leaf size 是否太小
	int64_t dx = static_cast<int64_t>((max_p[0] - min_p[0]) * inverse_leaf_size_[0]) + 1;
	int64_t dy = static_cast<int64_t>((max_p[1] - min_p[1]) * inverse_leaf_size_[1]) + 1;
	int64_t dz = static_cast<int64_t>((max_p[2] - min_p[2]) * inverse_leaf_size_[2]) + 1;

	if ((dx*dy*dz) > static_cast<int64_t>(INT_MAX))
	{
		Error("[uniformSampling] Leaf size is too small for the input dataset. Integer indices would overflow.");
		output = input;
		return;
	}
	// 栅格最大最小值，各轴分段数，索引用系数
	Eigen::Vector3i min_b_, max_b_, div_b_, divb_mul_;
	min_b_[0] = static_cast<int> (floor(min_p[0] * inverse_leaf_size_[0]));
	max_b_[0] = static_cast<int> (floor(max_p[0] * inverse_leaf_size_[0]));
	min_b_[1] = static_cast<int> (floor(min_p[1] * inverse_leaf_size_[1]));
	max_b_[1] = static_cast<int> (floor(max_p[1] * inverse_leaf_size_[1]));
	min_b_[2] = static_cast<int> (floor(min_p[2] * inverse_leaf_size_[2]));
	max_b_[2] = static_cast<int> (floor(max_p[2] * inverse_leaf_size_[2]));

	div_b_ = max_b_ - min_b_ + Eigen::Vector3i::Ones();
	divb_mul_ = Eigen::Vector3i(1, div_b_[0], div_b_[0] * div_b_[1]);

	std::unordered_map<int, int> leaves;
	int count = div_b_[0] * div_b_[1] * div_b_[2];
	for (int i = 0; i < count; ++i) {
		leaves.insert(std::make_pair(i, -1));
	}
	for (int i = 0; i < input.size(); ++i) {

		if (!std::isfinite<float>(input[i](0)) ||
			!std::isfinite<float>(input[i](1)) ||
			!std::isfinite<float>(input[i](2))) {
			continue;
		}
		Eigen::Vector3i ijk = Eigen::Vector3i::Zero();
		ijk[0] = static_cast<int> (floor(input[i](0) * inverse_leaf_size_[0]));
		ijk[1] = static_cast<int> (floor(input[i](1) * inverse_leaf_size_[1]));
		ijk[2] = static_cast<int> (floor(input[i](2) * inverse_leaf_size_[2]));

		int idx = (ijk - min_b_).dot(divb_mul_);
		if (leaves[idx] == -1) {
			leaves[idx] = i;
			continue;
		}
		float diff_cur = (input[i] - ijk.cast<float>()).squaredNorm();
		float diff_prev = (input[leaves[idx]] - ijk.cast<float>()).squaredNorm();
		if (diff_cur < diff_prev) {
			leaves[idx] = i;
		}
	}
	output.resize(count);
	int cp = 0;
	for (std::unordered_map<int, int>::const_iterator it = leaves.begin(); it != leaves.end(); ++it) {
		if (it->second != -1) {
			output[cp++] = input[it->second];
		}
	}
	output.resize(cp);
	std::cout << "uniformSampling() , before: " << beforeSize << " , after: " << output.size() << " , " << myTimeOff() << std::endl;
}

void mCP::cloudProcess::computeNormals(Points & input, Features & output, float radius, int numofthreads, Eigen::Vector3f vp)
{
	if (input.empty())
	{
		Error("computeNormals() , the input is empty");
		return;
	}
	myTimeStart();

	std::vector<int> indices(input.size());
	std::vector<float> dists(input.size());
	output.resize(input.size());

	KDTree tree(flann::KDTreeSingleIndexParams(15));
	BuildKDTree<Eigen::Vector3f>(input, &tree);

	output.resize(input.size());

#pragma omp parallel for shared (output) private (indices, dists) num_threads(numofthreads)
	for (int idx = 0; idx < input.size(); ++idx)
	{
		Eigen::VectorXf n(5);
		n.setZero();

		Eigen::Vector4f xyz_centroid;
		Eigen::Matrix3f covariance_matrix;

		if (!std::isfinite<float>(input[idx](0)) ||
			!std::isfinite<float>(input[idx](1)) ||
			!std::isfinite<float>(input[idx](2)) ||
			SearchKDTree<Eigen::Vector3f>(&tree, input[idx], indices, dists, radius) == 0 )
		{
			n[0] = n[1] = n[2] = n[3] = n[4] = std::numeric_limits<float>::quiet_NaN();
			output[idx] = n;
			continue;
		}
		if (indices.size() < 3 || computeMeanAndCovarianceMatrix(input, indices, xyz_centroid, covariance_matrix) == 0)
		{
			n[0] = n[1] = n[2] = n[3] = n[4] = std::numeric_limits<float>::quiet_NaN();
			output[idx] = n;
			continue;
		}
		// Get the plane normal and surface curvature
		Eigen::Vector4f plane_parameters = Eigen::Vector4f::Zero(); 
		float curvature;
		solvePlaneParameters(covariance_matrix, xyz_centroid, plane_parameters, curvature);

		Eigen::Vector3f vp_ = vp - input[idx];
		Eigen::Vector3f normal(plane_parameters[0], plane_parameters[1], plane_parameters[2]);
		if (vp_.dot(normal) < 0)
		{
			plane_parameters[0] *= -1; plane_parameters[1] *= -1; plane_parameters[2] *= -1;
		}

		n[0] = plane_parameters[0]; n[1] = plane_parameters[1]; n[2] = plane_parameters[2]; n[3] = plane_parameters[3];
		n[4] = curvature;
		output[idx] = n;
	}

	std::cout << "computeNormals() , " << myTimeOff() << std::endl;
}
