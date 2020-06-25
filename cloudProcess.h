#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <flann/flann.hpp>
#include <math.h>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <string>
#include <time.h>
#include <Windows.h>
#include <limits>
#include <iostream>
#include <omp.h>

namespace mCP {

	typedef std::vector<Eigen::Vector3f> Points;
	typedef std::vector<Eigen::VectorXf> Features;
	typedef flann::Index<flann::L2<float> > KDTree;
	typedef std::vector<std::pair<int, int> > Correspondences;


	class cloudProcess
	{
		template <typename T>
		void BuildKDTree(const std::vector<T>& data, KDTree* tree);

		template <typename T>
		int SearchKDTree(KDTree* tree, const T& input, std::vector<int>& indices,
			std::vector<float>& dists, int nn);

		template <typename T>
		int SearchKDTree(KDTree* tree, const T& input, std::vector<int>& indices,
			std::vector<float>& dists, float radius);

	public:
		template <typename Matrix> inline
			typename Matrix::Scalar determinant3x3Matrix(const Matrix& matrix);

		template <typename Matrix> inline
			typename Matrix::Scalar invert3x3Matrix(const Matrix& matrix, Matrix& inverse);

		template <typename Matrix> inline
			typename Matrix::Scalar invert3x3SymMatrix(const Matrix& matrix, Matrix& inverse);

		template <typename Matrix> inline
			typename Matrix::Scalar invert2x2(const Matrix& matrix, Matrix& inverse);

		template <typename Scalar, typename Roots> inline 
			void computeRoots2(const Scalar& b, const Scalar& c, Roots& roots);

		template <typename Matrix, typename Roots> inline
			void computeRoots(const Matrix& m, Roots& roots);

		template <typename Matrix, typename Vector> inline void
			eigen33(const Matrix& mat, typename Matrix::Scalar& eigenvalue, Vector& eigenvector);

		template <typename Matrix, typename Vector> inline void
			eigen33(const Matrix& mat, Matrix& evecs, Vector& evals);


		float computeResolution(Points& input);
		void getMinMax3D(Points& input, Eigen::Vector3f& minP, Eigen::Vector3f& maxP );
		unsigned int computeMeanAndCovarianceMatrix(const Points& input, const std::vector<int> &indices, Eigen::Matrix<float, 4, 1> &centroid, Eigen::Matrix<float, 3, 3> &covariance_matrix);
		void solvePlaneParameters(const Eigen::Matrix3f &covariance_matrix, const Eigen::Vector4f &point, Eigen::Vector4f &plane_parameters, float &curvature);

		void statisticalFilter(Points& input, Points& output, int k, float stddevMulThresh, std::vector<int>& removeIndices, bool negative = false);

		void voxelFilter(Points& input, Points& output, Eigen::Vector3f& leafSize, int min_points_per_voxel = 0);

		void uniformSampling(Points& input, Points& output, Eigen::Vector3f& leafSize);

		void computeNormals(Points& input, Features& output, float radius, int numofthreads, Eigen::Vector3f vp = Eigen::Vector3f::Zero());
	};

	template<typename T>
	inline void cloudProcess::BuildKDTree(const std::vector<T>& data, KDTree * tree)
	{
		int rows, dim;
		rows = data.size();
		dim = static_cast<int>(data[0].size());
		std::vector<float> dataset(rows * dim);
		flann::Matrix<float> dataset_mat(&dataset[0], rows, dim);
		for(int i = 0; i < rows; ++i)
			for (int j = 0; j < dim; ++j) {
				dataset[i * dim + j] = data[i][j];
			}
		KDTree temp_tree(dataset_mat, flann::KDTreeSingleIndexParams(15));
		temp_tree.buildIndex();
		*tree = temp_tree;
	}

	template<typename T>
	inline int cloudProcess::SearchKDTree(KDTree * tree, const T & input, std::vector<int>& indices, std::vector<float>& dists, int nn)
	{
		int rows_t = 1;
		int dim = input.size();

		std::vector<float> query;
		query.resize(rows_t * dim);
		for (int i = 0; i < dim; ++i)
			query[i] = input[i];
		flann::Matrix<float> query_mat(&query[0], rows_t, dim);

		indices.resize(rows_t*nn);
		dists.resize(rows_t*nn);
		flann::Matrix<int> indices_mat(&indices[0], rows_t, nn);
		flann::Matrix<float> dists_mat(&dists[0], rows_t, nn);
		return tree->knnSearch(query_mat, indices_mat, dists_mat, nn, flann::SearchParams(128));
	}
	template<typename T>
	inline int cloudProcess::SearchKDTree(KDTree * tree, const T & input, std::vector<int>& indices, std::vector<float>& dists, float radius)
	{
		int rows_t = 1;
		int dim = input.size();
		int nn = tree->size();

		std::vector<float> query;
		query.resize(rows_t * dim);
		for (int i = 0; i < dim; ++i)
			query[i] = input[i];
		flann::Matrix<float> query_mat(&query[0], rows_t, dim);

		indices.resize(nn);
		dists.resize(nn);
		flann::Matrix<int> indices_mat(&indices[0], rows_t, nn);
		flann::Matrix<float> dists_mat(&dists[0], rows_t, nn);
		int num = tree->radiusSearch(query_mat, indices_mat, dists_mat, radius, flann::SearchParams(128));
		indices.resize(num);
		dists.resize(num);
		return num;
	}

	template<typename Matrix>
	inline typename Matrix::Scalar cloudProcess::determinant3x3Matrix(const Matrix & matrix)
	{
		return matrix.coeff(0) * (matrix.coeff(4) * matrix.coeff(8) - matrix.coeff(5) * matrix.coeff(7)) +
			matrix.coeff(1) * (matrix.coeff(5) * matrix.coeff(6) - matrix.coeff(3) * matrix.coeff(8)) +
			matrix.coeff(2) * (matrix.coeff(3) * matrix.coeff(7) - matrix.coeff(4) * matrix.coeff(6));
	}

	template <typename Matrix> 
	inline typename Matrix::Scalar cloudProcess::invert3x3Matrix(const Matrix& matrix, Matrix& inverse)
	{
		typedef typename Matrix::Scalar Scalar;

		//| a b c |-1             |   ie-hf    hc-ib   fb-ec  |
		//| d e f |    =  1/det * |   gf-id    ia-gc   dc-fa  |
		//| g h i |               |   hd-ge    gb-ha   ea-db  |
		//det = a(ie-hf) + d(hc-ib) + g(fb-ec)
		 
		Scalar ie_hf = matrix.coeff(8) * matrix.coeff(4) - matrix.coeff(7) * matrix.coeff(5);
		Scalar hc_ib = matrix.coeff(7) * matrix.coeff(2) - matrix.coeff(8) * matrix.coeff(1);
		Scalar fb_ec = matrix.coeff(5) * matrix.coeff(1) - matrix.coeff(4) * matrix.coeff(2);
		Scalar det = matrix.coeff(0) * (ie_hf)+matrix.coeff(3) * (hc_ib)+matrix.coeff(6) * (fb_ec);

		if (det != 0)
		{
			inverse.coeffRef(0) = ie_hf;
			inverse.coeffRef(1) = hc_ib;
			inverse.coeffRef(2) = fb_ec;
			inverse.coeffRef(3) = matrix.coeff(6) * matrix.coeff(5) - matrix.coeff(8) * matrix.coeff(3);
			inverse.coeffRef(4) = matrix.coeff(8) * matrix.coeff(0) - matrix.coeff(6) * matrix.coeff(2);
			inverse.coeffRef(5) = matrix.coeff(3) * matrix.coeff(2) - matrix.coeff(5) * matrix.coeff(0);
			inverse.coeffRef(6) = matrix.coeff(7) * matrix.coeff(3) - matrix.coeff(6) * matrix.coeff(4);
			inverse.coeffRef(7) = matrix.coeff(6) * matrix.coeff(1) - matrix.coeff(7) * matrix.coeff(0);
			inverse.coeffRef(8) = matrix.coeff(4) * matrix.coeff(0) - matrix.coeff(3) * matrix.coeff(1);

			inverse /= det;
		}
		return det;
	}
	template<typename Matrix>
	inline typename Matrix::Scalar cloudProcess::invert3x3SymMatrix(const Matrix & matrix, Matrix & inverse)
	{
		typedef typename Matrix::Scalar Scalar;

		//| a b c |-1             |   fd-ee    ce-bf   be-cd  |
		//| b d e |    =  1/det * |   ce-bf    af-cc   bc-ae  |
		//| c e f |               |   be-cd    bc-ae   ad-bb  |

		//det = a(fd-ee) + b(ec-fb) + c(eb-dc)

		Scalar fd_ee = matrix.coeff(4) * matrix.coeff(8) - matrix.coeff(7) * matrix.coeff(5);
		Scalar ce_bf = matrix.coeff(2) * matrix.coeff(5) - matrix.coeff(1) * matrix.coeff(8);
		Scalar be_cd = matrix.coeff(1) * matrix.coeff(5) - matrix.coeff(2) * matrix.coeff(4);

		Scalar det = matrix.coeff(0) * fd_ee + matrix.coeff(1) * ce_bf + matrix.coeff(2) * be_cd;

		if (det != 0)
		{
			//Scalar inv_det = Scalar (1.0) / det;
			inverse.coeffRef(0) = fd_ee;
			inverse.coeffRef(1) = inverse.coeffRef(3) = ce_bf;
			inverse.coeffRef(2) = inverse.coeffRef(6) = be_cd;
			inverse.coeffRef(4) = (matrix.coeff(0) * matrix.coeff(8) - matrix.coeff(2) * matrix.coeff(2));
			inverse.coeffRef(5) = inverse.coeffRef(7) = (matrix.coeff(1) * matrix.coeff(2) - matrix.coeff(0) * matrix.coeff(5));
			inverse.coeffRef(8) = (matrix.coeff(0) * matrix.coeff(4) - matrix.coeff(1) * matrix.coeff(1));
			inverse /= det;
		}
		return det;
	}
	template<typename Matrix>
	inline typename Matrix::Scalar cloudProcess::invert2x2(const Matrix & matrix, Matrix & inverse)
	{
		typedef typename Matrix::Scalar Scalar;
		Scalar det = matrix.coeff(0) * matrix.coeff(3) - matrix.coeff(1) * matrix.coeff(2);

		if (det != 0)
		{
			//Scalar inv_det = Scalar (1.0) / det;
			inverse.coeffRef(0) = matrix.coeff(3);
			inverse.coeffRef(1) = -matrix.coeff(1);
			inverse.coeffRef(2) = -matrix.coeff(2);
			inverse.coeffRef(3) = matrix.coeff(0);
			inverse /= det;
		}
		return det;
	}
	template<typename Scalar, typename Roots>
	inline void cloudProcess::computeRoots2(const Scalar & b, const Scalar & c, Roots & roots)
	{
		roots(0) = Scalar(0);
		Scalar d = Scalar(b * b - 4.0 * c);
		if (d < 0.0)  // no real roots ! THIS SHOULD NOT HAPPEN!
			d = 0.0;

		Scalar sd = ::std::sqrt(d);

		roots(2) = 0.5f * (b + sd);
		roots(1) = 0.5f * (b - sd);
	}
	template<typename Matrix, typename Roots>
	inline void cloudProcess::computeRoots(const Matrix & m, Roots & roots)
	{
		typedef typename Matrix::Scalar Scalar;

		// The characteristic equation is x^3 - c2*x^2 + c1*x - c0 = 0.  The
		// eigenvalues are the roots to this equation, all guaranteed to be
		// real-valued, because the matrix is symmetric.
		Scalar c0 = m(0, 0) * m(1, 1) * m(2, 2)
			+ Scalar(2) * m(0, 1) * m(0, 2) * m(1, 2)
			- m(0, 0) * m(1, 2) * m(1, 2)
			- m(1, 1) * m(0, 2) * m(0, 2)
			- m(2, 2) * m(0, 1) * m(0, 1);
		Scalar c1 = m(0, 0) * m(1, 1) -
			m(0, 1) * m(0, 1) +
			m(0, 0) * m(2, 2) -
			m(0, 2) * m(0, 2) +
			m(1, 1) * m(2, 2) -
			m(1, 2) * m(1, 2);
		Scalar c2 = m(0, 0) + m(1, 1) + m(2, 2);

		if (fabs(c0) < Eigen::NumTraits < Scalar > ::epsilon())  // one root is 0 -> quadratic equation
			computeRoots2(c2, c1, roots);
		else
		{
			const Scalar s_inv3 = Scalar(1.0 / 3.0);
			const Scalar s_sqrt3 = std::sqrt(Scalar(3.0));
			// Construct the parameters used in classifying the roots of the equation
			// and in solving the equation for the roots in closed form.
			Scalar c2_over_3 = c2 * s_inv3;
			Scalar a_over_3 = (c1 - c2 * c2_over_3) * s_inv3;
			if (a_over_3 > Scalar(0))
				a_over_3 = Scalar(0);

			Scalar half_b = Scalar(0.5) * (c0 + c2_over_3 * (Scalar(2) * c2_over_3 * c2_over_3 - c1));

			Scalar q = half_b * half_b + a_over_3 * a_over_3 * a_over_3;
			if (q > Scalar(0))
				q = Scalar(0);

			// Compute the eigenvalues by solving for the roots of the polynomial.
			Scalar rho = std::sqrt(-a_over_3);
			Scalar theta = std::atan2(std::sqrt(-q), half_b) * s_inv3;
			Scalar cos_theta = std::cos(theta);
			Scalar sin_theta = std::sin(theta);
			roots(0) = c2_over_3 + Scalar(2) * rho * cos_theta;
			roots(1) = c2_over_3 - rho * (cos_theta + s_sqrt3 * sin_theta);
			roots(2) = c2_over_3 - rho * (cos_theta - s_sqrt3 * sin_theta);

			// Sort in increasing order.
			if (roots(0) >= roots(1))
				std::swap(roots(0), roots(1));
			if (roots(1) >= roots(2))
			{
				std::swap(roots(1), roots(2));
				if (roots(0) >= roots(1))
					std::swap(roots(0), roots(1));
			}

			if (roots(0) <= 0)  // eigenval for symmetric positive semi-definite matrix can not be negative! Set it to 0
				computeRoots2(c2, c1, roots);
		}
	}
	template<typename Matrix, typename Vector>
	inline void cloudProcess::eigen33(const Matrix & mat, typename Matrix::Scalar & eigenvalue, Vector & eigenvector)
	{
		typedef typename Matrix::Scalar Scalar;
		// Scale the matrix so its entries are in [-1,1].  The scaling is applied
		// only when at least one matrix entry has magnitude larger than 1.

		Scalar scale = mat.cwiseAbs().maxCoeff();
		if (scale <= (std::numeric_limits < Scalar > ::min)())
			scale = Scalar(1.0);

		Matrix scaledMat = mat / scale;

		Vector eigenvalues;
		computeRoots(scaledMat, eigenvalues);

		eigenvalue = eigenvalues(0) * scale;

		scaledMat.diagonal().array() -= eigenvalues(0);

		Vector vec1 = scaledMat.row(0).cross(scaledMat.row(1));
		Vector vec2 = scaledMat.row(0).cross(scaledMat.row(2));
		Vector vec3 = scaledMat.row(1).cross(scaledMat.row(2));

		Scalar len1 = vec1.squaredNorm();
		Scalar len2 = vec2.squaredNorm();
		Scalar len3 = vec3.squaredNorm();

		if (len1 >= len2 && len1 >= len3)
			eigenvector = vec1 / std::sqrt(len1);
		else if (len2 >= len1 && len2 >= len3)
			eigenvector = vec2 / std::sqrt(len2);
		else
			eigenvector = vec3 / std::sqrt(len3);
	}

	template<typename Matrix, typename Vector>
	inline void cloudProcess::eigen33(const Matrix & mat, Matrix & evecs, Vector & evals)
	{
		typedef typename Matrix::Scalar Scalar;
		// Scale the matrix so its entries are in [-1,1].  The scaling is applied
		// only when at least one matrix entry has magnitude larger than 1.

		Scalar scale = mat.cwiseAbs().maxCoeff();
		if (scale <= (std::numeric_limits < Scalar > ::min)())
			scale = Scalar(1.0);

		Matrix scaledMat = mat / scale;

		// Compute the eigenvalues
		computeRoots(scaledMat, evals);

		if ((evals(2) - evals(0)) <= Eigen::NumTraits < Scalar > ::epsilon())
		{
			// all three equal
			evecs.setIdentity();
		}
		else if ((evals(1) - evals(0)) <= Eigen::NumTraits < Scalar > ::epsilon())
		{
			// first and second equal
			Matrix tmp;
			tmp = scaledMat;
			tmp.diagonal().array() -= evals(2);

			Vector vec1 = tmp.row(0).cross(tmp.row(1));
			Vector vec2 = tmp.row(0).cross(tmp.row(2));
			Vector vec3 = tmp.row(1).cross(tmp.row(2));

			Scalar len1 = vec1.squaredNorm();
			Scalar len2 = vec2.squaredNorm();
			Scalar len3 = vec3.squaredNorm();

			if (len1 >= len2 && len1 >= len3)
				evecs.col(2) = vec1 / std::sqrt(len1);
			else if (len2 >= len1 && len2 >= len3)
				evecs.col(2) = vec2 / std::sqrt(len2);
			else
				evecs.col(2) = vec3 / std::sqrt(len3);

			evecs.col(1) = evecs.col(2).unitOrthogonal();
			evecs.col(0) = evecs.col(1).cross(evecs.col(2));
		}
		else if ((evals(2) - evals(1)) <= Eigen::NumTraits < Scalar > ::epsilon())
		{
			// second and third equal
			Matrix tmp;
			tmp = scaledMat;
			tmp.diagonal().array() -= evals(0);

			Vector vec1 = tmp.row(0).cross(tmp.row(1));
			Vector vec2 = tmp.row(0).cross(tmp.row(2));
			Vector vec3 = tmp.row(1).cross(tmp.row(2));

			Scalar len1 = vec1.squaredNorm();
			Scalar len2 = vec2.squaredNorm();
			Scalar len3 = vec3.squaredNorm();

			if (len1 >= len2 && len1 >= len3)
				evecs.col(0) = vec1 / std::sqrt(len1);
			else if (len2 >= len1 && len2 >= len3)
				evecs.col(0) = vec2 / std::sqrt(len2);
			else
				evecs.col(0) = vec3 / std::sqrt(len3);

			evecs.col(1) = evecs.col(0).unitOrthogonal();
			evecs.col(2) = evecs.col(0).cross(evecs.col(1));
		}
		else
		{
			Matrix tmp;
			tmp = scaledMat;
			tmp.diagonal().array() -= evals(2);

			Vector vec1 = tmp.row(0).cross(tmp.row(1));
			Vector vec2 = tmp.row(0).cross(tmp.row(2));
			Vector vec3 = tmp.row(1).cross(tmp.row(2));

			Scalar len1 = vec1.squaredNorm();
			Scalar len2 = vec2.squaredNorm();
			Scalar len3 = vec3.squaredNorm();
#ifdef _WIN32
			Scalar *mmax = new Scalar[3];
#else
			Scalar mmax[3];
#endif
			unsigned int min_el = 2;
			unsigned int max_el = 2;
			if (len1 >= len2 && len1 >= len3)
			{
				mmax[2] = len1;
				evecs.col(2) = vec1 / std::sqrt(len1);
			}
			else if (len2 >= len1 && len2 >= len3)
			{
				mmax[2] = len2;
				evecs.col(2) = vec2 / std::sqrt(len2);
			}
			else
			{
				mmax[2] = len3;
				evecs.col(2) = vec3 / std::sqrt(len3);
			}

			tmp = scaledMat;
			tmp.diagonal().array() -= evals(1);

			vec1 = tmp.row(0).cross(tmp.row(1));
			vec2 = tmp.row(0).cross(tmp.row(2));
			vec3 = tmp.row(1).cross(tmp.row(2));

			len1 = vec1.squaredNorm();
			len2 = vec2.squaredNorm();
			len3 = vec3.squaredNorm();
			if (len1 >= len2 && len1 >= len3)
			{
				mmax[1] = len1;
				evecs.col(1) = vec1 / std::sqrt(len1);
				min_el = len1 <= mmax[min_el] ? 1 : min_el;
				max_el = len1 > mmax[max_el] ? 1 : max_el;
			}
			else if (len2 >= len1 && len2 >= len3)
			{
				mmax[1] = len2;
				evecs.col(1) = vec2 / std::sqrt(len2);
				min_el = len2 <= mmax[min_el] ? 1 : min_el;
				max_el = len2 > mmax[max_el] ? 1 : max_el;
			}
			else
			{
				mmax[1] = len3;
				evecs.col(1) = vec3 / std::sqrt(len3);
				min_el = len3 <= mmax[min_el] ? 1 : min_el;
				max_el = len3 > mmax[max_el] ? 1 : max_el;
			}

			tmp = scaledMat;
			tmp.diagonal().array() -= evals(0);

			vec1 = tmp.row(0).cross(tmp.row(1));
			vec2 = tmp.row(0).cross(tmp.row(2));
			vec3 = tmp.row(1).cross(tmp.row(2));

			len1 = vec1.squaredNorm();
			len2 = vec2.squaredNorm();
			len3 = vec3.squaredNorm();
			if (len1 >= len2 && len1 >= len3)
			{
				mmax[0] = len1;
				evecs.col(0) = vec1 / std::sqrt(len1);
				min_el = len3 <= mmax[min_el] ? 0 : min_el;
				max_el = len3 > mmax[max_el] ? 0 : max_el;
			}
			else if (len2 >= len1 && len2 >= len3)
			{
				mmax[0] = len2;
				evecs.col(0) = vec2 / std::sqrt(len2);
				min_el = len3 <= mmax[min_el] ? 0 : min_el;
				max_el = len3 > mmax[max_el] ? 0 : max_el;
			}
			else
			{
				mmax[0] = len3;
				evecs.col(0) = vec3 / std::sqrt(len3);
				min_el = len3 <= mmax[min_el] ? 0 : min_el;
				max_el = len3 > mmax[max_el] ? 0 : max_el;
			}

			unsigned mid_el = 3 - min_el - max_el;
			evecs.col(min_el) = evecs.col((min_el + 1) % 3).cross(evecs.col((min_el + 2) % 3)).normalized();
			evecs.col(mid_el) = evecs.col((mid_el + 1) % 3).cross(evecs.col((mid_el + 2) % 3)).normalized();
#ifdef _WIN32
			delete[] mmax;
#endif
		}
		// Rescale back to the original size.
		evals *= scale;
	}
}
