#pragma once
#include <memory>

class Helper
{
public:
	void ShowMatrix(std::unique_ptr<std::unique_ptr<double[]>[]>& matrix, int rows, int cols);
	void ShowVector(std::unique_ptr<double[]>& ptr, int N);
	void SwapRows(std::unique_ptr<double[]>& row1, std::unique_ptr<double[]>& row2, int cols);
	void SwapScalars(double& x1, double& x2);
	void Shuffle(std::unique_ptr<std::unique_ptr<double[]>[]>& matrix, std::unique_ptr<double[]>& vector, int rows, int cols);
	void FindMinMax(std::unique_ptr<double[]>& xmin, std::unique_ptr<double[]>& xmax,
		double& targetMin, double& targetMax,
		std::unique_ptr<std::unique_ptr<double[]>[]>& matrix,
		std::unique_ptr<double[]>& target, int nRows, int nCols);
};

