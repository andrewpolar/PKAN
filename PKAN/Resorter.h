#pragma once
#include <memory>
#include <vector>
#include <algorithm>
#include <iostream>

class Resorter
{
public:
	void Resort(std::unique_ptr<std::unique_ptr<double[]>[]>& inputs,
		std::unique_ptr<double[]>& target, int nBlocks, int nRecords, int nFeatures, int nPoints);
private:
	std::vector<int> GetSortedIndexes(std::vector<double> x);
	void ResortArray(std::unique_ptr<double[]>& x, std::vector<int> indexes);
	void ResortMatrix(std::unique_ptr<std::unique_ptr<double[]>[]>& m, std::vector<int> indexes, int cols);
};

