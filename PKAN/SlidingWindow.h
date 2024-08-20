#pragma once
#include <memory>
#include "UrysohnPL.h"

class SlidingWindow
{
public:
	void BuildEnsemble(std::unique_ptr<std::unique_ptr<double[]>[]>& inputs,
		std::unique_ptr<double[]>& target, int nRecords, int nFeatures, int nWindow, int nShift, int nBlocks);

	std::unique_ptr<double[]> GetSample(std::unique_ptr<double[]>& input, int nFeatures);
	int GetSampleSize();

private:
	std::unique_ptr<std::unique_ptr<UrysohnPL>[]> _u = NULL;
	int _nU;
};

