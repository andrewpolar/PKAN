#pragma once
#include <iostream>

class DataHolder {
public:
	bool ReadDataMFormula();
	std::unique_ptr<double[]> makeRandomInput();
	double computeTarget(std::unique_ptr<double[]>& input);
	void AddNoise(std::unique_ptr<double[]>& x, double delta);
	std::unique_ptr<double[]> GetEnsemble(std::unique_ptr<double[]>& input, int size);
	std::unique_ptr<std::unique_ptr<double[]>[]> inputs;
	std::unique_ptr<double[]> target;
	const int nRecords = 10000;
	const int nFeatures = 5;
};