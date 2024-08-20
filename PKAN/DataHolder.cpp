#include "DataHolder.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

std::unique_ptr<double[]> DataHolder::makeRandomInput() {
	auto input = std::make_unique<double[]>(nFeatures);
	for (int i = 0; i < nFeatures; ++i) {
		input[i] = static_cast<double>((rand() % 1000) / 1000.0);
	}
	return input;
}

void DataHolder::AddNoise(std::unique_ptr<double[]>& x, double delta) {
	for (int i = 0; i < nFeatures; ++i) {
		x[i] += delta * (std::rand() % 1000 - 500) / 1000.0;
	}
}

double DataHolder::computeTarget(std::unique_ptr<double[]>& input) {
	auto x = std::make_unique<double[]>(nFeatures);
	for (int j = 0; j < nFeatures; ++j) {
		x[j] = input[j];
	}
	AddNoise(x, 0.4);
	double pi = 3.14159265359;
	double y = (1.0 / pi);
	y *= (2.0 + 2.0 * x[2]);
	y *= (1.0 / 3.0);
	y *= atan(20.0 * (x[0] - 0.5 + x[1] / 6.0) * exp(x[4])) + pi / 2.0;

	double z = (1.0 / pi);
	z *= (2.0 + 2.0 * x[3]);
	z *= (1.0 / 3.0);
	z *= atan(20.0 * (x[0] - 0.5 - x[1] / 6.0) * exp(x[4])) + pi / 2.0;
	return static_cast<double>(y + z);
}

bool DataHolder::ReadDataMFormula() {
	target = std::make_unique<double[]>(nRecords);
	inputs = std::make_unique<std::unique_ptr<double[]>[]>(nRecords);
	int counter = 0;
	while (true) {
		inputs[counter] = makeRandomInput(); 
		target[counter] = computeTarget(inputs[counter]);
		if (++counter >= nRecords) break;
	}
	return true;
}

std::unique_ptr<double[]> DataHolder::GetEnsemble(std::unique_ptr<double[]>& input, int size) {
	auto ensemble = std::make_unique<double[]>(size);
	for (int i = 0; i < size; ++i) {
		ensemble[i] = computeTarget(input);
	}
	return ensemble;
}


