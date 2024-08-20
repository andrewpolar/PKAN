#include "SlidingWindow.h"
#include "Helper.h"

void SlidingWindow::BuildEnsemble(std::unique_ptr<std::unique_ptr<double[]>[]>& inputs,
	std::unique_ptr<double[]>& target, int nRecords, int nFeatures, int nWindow, int nShift, int nPoints) {

	auto helper = std::unique_ptr<Helper>();
	auto ymin = std::make_unique<double[]>(nFeatures);
	auto ymax = std::make_unique<double[]>(nFeatures);
	double targetMin, targetMax;
	helper->FindMinMax(ymin, ymax, targetMin, targetMax, inputs, target, nRecords, nFeatures);
	std::unique_ptr<int[]> interior_structure = std::make_unique<int[]>(nFeatures);
	for (int i = 0; i < nFeatures; ++i)
	{
		interior_structure[i] = static_cast<int>(nPoints);
	}

	_nU = (nRecords - nWindow) / nShift;
	if (0 != (nRecords - nWindow) % nShift) _nU += 1;
	auto firstIndex = std::make_unique<int[]>(_nU);
	auto lastIndex = std::make_unique<int[]>(_nU);
	for (int i = 0; i < _nU; ++i) {
		firstIndex[i] = nShift * i;
	}
	for (int i = 0; i < _nU - 1; ++i) {
		lastIndex[i] = nWindow + nShift * i - 1;
	}
	lastIndex[_nU - 1] = nRecords - 1;

	_u = std::make_unique<std::unique_ptr<UrysohnPL>[]>(_nU);
	for (int i = 0; i < _nU; ++i) {
		_u[i] = std::make_unique<UrysohnPL>(ymin, ymax, targetMin, targetMax, interior_structure, nFeatures);
	}

	//obtaining residuals
	double mu = 0.01;
	int nEpochs = 20;
	auto residuals = std::make_unique<double[]>(nRecords);
	for (int block = 0; block < _nU; ++block) {
		for (int epoch = 0; epoch < nEpochs; ++epoch) {
			double error = 0.0;
			int cnt = 0;
			for (int record = firstIndex[block]; record <= lastIndex[block]; ++record) {
				double residual = target[record];
				residual -= _u[block]->GetValueUsingInput(inputs[record]);
				_u[block]->UpdateUsingMemory(residual, mu);
				if (nEpochs - 1 == epoch) {
					residuals[record] = residual;
				}
				error += residual * residual;
				++cnt;
			}
			if (nEpochs - 1 == epoch) {
				error /= cnt;
				error = sqrt(error);
				error /= (targetMax - targetMin);
				printf("Relative residual error in sliding window block %6.4f\n", error);
			}
		}
	}
	printf("\n");
}

int SlidingWindow::GetSampleSize() {
	return _nU;
}

std::unique_ptr<double[]> SlidingWindow::GetSample(std::unique_ptr<double[]>& input, int nFeatures) {
	auto sample = std::make_unique<double[]>(_nU);
	for (int i = 0; i < _nU; ++i) {
		sample[i] = _u[i]->GetValueUsingInput(input);
	}
	return sample;
}
