#include "Resorter.h"
#include "Helper.h"
#include "UrysohnPL.h"

std::vector<int> Resorter::GetSortedIndexes(std::vector<double> x) {
	std::vector<std::pair<double, int>> indexedArray;
	for (int i = 0; i < x.size(); ++i) {
		indexedArray.emplace_back(x[i], i);
	}
	std::sort(indexedArray.begin(), indexedArray.end());
	std::vector<int> sortedIndices;
	for (const auto& pair : indexedArray) {
		sortedIndices.push_back(pair.second);
	}
	return sortedIndices;
}

void Resorter::ResortArray(std::unique_ptr<double[]>& x, std::vector<int> indexes) {
	size_t N = indexes.size();
	auto tmp = std::make_unique<double[]>(N);
	for (int i = 0; i < N; ++i) {
		tmp[i] = x[indexes[i]];
	}
	for (int i = 0; i < N; ++i) {
		x[i] = tmp[i];
	}
}

void Resorter::ResortMatrix(std::unique_ptr<std::unique_ptr<double[]>[]>& m, std::vector<int> indexes, int cols) {
	size_t N = indexes.size();
	auto tmp = std::make_unique<std::unique_ptr<double[]>[]>(N);
	for (int i = 0; i < N; ++i) {
		tmp[i] = std::make_unique<double[]>(cols);
		for (int j = 0; j < cols; ++j) {
			tmp[i][j] = m[indexes[i]][j];
		}
	}
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < cols; ++j) {
			m[i][j] = tmp[i][j];
		}
	}
}

void Resorter::Resort(std::unique_ptr<std::unique_ptr<double[]>[]>& inputs,
	std::unique_ptr<double[]>& target, int nBlocks, int nRecords, int nFeatures, int nPoints) {

	//prepare block structure
	auto firstIndex = std::make_unique<int[]>(nBlocks);
	auto lastIndex = std::make_unique<int[]>(nBlocks);
	int blockSize = nRecords / nBlocks;
	for (int i = 0; i < nBlocks; ++i) {
		firstIndex[i] = blockSize * i;
	}
	for (int i = 0; i < nBlocks - 1; ++i) {
		lastIndex[i] = firstIndex[i + 1] - 1;
	}
	lastIndex[nBlocks - 1] = nRecords - 1;

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

	auto u = std::make_unique<std::unique_ptr<UrysohnPL>[]>(nBlocks);
	for (int i = 0; i < nBlocks; ++i) {
		u[i] = std::make_unique<UrysohnPL>(ymin, ymax, targetMin, targetMax, interior_structure, nFeatures);
	}

	//obtaining residuals
	double mu = 0.01;
	int nEpochs = 20;
	auto residuals = std::make_unique<double[]>(nRecords);
	for (int block = 0; block < nBlocks; ++block) {
		for (int epoch = 0; epoch < nEpochs; ++epoch) {
			double error = 0.0;
			int cnt = 0;
			for (int record = firstIndex[block]; record <= lastIndex[block]; ++record) {
				double residual = target[record];
				residual -= u[block]->GetValueUsingInput(inputs[record]);
				u[block]->UpdateUsingMemory(residual, mu);
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
				printf("Relative residual error in resorted block %6.4f\n", error);
			}
 		}
	}
	printf("\n");

	//resorting records
	std::vector<int> allIndexes;
	for (int block = 0; block < nBlocks; ++block) {
		std::vector<double> residualV;
		for (int record = firstIndex[block]; record <= lastIndex[block]; ++record) {
			residualV.push_back(residuals[record]);
		}
		auto indexes = GetSortedIndexes(residualV);
		for (int k = 0; k < indexes.size(); ++k) {
			allIndexes.push_back(indexes[k] + firstIndex[block]);
		}
	}
	ResortArray(target, allIndexes);
	ResortMatrix(inputs, allIndexes, nFeatures);
}
