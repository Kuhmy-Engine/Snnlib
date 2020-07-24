#include "KumaNN.h"
#include <iostream>
#include <cmath>
#include <memory>
#include <random>

using namespace std;

int main() {

	vector<vector<double>> train_x_data;
	vector<vector<double>> train_t_data;
	const double PI = 3.14159265;
	std::random_device rnd;
	std::mt19937 mt(rnd());
	std::uniform_real_distribution<> rand_theta(0.0, 1.0);
	//std::uniform_int_distribution<> rand_theta(0, 1);
	const int data_size = 1024;
	for (int i = 0; i < data_size; ++i) {
		vector<double> temp_x;
		vector<double> temp_t;
		double theta1 = rand_theta(mt);
		//double theta2 = rand_theta(mt);
		temp_x.push_back(theta1);
		//temp_x.push_back(theta2);
		temp_t.push_back(sin(2*theta1 * PI));
		train_x_data.push_back(temp_x);
		train_t_data.push_back(temp_t);
	}

	unique_ptr<MultilayerPerceptron> nn(new MultilayerPerceptron(
		1, 2, 8, 1, 1000, 64, 0.1,
		ActivateFunction::tanh,
		ActivateFunction::identity,
		LossFunction::mean_squared_error));

	nn->train(train_x_data, train_t_data);
	
	auto test_result = nn->predict({ 0 });
	for (auto& result : test_result) {
		cout << result << endl;
	}
	test_result = nn->predict({ 0.25 });
	for (auto& result : test_result) {
		cout << result << endl;
	}
	test_result = nn->predict({ 0.5 });
	for (auto& result : test_result) {
		cout << result << endl;
	}
	test_result = nn->predict({ 0.75 });
	for (auto& result : test_result) {
		cout << result << endl;
	}
	test_result = nn->predict({ 1.0 });
	for (auto& result : test_result) {
		cout << result << endl;
	}
	

	return 0;
}