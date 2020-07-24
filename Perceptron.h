#pragma once
#include <iostream>
#include <memory>
#include <vector>
#include <functional>
#include <cmath>

#include "ActivateFunction.h"

using namespace std;

class Perceptron {
public:
	struct Link {
		shared_ptr<Perceptron> m_perceptron;
		double                 m_weight;
	};

private:
	double eta;

private:
	function<double(const double&)> m_act_func;
	function<double(const double&)> m_act_d_func;
	vector<Link>                    m_perc_connect;
	double                          m_output;
	double                          m_a;
	double                          m_sum_input_val;

public:
	Perceptron(const double& eta = 0.1) { this->eta = eta; }
	virtual ~Perceptron() {}

public:
	/* èâä˙âª */
	void init(const vector<Link>& perc_connect, const ActivateFunction& act_type);
	/* èáì`îd */
	void prop_forward();
	void prop_forward(const double& output) { m_output = output; }
	/* ãtì`îd */
	void prop_back(const double& dE_da);
	double prop_back(const vector<double>& dE_da);
	double get_output()const { return m_output; }
	double get_dy_da()const { return m_act_d_func(m_a); }

private:
	double step(const double& x) {
		return (x >= 0.0);
	}
	double d_step(const double& x) {
		return 0.0;
	}
	double sigmoid(const double& x) {
		return 1.0 / (1.0 + exp(-x));
	}
	double d_sigmoid(const double& x) {
		return sigmoid(x) * (1.0 - sigmoid(x));
	}
	double tanh(const double& x) {
		return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
	}
	double d_tanh(const double& x) {
		return 4.0 / pow(exp(x) + exp(-x), 2);
	}
	double relu(const double& x) {
		return (x > 0.0) ? x : 0.0;
	}
	double d_relu(const double& x) {
		return (x > 0.0);
	}
	double mish(const double& x) {
		return x * tanh(log(1 + exp(x) + 1e-07));
	}
	double d_mish(const double& x) {
		double w = 4 * (x + 1) + 4 * exp(2 * x) + exp(3 * x) + (4 * x + 6) * exp(x);
		double d = 2 * exp(x) + exp(2 * x) + 2;
		return (exp(x) * w) / (d * d);
	}

	double identity(const double& x) {
		return x;
	}
	double d_identity(const double& x) {
		return 1;
	}
	double softmax(const double& x) {
		return exp(x) / m_sum_input_val;
	}
	double d_softmax(const double& x) {
		return softmax(x) * (1.0 - softmax(x));
	}
};