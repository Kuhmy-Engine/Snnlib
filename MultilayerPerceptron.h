#pragma once
#include <memory>
#include <vector>
#include <functional>

#include "Perceptron.h"
#include "ActivateFunction.h"
#include "LossFunction.h"
#include "NNParameter.h"

using namespace std;

class MultilayerPerceptron {
private:
	unique_ptr<NNParameter> m_nnparameter;

private:
	vector<shared_ptr<Perceptron>>         m_input_layer;
	vector<vector<shared_ptr<Perceptron>>> m_hidden_layers;
	vector<shared_ptr<Perceptron>>         m_output_layer;
	function<const double(const vector<double>&)>         m_loss_func;
	function<const vector<double>(const vector<double>&)> m_loss_d_func;

public:
	MultilayerPerceptron(const int& inputlayer_dim,
						 const int& hiddenlayer_num,
						 const int& hiddenlayer_dim,
						 const int& outputlayer_dim,
						 const int& epoch,
						 const int& batch_size,
						 const double& eta,
						 const ActivateFunction& act_hidden_type,
						 const ActivateFunction& act_output_type,
						 const LossFunction&     loss_type) : m_nnparameter(new NNParameter()){
		m_nnparameter->inputlayer_dim  = inputlayer_dim;
		m_nnparameter->hiddenlayer_num = hiddenlayer_num;
		m_nnparameter->hiddenlayer_dim = hiddenlayer_dim;
		m_nnparameter->outputlayer_dim = outputlayer_dim;
		m_nnparameter->epoch           = epoch;
		m_nnparameter->batch_size      = batch_size;
		m_nnparameter->eta			   = eta;
		m_nnparameter->act_hidden_type = act_hidden_type;
		m_nnparameter->act_output_type = act_output_type;
		m_nnparameter->loss_type       = loss_type;
		init();
	}
	virtual ~MultilayerPerceptron() {}

public:
	void           train(vector<vector<double>>& train_x_data, vector<vector<double>>& train_t_data);
	vector<double> predict(const vector<double>& x_data);

private:
	void   init();
	void   prop_forward(const vector<double>& train_x_data);
	void   prop_back(const vector<double>& train_t_data);
	double get_loss_average(const vector<vector<double>>& train_x_data, const vector<vector<double>>& train_t_data);
	vector<shared_ptr<Perceptron>> create_layer(const int& layer_dim);
	vector<Perceptron::Link> create_link(const vector<shared_ptr<Perceptron>>& layer);

private:
	double mean_squared_error(const vector<double>& train_t_data) {
		double E = 0.0;
		double D = m_output_layer.size();

		for (int d = 0; d < D; ++d) {
			double y = m_output_layer[d]->get_output();
			double t = train_t_data[d];
			E += (y - t) * (y - t);
		}
		E /= D;

		return E;
	}
	double cross_entropy(const vector<double>& train_t_data) {
		double E = 0.0;
		double D = m_output_layer.size();
		for (int d = 0; d < D; ++d) {
			double y = m_output_layer[d]->get_output();
			double t = train_t_data[d];
			E += t * log(y + 1e07);
		}

		return -E;
	}
	vector<double> d_mean_squared_error(const vector<double>& train_t_data) {
		vector<double> temp_E;
		double D = m_output_layer.size();
		for (int d = 0; d < D; ++d) {
			double y = m_output_layer[d]->get_output();
			double t = train_t_data[d];
			temp_E.push_back(2 * (y - t) / D);
		}

		return temp_E;
	}
	vector<double> d_cross_entropy(const vector<double>& train_t_data) {
		vector<double> temp_E;
		double D = m_output_layer.size();
		for (int d = 0; d < D; ++d) {
			double y = m_output_layer[d]->get_output();
			double t = train_t_data[d];
			temp_E.push_back(-(t / y));
		}

		return temp_E;
	}
};