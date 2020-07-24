#include "MultilayerPerceptron.h"
#include <random>
#include <algorithm>
#include <iostream>

using namespace std;

void MultilayerPerceptron::train(vector<vector<double>>& train_x_data, vector<vector<double>>& train_t_data) {
	cout << " train start" << endl;

	vector<int> train_index;
	for (int i = 0; i < train_x_data.size(); ++i) {
		train_index.push_back(i);
	}

	for (int i = 0; i < m_nnparameter->epoch; ++i) {
		int pivot = 0;
		for (int j = 0; j < train_x_data.size() / m_nnparameter->batch_size + 1; ++j) {
			for (int k = 0; k < m_nnparameter->batch_size; ++k) {
				if ((k + pivot) >= train_x_data.size())break;
				prop_forward(train_x_data[train_index[k + pivot]]);
				prop_back(train_t_data[train_index[k + pivot]]);
			}
			pivot += train_x_data.size() / m_nnparameter->batch_size;
		}
		std::random_device rnd;
		std::mt19937 mt(rnd());
		shuffle(train_index.begin(), train_index.end(), mt);

		cout << " /--------------------------------/" << endl;
		cout << " epoch : " << i + 1 << "/" << m_nnparameter->epoch << endl;
		cout << " -- score : " << get_loss_average(train_x_data, train_t_data) << endl;
	}
	cout << " /--------------------------------/" << endl;
	cout << " train finished" << endl;
}

vector<double> MultilayerPerceptron::predict(const vector<double>& x_data) {
	prop_forward(x_data);
	vector<double> result;
	for (auto& output_perc : m_output_layer) {
		result.push_back(output_perc->get_output());
	}

	return result;
}

void MultilayerPerceptron::init() {
	//ëπé∏ä÷êîê›íË
	switch (m_nnparameter->loss_type)
	{
	case LossFunction::mean_squared_error:
		m_loss_func   = [=](const vector<double>& train_t) { return mean_squared_error(train_t); };
		m_loss_d_func = [=](const vector<double>& train_t) { return d_mean_squared_error(train_t); };
		break;
	case LossFunction::cross_entropy:
		m_loss_func   = [=](const vector<double>& train_t) { return cross_entropy(train_t); };
		m_loss_d_func = [=](const vector<double>& train_t) { return d_cross_entropy(train_t); };
		break;
	}

	//äeëwê∂ê¨
	m_input_layer = create_layer(m_nnparameter->inputlayer_dim);
	
	for (int l = 0; l < m_nnparameter->hiddenlayer_num; ++l) {
		m_hidden_layers.push_back(create_layer(m_nnparameter->hiddenlayer_dim));
	}

	m_output_layer = create_layer(m_nnparameter->outputlayer_dim - 1);

	//äeëwäeÉmÅ[ÉhÇÃÉäÉìÉNå`ê¨
	for (int l = 0; l < m_hidden_layers.size(); ++l) {
		for (int i = 0; i < m_hidden_layers[l].size(); ++i) {
			if (l == 0) 
				m_hidden_layers[l][i]->init(create_link(m_input_layer), m_nnparameter->act_hidden_type);
			else 
				m_hidden_layers[l][i]->init(create_link(m_hidden_layers[l - 1]), m_nnparameter->act_hidden_type);
		}
	}
	for (int i = 0; i < m_output_layer.size(); ++i) {
		m_output_layer[i]->init(create_link(m_hidden_layers[m_hidden_layers.size() - 1]), m_nnparameter->act_output_type);
	}

}

void MultilayerPerceptron::prop_forward(const vector<double>& train_x_data) {
	//ì¸óÕëwÇ…åPó˚ÉfÅ[É^ì¸óÕ
	m_input_layer[0]->prop_forward(1.0); //ÉoÉCÉAÉX
	for (int i = 1; i < m_input_layer.size(); ++i) {
		m_input_layer[i]->prop_forward(train_x_data[i - 1]);
	}

	//èáì`îd
	for (int l = 0; l < m_hidden_layers.size(); ++l) {
		m_hidden_layers[l][0]->prop_forward(1.0);
		for (int i = 1; i < m_hidden_layers[l].size(); ++i) {
			m_hidden_layers[l][i]->prop_forward();
		}
	}
	for (int i = 0; i < m_output_layer.size(); ++i) {
		m_output_layer[i]->prop_forward();
	}
}

void MultilayerPerceptron::prop_back(const vector<double>& train_t_data) {
	//èoóÕëwÇ≈dE_daÇéZèo  sigma(d_loss * d_act)
	vector<double> dE_da;
	auto& dE_dy = m_loss_d_func(train_t_data);

	for (int i = 0; i < m_output_layer.size(); ++i) {
		double temp = 0.0;
		for (int k = 0; k < m_output_layer.size(); ++k) {
			temp += dE_dy[k] * m_output_layer[i]->get_dy_da();
		}
		dE_da.push_back(temp);
	}
	
	for (int i = 0; i < m_output_layer.size(); ++i) {
		m_output_layer[i]->prop_back(dE_da[i]);
	}

	for (int l = m_hidden_layers.size() - 1; l >= 0; --l) {
		vector<double> temp;
		for (int i = 0; i < m_hidden_layers[l].size(); ++i) {
			double next_dE_da = m_hidden_layers[l][i]->prop_back(dE_da);
			temp.push_back(next_dE_da);
		}
		vector<double>().swap(dE_da);
		dE_da = temp;
	}
}

double MultilayerPerceptron::get_loss_average(const vector<vector<double>>& train_x_data, const vector<vector<double>>& train_t_data) {
	double score = 0.0;
	for (int i = 0; i < train_t_data.size(); ++i) {
		prop_forward(train_x_data[i]);
		score += m_loss_func(train_t_data[i]);
	}
	score /= train_t_data.size();
	return score;
}

vector<shared_ptr<Perceptron>> MultilayerPerceptron::create_layer(const int& layer_dim) {
	vector<shared_ptr<Perceptron>> temp_layer;
	for (int i = 0; i < layer_dim + 1; ++i) {
		shared_ptr<Perceptron> temp_perceptron(new Perceptron(m_nnparameter->eta));
		temp_layer.push_back(move(temp_perceptron));
	}
	return temp_layer;
}

vector<Perceptron::Link> MultilayerPerceptron::create_link(const vector<shared_ptr<Perceptron>>& layer) {
	random_device rnd; 
	mt19937 mt(rnd());
	double aver = 0.0;
	double dist = 1.0;
	if (m_nnparameter->act_hidden_type == ActivateFunction::sigmoid)//Xivier
		dist = 1.0 / (layer.size());
	if (m_nnparameter->act_hidden_type == ActivateFunction::tanh)//Xivier
		dist = 1.0 / (layer.size());
	if (m_nnparameter->act_hidden_type == ActivateFunction::relu)//He
		dist = 2.0 / (layer.size());
	if (m_nnparameter->act_hidden_type == ActivateFunction::mish)//He
		dist = 2.0 / (layer.size());
	normal_distribution<> norm(aver, dist);

	vector<Perceptron::Link> temp_link_list;
	if (!layer.empty()) {
		for (int i = 0; i < layer.size(); ++i) {
			Perceptron::Link temp_link;
			temp_link.m_perceptron = layer[i];
			temp_link.m_weight     = norm(mt);
			//cout << "w = " << temp_link.m_weight << endl;
			temp_link_list.push_back(temp_link);
		}
	}
	
	return temp_link_list;
}