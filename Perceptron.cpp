#include "Perceptron.h"

void Perceptron::init(const vector<Link>& perc_connect, const ActivateFunction& act_type) {
	
	m_perc_connect = perc_connect;
	for (auto& x : m_perc_connect) {
		m_sum_input_val += exp(x.m_perceptron->get_output());
	}
	
	switch (act_type)
	{
	case ActivateFunction::step:
		m_act_func   = [=](const double& x) { return step(x); };
		m_act_d_func = [=](const double& x) { return d_step(x); };
		break;
	case ActivateFunction::sigmoid:
		m_act_func   = [=](const double& x) { return sigmoid(x); };
		m_act_d_func = [=](const double& x) { return d_sigmoid(x); };
		break;
	case ActivateFunction::tanh:
		m_act_func   = [=](const double& x) { return tanh(x); };
		m_act_d_func = [=](const double& x) { return d_tanh(x); };
		break;
	case ActivateFunction::relu:
		m_act_func   = [=](const double& x) { return relu(x); };
		m_act_d_func = [=](const double& x) { return d_relu(x); };
		break;
	case ActivateFunction::mish:
		m_act_func   = [=](const double& x) { return mish(x); };
		m_act_d_func = [=](const double& x) { return d_mish(x); };
		break;
	case ActivateFunction::identity:
		m_act_func   = [=](const double& x) { return identity(x); };
		m_act_d_func = [=](const double& x) { return d_identity(x); };
		break;
	case ActivateFunction::softmax:
		m_act_func   = [=](const double& x) { return softmax(x); };
		m_act_d_func = [=](const double& x) { return d_softmax(x); };
		break;
	default:
		m_act_func   = [=](const double& x) { return relu(x); };
		m_act_d_func = [=](const double& x) { return d_relu(x); };
		break;
	}
}

/* ‡“`”d */
void Perceptron::prop_forward() {
	m_a = 0.0;
	for (auto& link : m_perc_connect) {
		m_a += link.m_perceptron->get_output() * link.m_weight;
		//cout << m_a << endl;
		/*if (isnan(m_a)) {
			cout << link.m_perceptron->get_output() << endl;
			cout << link.m_weight << endl;
			exit(1);
		}*/

	}

	m_output = m_act_func(m_a);
}

/* ‹t“`”d */
void Perceptron::prop_back(const double& dE_da) {
	double temp_dE_da = dE_da;
	//if (temp_dE_da >= 10)temp_dE_da = 10;

	//cout << "*-----o—Í‘w-----*" << endl;
	for (auto& link : m_perc_connect) {
		//cout << link.m_weight << " -> ";
		link.m_weight -= eta * temp_dE_da * link.m_perceptron->get_output();
		//cout << link.m_weight << endl;
	}
	//cout << "*-----‰B‚ê‘w-----*" << endl;
}

double Perceptron::prop_back(const vector<double>& dE_da) {
	double sigma = 0.0;
	for (int i = 0; i < m_perc_connect.size(); ++i) {
		for (int j = 0; j < dE_da.size(); ++j) {
			sigma += dE_da[j] * m_perc_connect[i].m_weight;
		}
	}
	
	double next_dE_da = sigma * m_act_d_func(m_a);
	//if (next_dE_da >= 10)next_dE_da = 10;

	//d‚İw‚ÌXV
	for (auto& link : m_perc_connect) {
		//cout << link.m_weight << " -> ";
		link.m_weight -= eta * next_dE_da * link.m_perceptron->get_output();
		//cout << link.m_weight << endl;
	}

	return next_dE_da;
}
