#pragma once
#include "ActivateFunction.h"
#include "LossFunction.h"

class NNParameter {
public:
	int    inputlayer_dim  = 2;
	int    hiddenlayer_dim = 5;
	int    outputlayer_dim = 2;
	int    hiddenlayer_num = 1;
	int    epoch           = 50;
	int    batch_size      = 200;
	double eta             = 0.1;
	ActivateFunction act_hidden_type = ActivateFunction::relu;
	ActivateFunction act_output_type = ActivateFunction::identity;
	LossFunction     loss_type       = LossFunction::mean_squared_error;
public:
	NNParameter(){}
	~NNParameter(){}
};