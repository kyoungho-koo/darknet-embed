#include "darknet.h"

kvp net[] = {
	{"batch","1",0},
	{"subdivisions","1",0},
	{"width","224",0},
	{"height","224", 0},
	{"channels","3", 0},
	{"momentum","0.9",0},
	{"decay","0.0005",0},
	{"angle","0", 0},
	{"saturation", "1.5", 0},
	{"exposure", "1.5", 0},
	{"hue", ".1", 0},

	{"learning_rate", "0.001", 0},
	{"burn_in", "1000", 0},
	{"max_batches", "600200", 0},
	{"policy", "steps", 0},
	{"steps", "400000,450000", 0},
	{"scales", ".1,.1", 0},
};

kvp convolutional1[] = {
	{"batch_normalize", "0", 0},
	{"filters", "16", 0},
	{"size", "3", 0},
	{"stride", "1", 0},
	{"pad", "1", 0},
	{"activation", "leaky", 0},
};

kvp maxpool1[] = {
	{"size", "2", 0},
	{"stride", "2", 0},
};

kvp convolutional2[] = {
	{"batch_normalize", "0", 0},
	{"filters", "32", 0},
	{"size", "3", 0},
	{"stride", "1", 0},
	{"pad", "1", 0},
	{"activation", "leaky", 0},
};

kvp maxpool2[] = {
	{"size", "2", 0},
	{"stride", "2", 0},
};

kvp convolutional3[] = {
	{"batch_normalize", "0", 0},
	{"filters", "64", 0},
	{"size", "3", 0},
	{"stride", "1", 0},
	{"pad", "1", 0},
	{"activation", "leaky", 0},
};


kvp maxpool3[] = {
	{"size", "2", 0},
	{"stride", "2", 0},
};

kvp convolutional4[] = {
	{"batch_normalize", "0", 0},
	{"filters", "128", 0},
	{"size", "3", 0},
	{"stride", "1", 0},
	{"pad", "1", 0},
	{"activation", "leaky", 0},
};

kvp maxpool4[] = {
	{"size", "2", 0},
	{"stride", "2", 0},
};

kvp convolutional5[] = {
	{"batch_normalize", "0", 0},
	{"filters", "128", 0},
	{"size", "3", 0},
	{"stride", "1", 0},
	{"pad", "1", 0},
	{"activation", "leaky", 0},
};

kvp maxpool5[] = {
	{"size", "2", 0},
	{"stride", "2", 0},
};

kvp convolutional6[] = {
	{"batch_normalize", "0", 0},
	{"size", "3", 0},
	{"stride", "1", 0},
	{"pad", "1", 0},
	{"filters", "256", 0},
	{"activation", "leaky", 0},
};

kvp convolutional7[] = {
	{"batch_normalize", "0", 0},
	{"size", "1", 0},
	{"stride", "1", 0},
	{"pad", "1", 0},
	{"filters", "125", 0},
	{"activation", "linear", 0},
};

kvp yolo1[] = {
	{"mask", " 0,1,2,3,4", 0},
	{"anchors ", " 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52", 0},
	{"classes", "20", 0},
	{"num", "5", 0},
	{"jitter", ".3", 0},
	{"truth_thresh", "1", 0},
	{"thresh ", " .8", 0},
	{"random", "1", 0},
};


cfg_section yolo_lite_cfg[] = {
	{NETWORK, net, 17},
	{CONVOLUTIONAL, convolutional1, 6},
	{MAXPOOL, maxpool1, 2},
	{CONVOLUTIONAL, convolutional2, 6},
	{MAXPOOL, maxpool2, 2},
	{CONVOLUTIONAL, convolutional3, 6},
	{MAXPOOL, maxpool3, 2},
	{CONVOLUTIONAL, convolutional4, 6},
	{MAXPOOL, maxpool4, 2},
	{CONVOLUTIONAL, convolutional5, 6},
	{MAXPOOL, maxpool5, 2},
	{CONVOLUTIONAL, convolutional6, 6},
	{CONVOLUTIONAL, convolutional7, 6},
	{YOLO, yolo1, 8},
};

int yolo_lite_cfg_size = 14;
