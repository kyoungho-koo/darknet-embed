#include "darknet.h"


/*
[net]
batch=1
subdivisions=1
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.001
burn_in=1000
max_batches = 500200
policy=steps
steps=400000,450000
scales=.1,.1

*/
static kvp net[] = {
	{"batch","1",0},
	{"subdivisions","1",0},
	{"width","416",0},
	{"height","416", 0},
	{"channels","3", 0},
	{"momentum","0.9",0},
	{"decay","0.0005",0},
	{"angle","0", 0},
	{"saturation", "1.5", 0},
	{"exposure", "1.5", 0},
	{"hue", ".1", 0},

	{"learning_rate", "0.001", 0},
	{"burn_in", "1000", 0},
	{"max_batches", "500200", 0},
	{"policy", "steps", 0},
	{"steps", "400000,450000", 0},
	{"scales", ".1,.1", 0},
};


/*
[convolutional]
batch_normalize=1
filters=16
size=3
stride=1
pad=1
activation=leaky
*/

static kvp convolutional1[] = {
	{"batch_normalize", "1", 0},
	{"filters", "16", 0},
	{"size", "3", 0},
	{"stride", "1", 0},
	{"pad", "1", 0},
	{"activation", "leaky", 0},
};

/*
[maxpool]
size=2
stride=2
*/

static kvp maxpool1[] = {
	{"size", "2", 0},
	{"stride", "2", 0},
};

/*
[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky
*/

static kvp convolutional2[] = {
	{"batch_normalize", "1", 0},
	{"filters", "32", 0},
	{"size", "3", 0},
	{"stride", "1", 0},
	{"pad", "1", 0},
	{"activation", "leaky", 0},
};

/*
[maxpool]
size=2
stride=2
*/

static kvp maxpool2[] = {
	{"size", "2", 0},
	{"stride", "2", 0},
};

/*
[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky
*/

static kvp convolutional3[] = {
	{"batch_normalize", "1", 0},
	{"filters", "64", 0},
	{"size", "3", 0},
	{"stride", "1", 0},
	{"pad", "1", 0},
	{"activation", "leaky", 0},
};


/*
[maxpool]
size=2
stride=2
*/

static kvp maxpool3[] = {
	{"size", "2", 0},
	{"stride", "2", 0},
};

/*
[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky
*/

static kvp convolutional4[] = {
	{"batch_normalize", "1", 0},
	{"filters", "128", 0},
	{"size", "3", 0},
	{"stride", "1", 0},
	{"pad", "1", 0},
	{"activation", "leaky", 0},
};

/*
[maxpool]
size=2
stride=2
*/

static kvp maxpool4[] = {
	{"size", "2", 0},
	{"stride", "2", 0},
};

/*
[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky
*/

static kvp convolutional5[] = {
	{"batch_normalize", "1", 0},
	{"filters", "256", 0},
	{"size", "3", 0},
	{"stride", "1", 0},
	{"pad", "1", 0},
	{"activation", "leaky", 0},
};

/*
[maxpool]
size=2
stride=2
*/

static kvp maxpool5[] = {
	{"size", "2", 0},
	{"stride", "2", 0},
};

/*
[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky
*/

static kvp convolutional6[] = {
	{"batch_normalize", "1", 0},
	{"filters", "512", 0},
	{"size", "3", 0},
	{"stride", "1", 0},
	{"pad", "1", 0},
	{"activation", "leaky", 0},
};

/*
[maxpool]
size=2
stride=1
*/

static kvp maxpool6[] = {
	{"size", "2", 0},
	{"stride", "1", 0},
};

/*
[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky
*/

static kvp convolutional7[] = {
	{"batch_normalize", "1", 0},
	{"filters", "1024", 0},
	{"size", "3", 0},
	{"stride", "1", 0},
	{"pad", "1", 0},
	{"activation", "leaky", 0},
};


/*
[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky
*/


static kvp convolutional8[] = {
	{"batch_normalize", "1", 0},
	{"filters", "256", 0},
	{"size", "1", 0},
	{"stride", "1", 0},
	{"pad", "1", 0},
	{"activation", "leaky", 0},
};

/*
[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky
*/

static kvp convolutional9[] = {
	{"batch_normalize", "1", 0},
	{"filters", "512", 0},
	{"size", "3", 0},
	{"stride", "1", 0},
	{"pad", "1", 0},
	{"activation", "leaky", 0},
};

/*
[convolutional]
size=1
stride=1
pad=1
filters=255
activation=linear
*/


static kvp convolutional10[] = {
	{"size", "1", 0},
	{"stride", "1", 0},
	{"pad", "1", 0},
	{"filters", "255", 0},
	{"activation", "linear", 0},
};

/*
[yolo]
mask = 3,4,5
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes=80
num=6
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1
*/

static kvp yolo1[] = {
	{"mask", " 3,4,5", 0},
	{"anchors", " 10,14,  23,27,  37,58,  81,82,  135,169,  344,319", 0},
	{"classes", "80", 0},
	{"num", "6", 0},
	{"jitter", ".3", 0},
	{"ignore_thresh ", " .7", 0},
	{"truth_thresh ", " 1", 0},
	{"random", "1", 0},
};

/*
[route]
layers = -4
*/

static kvp route1[] = {
	{"layers", "-4", 0},
};

/*
[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky
*/

static kvp convolutional11[] = {
	{"batch_normalize", "1", 0},
	{"filters", "128", 0},
	{"size", "1", 0},
	{"stride", "1", 0},
	{"pad", "1", 0},
	{"activation", "leaky", 0},
};

/*
[upsample]
stride=2
*/

static kvp upsample1[] = {
	{"stride", "2", 0},
};

/*
[route]
layers = -1, 8
*/

static kvp route2[] = {
	{"layers", "-1, 8", 0},
};

/*
[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky
*/

static kvp convolutional12[] = {
	{"batch_normalize", "1", 0},
	{"filters", "256", 0},
	{"size", "3", 0},
	{"stride", "1", 0},
	{"pad", "1", 0},
	{"activation", "leaky", 0},
};

/*
[convolutional]
size=1
stride=1
pad=1
filters=255
activation=linear
*/

static kvp convolutional13[] = {
	{"size", "1", 0},
	{"stride", "1", 0},
	{"pad", "1", 0},
	{"filters", "255", 0},
	{"activation", "linear", 0},
};

/*
[yolo]
mask = 0,1,2
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes=80
num=6
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1
*/

static kvp yolo2[] = {
	{"mask", " 0,1,2", 0},
	{"anchors ", " 10,14,  23,27,  37,58,  81,82,  135,169,  344,319", 0},
	{"classes", "80", 0},
	{"num", "6", 0},
	{"jitter", ".3", 0},
	{"ignore_thresh ", " .7", 0},
	{"truth_thresh ", " 1", 0},
	{"random", "1", 0},
};


cfg_section yolov3_tiny_cfg[] = {
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
	{MAXPOOL, maxpool6, 2},
	{CONVOLUTIONAL, convolutional7, 6},
	{CONVOLUTIONAL, convolutional8, 6},
	{CONVOLUTIONAL, convolutional9, 6},
	{CONVOLUTIONAL, convolutional10, 5},
	{YOLO, yolo1, 8},
	{ROUTE, route1, 1},
	{CONVOLUTIONAL, convolutional11, 6},
	{UPSAMPLE, upsample1, 1},
	{ROUTE, route2, 1},
	{CONVOLUTIONAL, convolutional12, 6},
	{CONVOLUTIONAL, convolutional13, 5},
	{YOLO, yolo2, 8},
};

int yolov3_tiny_cfg_size = 25;
