#include <stdio.h>
#include <time.h>
#include <assert.h>
#include "network.h"


//#include "utils.h"
#include "blas.h"

#include "local_layer.h"
#include "convolutional_layer.h"
#include "yolo_layer.h"
#include "maxpool_layer.h"
#include "softmax_layer.h"
#include "route_layer.h"
#include "upsample_layer.h"
#include "parser.h"

load_args get_base_args(network *net)
{
    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.size = net->w;

    args.min = net->min_crop;
    args.max = net->max_crop;
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.center = net->center;
    args.saturation = net->saturation;
    args.hue = net->hue;
    return args;
}

#ifdef TEST
network *embed_load_network(cfg_section *cfg, int size, int clear, int quantized)
{
    //network *net = parse_network_cfg(cfg);
    xil_printf("[%s]\r\n",__func__);
    network *net = embed_parse_network_cfg(cfg, size, quantized);
    test_load_weights(net);
    if(clear) (*net->seen) = 0;
    return net;
}
#endif

size_t get_current_batch(network *net)
{
    size_t batch_num = (*net->seen)/(net->batch*net->subdivisions);
    return batch_num;
}

void reset_network_state(network *net, int b)
{
    int i;
    for (i = 0; i < net->n; ++i) {
    }
}

void reset_rnn(network *net)
{
    reset_network_state(net, 0);
}

/*
float get_current_rate(network *net)
{
    size_t batch_num = get_current_batch(net);
    int i;
    float rate;
    if (batch_num < net->burn_in) return net->learning_rate * pow((float)batch_num / net->burn_in, net->power);
    switch (net->policy) {
        case CONSTANT:
            return net->learning_rate;
        case STEP:
            return net->learning_rate * pow(net->scale, batch_num/net->step);
        case STEPS:
            rate = net->learning_rate;
            for(i = 0; i < net->num_steps; ++i){
                if(net->steps[i] > batch_num) return rate;
                rate *= net->scales[i];
            }
            return rate;
        case EXP:
            return net->learning_rate * pow(net->gamma, batch_num);
        case POLY:
            return net->learning_rate * pow(1 - (float)batch_num / net->max_batches, net->power);
        case RANDOM:
            return net->learning_rate * pow(rand_uniform(0,1), net->power);
        case SIG:
            return net->learning_rate * (1./(1.+exp(net->gamma*(batch_num - net->step))));
        default:
            fprintf(stderr, "Policy is weird!\n");
            return net->learning_rate;
    }
}

*/

char *get_layer_string(LAYER_TYPE a)
{
    switch(a){
        case CONVOLUTIONAL:
            return "convolutional";
        case ACTIVE:
            return "activation";
        case LOCAL:
            return "local";
        case DECONVOLUTIONAL:
            return "deconvolutional";
        case CONNECTED:
            return "connected";
        case RNN:
            return "rnn";
        case GRU:
            return "gru";
        case LSTM:
	    return "lstm";
        case CRNN:
            return "crnn";
        case MAXPOOL:
            return "maxpool";
        case REORG:
            return "reorg";
        case AVGPOOL:
            return "avgpool";
        case SOFTMAX:
            return "softmax";
        case DETECTION:
            return "detection";
        case REGION:
            return "region";
        case YOLO:
            return "yolo";
        case DROPOUT:
            return "dropout";
        case CROP:
            return "crop";
        case COST:
            return "cost";
        case ROUTE:
            return "route";
        case SHORTCUT:
            return "shortcut";
        case NORMALIZATION:
            return "normalization";
        case BATCHNORM:
            return "batchnorm";
        default:
            break;
    }
    return "none";
}

network *make_network(int n)
{
    network *net = calloc(1, sizeof(network));
	printf ("sizeof network : %d\r\n", sizeof(network));
    net->n = n;
    net->layers = calloc(net->n, sizeof(layer));
	printf ("layer : %d\r\n", sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));
    return net;
}

void forward_network(network *netp)
{
//	xil_printf("\n[%s]\r\n",__func__);
    network net = *netp;
    int i;
//  xil_printf("[%s] net.n : %d \r\n",__func__, net.n);
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.delta){
            fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
        }
        l.forward(l, net);
        net.input = l.output;
        if(l.truth) {
            net.truth = l.output;
        }
    }
    calc_network_cost(netp);
}


void calc_network_cost(network *netp)
{
//	xil_printf("\n[%s]\n",__func__);
    network net = *netp;
    int i;
    float sum = 0;
    int count = 0;
    for(i = 0; i < net.n; ++i){
        if(net.layers[i].cost){
            sum += net.layers[i].cost[0];
            ++count;
        }
    }
    *net.cost = sum/count;
}

int get_predicted_class_network(network *net)
{
    return max_index(net->output, net->outputs);
}

/*
void backward_network(network *netp)
{
    network net = *netp;
    int i;
    network orig = net;
    for(i = net.n-1; i >= 0; --i){
        layer l = net.layers[i];
        if(l.stopbackward) break;
        if(i == 0){
            net = orig;
        }else{
            layer prev = net.layers[i-1];
            net.input = prev.output;
            net.delta = prev.delta;
        }
        net.index = i;
        l.backward(l, net);
    }
}
*/


void set_temp_network(network *net, float t)
{
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].temperature = t;
    }
}


void set_batch_network(network *net, int b)
{
    net->batch = b;
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].batch = b;
    }
}

layer get_network_detection_layer(network *net)
{
    int i;
    for(i = 0; i < net->n; ++i){
        if(net->layers[i].type == DETECTION){
            return net->layers[i];
        }
    }
    fprintf(stderr, "Detection layer not found!!\n");
    layer l = {0};
    return l;
}



void top_predictions(network *net, int k, int *index)
{
    top_k(net->output, net->outputs, k, index);
}


float *network_predict(network *net, float *input)
{
//	xil_printf("\n[%s]\n",__func__);
    network orig = *net;
    net->input = input;
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
    forward_network(net);
    float *out = net->output;
    *net = orig;
    return out;
}

int num_detections(network *net, float thresh)
{
    int i;
    int s = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
		printf("l.type %d\n",l.type);

        if(l.type == YOLO){
            s += yolo_num_detections(l, thresh);
        }
        if(l.type == DETECTION || l.type == REGION){
            s += l.w*l.h*l.n;
        }
    }
    return s;
}

detection *make_network_boxes(network *net, float thresh, int *num)
{
    layer l = net->layers[net->n - 1];
    int i;
    int nboxes = num_detections(net, thresh);
	printf("nboxes %d\n", nboxes);
    if(num) *num = nboxes;
    detection *dets = calloc(nboxes, sizeof(detection));
    for(i = 0; i < nboxes; ++i){
        dets[i].prob = calloc(l.classes, sizeof(float));
        if(l.coords > 4){
            dets[i].mask = calloc(l.coords-4, sizeof(float));
        }
    }
    return dets;
}

void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets)
{
    int j;
    for(j = 0; j < net->n; ++j){
        layer l = net->layers[j];
		if(0){
			int i;
			for(i = 0; i < l.n; ++i){
				printf("%g, ", l.rolling_mean[i]);
			}
			printf("\n");
			for(i = 0; i < l.n; ++i){
				printf("%g, ", l.rolling_variance[i]);
			}
			printf("\n");
		}
        if(l.type == YOLO){
            int count = get_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets);
            dets += count;
        }
    }
}

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num)
{
    detection *dets = make_network_boxes(net, thresh, num);
    fill_network_boxes(net, w, h, thresh, hier, map, relative, dets);
    return dets;
}

void free_detections(detection *dets, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        free(dets[i].prob);
        if(dets[i].mask) free(dets[i].mask);
    }
    free(dets);
}


int network_width(network *net){return net->w;}
int network_height(network *net){return net->h;}

matrix network_predict_data_multi(network *net, data test, int n)
{
    int i,j,b,m;
    int k = net->outputs;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net->batch*test.X.rows, sizeof(float));
    for(i = 0; i < test.X.rows; i += net->batch){
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        for(m = 0; m < n; ++m){
            float *out = network_predict(net, X);
            for(b = 0; b < net->batch; ++b){
                if(i+b == test.X.rows) break;
                for(j = 0; j < k; ++j){
                    pred.vals[i+b][j] += out[j+b*k]/n;
                }
            }
        }
    }
    free(X);
    return pred;   
}

matrix network_predict_data(network *net, data test)
{
    int i,j,b;
    int k = net->outputs;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net->batch*test.X.cols, sizeof(float));
    for(i = 0; i < test.X.rows; i += net->batch){
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        float *out = network_predict(net, X);
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            for(j = 0; j < k; ++j){
                pred.vals[i+b][j] = out[j+b*k];
            }
        }
    }
    free(X);
    return pred;   
}

void print_network(network *net)
{
    int i,j;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        float *output = l.output;
        int n = l.outputs;
        float mean = mean_array(output, n);
        float vari = variance_array(output, n);
        fprintf(stderr, "Layer %d - Mean: %f, Variance: %f\n",i,mean, vari);
        if(n > 100) n = 100;
        for(j = 0; j < n; ++j) fprintf(stderr, "%f, ", output[j]);
        if(n == 100)fprintf(stderr,".....\n");
        fprintf(stderr, "\n");
    }
}


float network_accuracy(network *net, data d)
{
    matrix guess = network_predict_data(net, d);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

float *network_accuracies(network *net, data d, int n)
{
    static float acc[2];
    matrix guess = network_predict_data(net, d);
    acc[0] = matrix_topk_accuracy(d.y, guess, 1);
    acc[1] = matrix_topk_accuracy(d.y, guess, n);
    free_matrix(guess);
    return acc;
}

layer get_network_output_layer(network *net)
{
    int i;
    for(i = net->n - 1; i >= 0; --i){
        if(net->layers[i].type != COST) break;
    }
    return net->layers[i];
}

float network_accuracy_multi(network *net, data d, int n)
{
    matrix guess = network_predict_data_multi(net, d, n);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

void free_network(network *net)
{
    int i;
    for(i = 0; i < net->n; ++i){
        free_layer(net->layers[i]);
    }
    free(net->layers);
    if(net->input) free(net->input);
    if(net->truth) free(net->truth);
    free(net);
}

// Some day...
// ^ What the hell is this comment for?


layer network_output_layer(network *net)
{
    int i;
    for(i = net->n - 1; i >= 0; --i){
        if(net->layers[i].type != COST) break;
    }
    return net->layers[i];
}

int network_inputs(network *net)
{
    return net->layers[0].inputs;
}

int network_outputs(network *net)
{
    return network_output_layer(net).outputs;
}

float *network_output(network *net)
{
    return network_output_layer(net).output;
}

