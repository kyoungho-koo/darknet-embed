#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "logistic_layer.h"
#include "l2norm_layer.h"
//#include "activations.h"
//#include "batchnorm_layer.h"
#include "blas.h"
#include "convolutional_layer.h"
//#include "cost_layer.h"
//#include "list.h"
#include "local_layer.h"
#include "maxpool_layer.h"
//#include "normalization_layer.h"
#include "parser.h"
//#include "region_layer.h"
#include "yolo_layer.h"
//#include "reorg_layer.h"
#include "route_layer.h"
#include "upsample_layer.h"
#include "softmax_layer.h"
//#include "utils.h"


//#define DEBUG
#ifdef DEBUG
#define LOG_ARG(fmt, ...)	printf("[%s] " #__VA_ARGS__ "  (" fmt ") \n", __func__, ##__VA_ARGS__)
#define LOG(fmt, ...)	printf("[%s] " fmt "\n", __func__, ##__VA_ARGS__)
#else
#define LOG_ARG(fmt, ...)	
#define LOG(fmt, ...)	
#endif
/*
typedef struct{
    char *type;
    list *options;
}section;
*/

static inline char *__section_find(kvp *section, int size, char *key)
{
	int debug = 0;
	int i;
	if (!strcmp("padding", key)) {
		debug = 1;
		LOG("%p %d \n", section, size);
	}

	for (i = 0; i < size; i++) {
		if (debug == 1) {
			LOG(" section %d\n",i);
		}
		if(strcmp(section[i].key, key) == 0) {
			section[i].used = 1;
			return section[i].val;
		}

	}
    return 0;
}

static inline char *__section_find_str(kvp *section, int size, char *key, char *def)
{
    char *v = __section_find(section, size, key);
    if(v) 
		return v;
    return def;
}

static inline int __section_find_int(kvp *section, int size, char *key,const  int def)
{
    char *v = __section_find(section, size, key);
    if(v) 
		return atoi(v);
    return def;
}

static inline float __section_find_float(kvp *section, int size,char *key, float def)
{
    char *v = __section_find(section, size, key);
	if (v)
		return  atof(v);
	return def;
}
list *read_cfg(char *filename);

LAYER_TYPE string_to_layer_type(char * type)
{

    if (strcmp(type, "[shortcut]")==0) return SHORTCUT;
    if (strcmp(type, "[crop]")==0) return CROP;
    if (strcmp(type, "[cost]")==0) return COST;
    if (strcmp(type, "[detection]")==0) return DETECTION;
    if (strcmp(type, "[region]")==0) return REGION;
    if (strcmp(type, "[yolo]")==0) return YOLO;
    if (strcmp(type, "[iseg]")==0) return ISEG;
    if (strcmp(type, "[local]")==0) return LOCAL;
    if (strcmp(type, "[conv]")==0
            || strcmp(type, "[convolutional]")==0) return CONVOLUTIONAL;
    if (strcmp(type, "[deconv]")==0
            || strcmp(type, "[deconvolutional]")==0) return DECONVOLUTIONAL;
    if (strcmp(type, "[activation]")==0) return ACTIVE;
    if (strcmp(type, "[logistic]")==0) return LOGXENT;
    if (strcmp(type, "[l2norm]")==0) return L2NORM;
    if (strcmp(type, "[net]")==0
            || strcmp(type, "[network]")==0) return NETWORK;
    if (strcmp(type, "[crnn]")==0) return CRNN;
    if (strcmp(type, "[gru]")==0) return GRU;
    if (strcmp(type, "[lstm]") == 0) return LSTM;
    if (strcmp(type, "[rnn]")==0) return RNN;
    if (strcmp(type, "[conn]")==0
            || strcmp(type, "[connected]")==0) return CONNECTED;
    if (strcmp(type, "[max]")==0
            || strcmp(type, "[maxpool]")==0) return MAXPOOL;
    if (strcmp(type, "[reorg]")==0) return REORG;
    if (strcmp(type, "[avg]")==0
            || strcmp(type, "[avgpool]")==0) return AVGPOOL;
    if (strcmp(type, "[dropout]")==0) return DROPOUT;
    if (strcmp(type, "[lrn]")==0
            || strcmp(type, "[normalization]")==0) return NORMALIZATION;
    if (strcmp(type, "[batchnorm]")==0) return BATCHNORM;
    if (strcmp(type, "[soft]")==0
            || strcmp(type, "[softmax]")==0) return SOFTMAX;
    if (strcmp(type, "[route]")==0) return ROUTE;
    if (strcmp(type, "[upsample]")==0) return UPSAMPLE;
    return BLANK;
}


typedef struct size_params {
	int quantized;
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    network *net;
} size_params;


convolutional_layer parse_convolutional_sections(kvp *section, int size, size_params params)
{
    int n = __section_find_int(section, size, "filters",1);
    int layer_size = __section_find_int(section, size, "size",1);
    int stride = __section_find_int(section, size, "stride",1);
    int pad = __section_find_int(section, size, "pad",0);
    int padding = __section_find_int(section, size, "padding",0);
    int groups = __section_find_int(section, size, "groups", 1);
    if(pad) padding = layer_size/2;
//	printf("padding %d\n", padding);

    char *activation_s = __section_find_str(section, size, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
//	printf("[%s] %d %d %d\n", __func__,h,w,c);
    batch=params.batch;
    if(!(h && w && c)) error("Layer before convolutional layer must output image.");
    int batch_normalize = __section_find_int(section, size, "batch_normalize", 0);
    int binary = __section_find_int(section, size, "binary", 0);
    int xnor = __section_find_int(section, size, "xnor", 0);


    int quantized = params.quantized;
    if (params.index == 0 || activation == LINEAR || (params.index > 1 && stride>1) || size==1)
        quantized = 0; // disable Quantized for 1st and last layers
    convolutional_layer layer = make_convolutional_layer(batch,h,w,c,n,groups,layer_size,stride,padding,activation, batch_normalize, binary, xnor, params.net->adam, quantized);
    layer.flipped = __section_find_int(section, size, "flipped", 0);
    layer.dot = __section_find_float(section, size, "dot", 0);

    return layer;
	
}



int *parse_yolo_mask(char *a, int *num)
{
    int *mask = 0;
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        mask = calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            int val = atoi(a);
            mask[i] = val;
            a = strchr(a, ',')+1;
        }
        *num = n;
    }
    return mask;
}

layer parse_yolo_sections(kvp* section ,int size, size_params params)
{
    int classes = __section_find_int(section, size, "classes", 20);
    int total = __section_find_int(section, size, "num", 1);
    int num = total;

    char *a = __section_find_str(section, size, "mask", 0);
    int *mask = parse_yolo_mask(a, &num);
    layer l = make_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes);
    assert(l.outputs == params.inputs);

    l.max_boxes = __section_find_int(section, size, "max",90);
    l.jitter = __section_find_float(section, size, "jitter", .2);

    l.ignore_thresh = __section_find_float(section, size, "ignore_thresh", .5);
    l.truth_thresh = __section_find_float(section, size, "truth_thresh", 1);
    l.random = __section_find_int(section, size, "random", 0);


    a = __section_find_str(section, size, "anchors", 0);
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        for(i = 0; i < n; ++i){
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',')+1;
        }
    }
    return l;
}

maxpool_layer parse_maxpool_sections(kvp *section,int size, size_params params)
{
    int stride = __section_find_int(section, size, "stride",1);
    int layer_size = __section_find_int(section, size, "size",stride);
    int padding = __section_find_int(section, size, "padding", size-1);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before maxpool layer must output image.");

    maxpool_layer layer = make_maxpool_layer(batch,h,w,c,layer_size,stride,padding);
    return layer;
}




layer parse_upsample_sections(kvp *section, int size, size_params params, network *net)
{

    int stride = __section_find_int(section, size, "stride",2);
	LOG("%d %d %d\n",params.h,params.w,params.c);
    layer l = make_upsample_layer(params.batch, params.w, params.h, params.c, stride);
    l.scale = __section_find_float(section, size, "scale", 1);
    return l;
}


route_layer parse_route_sections(kvp *section, int size, size_params params, network *net)
{
    char *l = __section_find(section, size, "layers");
	//printf("[%s] %s\n", __func__, l);
    int len = strlen(l);
    if(!l) error("Route Layer must specify input layers");
    int n = 1;
    int i;
    for(i = 0; i < len; ++i){
        if (l[i] == ',') ++n;
    }
	//printf("[%s] %d\n", __func__, n);

    int *layers = calloc(n, sizeof(int));
    int *sizes = calloc(n, sizeof(int));
	//printf("[%s] ",__func__);
    for(i = 0; i < n; ++i){
        int index = atoi(l);
		//printf(" %s %d\n", l,index);
        l = strchr(l, ',')+1;
        if(index < 0) index = params.index + index;
        layers[i] = index;
        sizes[i] = net->layers[index].outputs;
    }
	//printf("\n");
    int batch = params.batch;

    route_layer layer = make_route_layer(batch, n, layers, sizes);

	LOG(" layers[0] %d\n",layers[0]);
    convolutional_layer first = net->layers[layers[0]];
    layer.out_w = first.out_w;
    layer.out_h = first.out_h;
    layer.out_c = first.out_c;
	LOG(" %d %d %d\n", layer.out_w, layer.out_h, layer.out_c);
    for(i = 1; i < n; ++i){
        int index = layers[i];
		LOG("<for> index: %d	i: %d \n", index, i);
        convolutional_layer next = net->layers[index];
		LOG("<for> %d\n", next.type);
		LOG("<for> %d %d %d\n", next.out_w, next.out_h, next.out_c);
        if(next.out_w == first.out_w && next.out_h == first.out_h){
            layer.out_c += next.out_c;
        }else{
            layer.out_h = layer.out_w = layer.out_c = 0;
        }
    }
	LOG("%d %d %d\n", layer.out_w, layer.out_h, layer.out_c);

    return layer;
}

learning_rate_policy get_policy(char *s)
{
    if (strcmp(s, "random")==0) return RANDOM;
    if (strcmp(s, "poly")==0) return POLY;
    if (strcmp(s, "constant")==0) return CONSTANT;
    if (strcmp(s, "step")==0) return STEP;
    if (strcmp(s, "exp")==0) return EXP;
    if (strcmp(s, "sigmoid")==0) return SIG;
    if (strcmp(s, "steps")==0) return STEPS;
    printf("Couldn't find policy %s, going with constant\n", s);
    return CONSTANT;
}



int bit_return(int a, int loc)
{
	int buf = a & 1 << loc;

	if (buf == 0) return 0;

	if (buf == 0) return 0;
	else return 1;
}



void parse_net_sections(kvp *section, int size, network *net)
{
    net->batch = __section_find_int(section, size, "batch",1);
    net->learning_rate = __section_find_float(section, size, "learning_rate", .001);
    net->momentum = __section_find_float(section, size, "momentum", .9);
    net->decay = __section_find_float(section, size, "decay", .0001);
    int subdivs = __section_find_int(section, size, "subdivisions",1);
    net->time_steps = __section_find_int(section, size, "time_steps",1);
    net->notruth = __section_find_int(section, size, "notruth",0);
    net->batch /= subdivs;
    net->batch *= net->time_steps;
    net->subdivisions = subdivs;
    net->random = __section_find_int(section, size, "random", 0);

    net->adam = __section_find_int(section, size, "adam", 0);
    if(net->adam){
        net->B1 = __section_find_float(section, size, "B1", .9);
        net->B2 = __section_find_float(section, size, "B2", .999);
        net->eps = __section_find_float(section, size, "eps", .0000001);
    }

    net->h = __section_find_int(section, size, "height",0);
    net->w = __section_find_int(section, size, "width",0);
    net->c = __section_find_int(section, size, "channels",0);
    net->inputs = __section_find_int(section, size, "inputs", net->h * net->w * net->c);
    net->max_crop = __section_find_int(section, size, "max_crop",net->w*2);
    net->min_crop = __section_find_int(section, size, "min_crop",net->w);
    net->max_ratio = __section_find_float(section, size, "max_ratio", (float) net->max_crop / net->w);
    net->min_ratio = __section_find_float(section, size, "min_ratio", (float) net->min_crop / net->w);
    net->center = __section_find_int(section, size, "center",0);
    net->clip = __section_find_float(section, size, "clip", 0);

    net->angle = __section_find_float(section, size, "angle", 0);
    net->aspect = __section_find_float(section, size, "aspect", 1);
    net->saturation = __section_find_float(section, size, "saturation", 1);
    net->exposure = __section_find_float(section, size, "exposure", 1);
    net->hue = __section_find_float(section, size, "hue", 0);

    if(!net->inputs && !(net->h && net->w && net->c)) error("No input parameters supplied");

    char *policy_s = __section_find_str(section, size, "policy", "constant");
    net->policy = get_policy(policy_s);
    net->burn_in = __section_find_int(section, size, "burn_in", 0);
    net->power = __section_find_float(section, size, "power", 4);
    if(net->policy == STEP){
        net->step = __section_find_int(section, size, "step", 1);
        net->scale = __section_find_float(section, size, "scale", 1);
    } else if (net->policy == STEPS){
        char *l = __section_find(section, size, "steps");
        char *p = __section_find(section, size, "scales");
        if(!l || !p) error("STEPS policy must have steps and scales in cfg file");

        int len = strlen(l);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (l[i] == ',') ++n;
        }
        int *steps = calloc(n, sizeof(int));
        float *scales = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            int step    = atoi(l);
            float scale = atof(p);
            l = strchr(l, ',')+1;
            p = strchr(p, ',')+1;
            steps[i] = step;
            scales[i] = scale;
        }
        net->scales = scales;
        net->steps = steps;
        net->num_steps = n;
    } else if (net->policy == EXP){
        net->gamma = __section_find_float(section, size, "gamma", 1);
    } else if (net->policy == SIG){
        net->gamma = __section_find_float(section, size, "gamma", 1);
        net->step = __section_find_int(section, size, "step", 1);
    } else if (net->policy == POLY || net->policy == RANDOM){
    }
    net->max_batches = __section_find_int(section, size, "max_batches", 0);
}

network *embed_parse_network_cfg(cfg_section *cfg, int size, int quantized)
{
    size_params params;

    network *net = make_network(size - 1);
	net->quantized = quantized;
	parse_net_sections(cfg[0].section, cfg[0].size, net);
	LOG_ARG("%d %f %f %f", 
			net->batch, 
			net->learning_rate, 
			net->momentum, 
			net->decay);

    params.h = net->h;
    params.w = net->w;
    params.c = net->c;
	params.quantized = quantized;
    params.inputs = net->inputs;
    params.batch = net->batch;
    params.time_steps = net->time_steps;
    params.net = net;

    size_t workspace_size = 0;
    int count = 0;
    int i;
    printf ("layer     filters    size              input                output\r\n");
	for (i = 1; i < size; i++) {
        params.index = count;
        printf("%5d ", count);
		cfg_section* s = &cfg[i];
		layer l = {0};
        LAYER_TYPE lt = s->type;
		
        if(lt == CONVOLUTIONAL){
            l = parse_convolutional_sections(s->section, s->size, params);
        } else if(lt == MAXPOOL){
            l = parse_maxpool_sections(s->section, s->size, params);
		} else if(lt == ROUTE){
            l = parse_route_sections(s->section, s->size, params, net);
		}else if(lt == YOLO){
            l = parse_yolo_sections(s->section, s->size, params);
		}else if(lt == UPSAMPLE){
            l = parse_upsample_sections(s->section, s->size, params, net);
		}
        l.clip = net->clip;
        l.truth = __section_find_int(s->section, s->size, "truth", 0);
        l.onlyforward = __section_find_int(s->section, s->size, "onlyforward", 0);
        l.stopbackward = __section_find_int(s->section, s->size, "stopbackward", 0);
        l.dontsave = __section_find_int(s->section, s->size, "dontsave", 0);
        l.dontload = __section_find_int(s->section, s->size, "dontload", 0);
        l.numload = __section_find_int(s->section, s->size, "numload", 0);
        l.dontloadscales = __section_find_int(s->section, s->size, "dontloadscales", 0);
        l.learning_rate_scale = __section_find_float(s->section, s->size, "learning_rate", 1);
        l.smooth = __section_find_float(s->section, s->size, "smooth", 0);
        net->layers[count] = l;

        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;

        ++count;
		params.h = l.out_h;
		params.w = l.out_w;
		params.c = l.out_c;
		//printf("%d %d %d\n", params.h, params.w, params.c);
		params.inputs = l.outputs;
	}
    layer out = get_network_output_layer(net);
    net->outputs = out.outputs;
    net->truths = out.outputs;
    if(net->layers[net->n-1].truths) net->truths = net->layers[net->n-1].truths;
    net->output = out.output;
    net->input = embed_calloc(net->inputs*net->batch, sizeof(float));
    net->truth = embed_calloc(net->truths*net->batch, sizeof(float));
    if(workspace_size){
        net->workspace = embed_calloc(1, workspace_size);
    }
    return net;
}


#ifdef TEST
extern unsigned char *yolov3_tiny_weights;
void test_load_convolutional_weights(layer* l, int* idx)
{
    if(l->numload) l->n =l->numload;
    int num = l->c/l->groups*l->n*l->size*l->size;

	l->biases =  (float *)&yolov3_tiny_weights[*idx];
	*idx += sizeof(float)*l->n;
    if (l->batch_normalize && (!l->dontloadscales)){
     //   fread(l.scales, sizeof(float), l.n, fp);
		l->scales =  (float *)&yolov3_tiny_weights[*idx];
		*idx += sizeof(float)*l->n;

		l->rolling_mean =  (float *)&yolov3_tiny_weights[*idx];
		*idx += sizeof(float)*l->n;

		l->rolling_variance =  (float *)&yolov3_tiny_weights[*idx];
		*idx += sizeof(float)*l->n;
    }
	l->weights =  (float *)&yolov3_tiny_weights[*idx];
	*idx += sizeof(float)*num;
}

void test_load_weights_upto(network *net, int start, int cutoff)
{
//	xil_printf("[%s]\n",__func__);
    int major;
    int minor;
    int revision;

	int idx = 0;

	
	memcpy(&major, yolov3_tiny_weights + idx, sizeof(int));
	idx+=4;
	memcpy(&minor, yolov3_tiny_weights + idx, sizeof(int));
	idx+=4;
	memcpy(&revision, yolov3_tiny_weights + idx, sizeof(int));
	idx+=4;

    if ((major*10 + minor) >= 2 && major < 1000 && minor < 1000){
		memcpy(net->seen, yolov3_tiny_weights + idx, sizeof(size_t));
		idx += sizeof(size_t);
    }

    int i;
    for(i = start; i < net->n && i < cutoff; ++i){
        layer* l = &net->layers[i];
        if(l->type == CONVOLUTIONAL || l->type == DECONVOLUTIONAL){
            test_load_convolutional_weights(l, &idx);
			if(0){
				int i;
				for(i = 0; i < l->n; ++i){
					printf("%g, ", l->rolling_mean[i]);
				}
				printf("\n");
				for(i = 0; i < l->n; ++i){
					printf("%g, ", l->rolling_variance[i]);
				}
				printf("\n");
			}
        }
    }
}

void test_load_weights(network *net)
{
//	xil_printf("[%s]\n",__func__);
    test_load_weights_upto(net, 0, net->n);
}
#endif
