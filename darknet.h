#ifndef DARKNET_API
#define DARKNET_API
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "debug.h"
#ifdef OPENSSD
#include "../memory_map.h"
#endif

#define TEST
#define SECRET_NUM -1234

#define SMART_SSD
#define YOLOV3_WEIGHTS_SIZE   0x21CB1CC


void *embed_calloc(int num_val, int size_val) ;
void embed_free (void *p) ;

typedef struct{
    int classes;
    char **names;
} metadata;

metadata get_metadata(char *file);

typedef struct{
    int *leaf;
    int n;
    int *parent;
    int *child;
    int *group;
    char **name;

    int groups;
    int *group_size;
    int *group_offset;
} tree;
tree *read_tree(char *filename);

typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU
} ACTIVATION;

typedef enum{
    PNG, BMP, TGA, JPG
} IMTYPE;

typedef enum{
    MULT, ADD, SUB, DIV
} BINARY_ACTIVATION;

typedef enum {
    CONVOLUTIONAL,	// 
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,		//
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,			//
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    GRU,
    LSTM,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    YOLO,			//
    ISEG,
    REORG,
    UPSAMPLE,		//
    LOGXENT,
    L2NORM,
    BLANK
} LAYER_TYPE;

typedef enum{
    SSE, MASKED, L1, SEG, SMOOTH,WGAN
} COST_TYPE;

typedef struct{
    int batch;
    float learning_rate;
    float momentum;
    float decay;
    int adam;
    float B1;
    float B2;
    float eps;
    int t;
} update_args;

struct network;
typedef struct network network;

struct layer;
struct layer_q;
typedef struct layer layer;
typedef struct layer_q layer_q;

struct layer {
    LAYER_TYPE type;
    ACTIVATION activation;
    COST_TYPE cost_type;
    void (*forward)   (struct layer, struct network);
    void (*backward)  (struct layer, struct network);
    void (*update)    (struct layer, update_args);
    int batch_normalize;
    int shortcut;
    int batch;
    int forced;
    int flipped;
    int inputs;
    int outputs;
    int nweights;
    int nbiases;
    int extra;
    int truths;
    int h,w,c;
    int out_h, out_w, out_c;
    int n;
    int max_boxes;
    int groups;
    int size;
    int side;
    int stride;
    int reverse;
    int flatten;
    int spatial;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;
    int xnor;
    int steps;
    int hidden;
    int truth;
    float smooth;
    float dot;
    float angle;
    float jitter;
    float saturation;
    float exposure;
    float shift;
    float ratio;
    float learning_rate_scale;
    float clip;
    int noloss;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int joint;
    int noadjust;
    int reorg;
    int log;
    int tanh;
    int *mask;
    int total;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float mask_scale;
    float class_scale;
    int bias_match;
    int random;
    float ignore_thresh;
    float truth_thresh;
    float thresh;
    float focus;
    int classfix;
    int absolute;

    int onlyforward;
    int stopbackward;
    int dontload;
    int dontsave;
    int dontloadscales;
    int numload;

    float temperature;
    float probability;
    float scale;

    char  * cweights;
    int   * indexes;
    int   * input_layers;
    int   * input_sizes;
    int   * map;
    int   * counts;
    float ** sums;
    float * rand;
    float * cost;
    float * state;
    float * prev_state;
    float * forgot_state;
    float * forgot_delta;
    float * state_delta;
    float * combine_cpu;
    float * combine_delta_cpu;

    float * concat;
    float * concat_delta;

    float * binary_weights;

    float * biases;
    float * bias_updates;

	int quantized;

    float * scales;
    float * scale_updates;

    float * weights;
	int8_t * weights_int8;
    float * weight_updates;
	int8_t * output_int8;
	float weights_quant_multipler;
	float input_quant_multipler;

    float * delta;
    float * output;
    float * loss;
    float * squared;
    float * norms;

    float * spatial_mean;
    float * mean;
    float * variance;

    float * mean_delta;
    float * variance_delta;

    float * rolling_mean;
    float * rolling_variance;

    float * x;
    float * x_norm;

    float * m;
    float * v;
    
    float * bias_m;
    float * bias_v;
    float * scale_m;
    float * scale_v;


    float *z_cpu;
    float *r_cpu;
    float *h_cpu;
    float * prev_state_cpu;

    float *temp_cpu;
    float *temp2_cpu;
    float *temp3_cpu;

    float *dh_cpu;
    float *hh_cpu;
    float *prev_cell_cpu;
    float *cell_cpu;
    float *f_cpu;
    float *i_cpu;
    float *g_cpu;
    float *o_cpu;
    float *c_cpu;
    float *dc_cpu; 

    float * binary_input;

    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    struct layer *reset_layer;
    struct layer *update_layer;
    struct layer *state_layer;

    struct layer *input_gate_layer;
    struct layer *state_gate_layer;
    struct layer *input_save_layer;
    struct layer *state_save_layer;
    struct layer *input_state_layer;
    struct layer *state_state_layer;

    struct layer *input_z_layer;
    struct layer *state_z_layer;

    struct layer *input_r_layer;
    struct layer *state_r_layer;

    struct layer *input_h_layer;
    struct layer *state_h_layer;
	
    struct layer *wz;
    struct layer *uz;
    struct layer *wr;
    struct layer *ur;
    struct layer *wh;
    struct layer *uh;
    struct layer *uo;
    struct layer *wo;
    struct layer *uf;
    struct layer *wf;
    struct layer *ui;
    struct layer *wi;
    struct layer *ug;
    struct layer *wg;

    tree *softmax_tree;

    size_t workspace_size;
};

struct layer_q {
	LAYER_TYPE type;
	ACTIVATION activation;
	COST_TYPE cost_type;
	void(*forward)   (struct layer_q, struct network_state);
	void(*backward)  (struct layer_q, struct network_state);
	void(*update)    (struct layer_q, int, float, float, float);
	int batch_normalize;
	int shortcut;
	int batch;
	int forced;
	int flipped;
	int inputs;
	int outputs;
	int truths;
	int h, w, c;
	int out_h, out_w, out_c;
	int n;
	int max_boxes;
	int groups;
	int size;
	int side;
	int stride;
	int reverse;
	int pad;
	int sqrt;
	int flip;
	int index;
	int binary;
	int xnor;
	int use_bin_output;
	int steps;
	int hidden;
	float dot;
	float angle;
	float jitter;
	float saturation;
	float exposure;
	float shift;
	float ratio;
	int focal_loss;
	int softmax;
	int classes;
	int coords;
	int background;
	int rescore;
	int objectness;
	int does_cost;
	int joint;
	int noadjust;
	int reorg;
	int log;

	int *mask;
	int total;
	float bflops;

	int adam;
	float B1;
	float B2;
	float eps;
	float *m_gpu;
	float *v_gpu;
	int t;
	float *m;
	float *v;

	tree *softmax_tree;
	int  *map;

	float alpha;
	float beta;
	float kappa;

	float coord_scale;
	float object_scale;
	float noobject_scale;
	float class_scale;
	int bias_match;
	int random;
	float ignore_thresh;
	float truth_thresh;
	float thresh;
	int classfix;
	int absolute;

	int dontload;
	int dontloadscales;

	float temperature;
	float probability;
	float scale;

	int *indexes;
	float *rand;
	float *cost;
	char  *cweights;
	float *state;
	float *prev_state;
	float *forgot_state;
	float *forgot_delta;
	float *state_delta;

	float *concat;
	float *concat_delta;

	float *binary_weights;

	char *align_bit_weights_gpu;
	float *mean_arr_gpu;
	float *align_workspace_gpu;
	float *transposed_align_workspace_gpu;
	int align_workspace_size;

	char *align_bit_weights;
	float *mean_arr;
	int align_bit_weights_size;
	int lda_align;
	int new_lda;
	int bit_align;

	float *biases;
	float *biases_quant;
	float *bias_updates;

	int quantized;

	float *scales;
	float *scale_updates;

	float *weights;
	int8_t * weights_int8;
	float *weight_updates;
	//float *weights_quant_multipler;
	float weights_quant_multipler;
	float input_quant_multipler;

	float *col_image;
	int   * input_layers;
	int   * input_sizes;
	float * delta;
	float * output;
	int output_pinned;
	//float *output_multipler;
	float output_multipler;
	int8_t * output_int8;
	float * squared;
	float * norms;

	float * spatial_mean;
	float * mean;
	float * variance;

	//float * mean_delta;
	//float * variance_delta;

	float * rolling_mean;
	float * rolling_variance;

	float * x;
	float * x_norm;

	struct layer *input_layer;
	struct layer *self_layer;
	struct layer *output_layer;

	struct layer *input_gate_layer;
	struct layer *state_gate_layer;
	struct layer *input_save_layer;
	struct layer *state_save_layer;
	struct layer *input_state_layer;
	struct layer *state_state_layer;

	struct layer *input_z_layer;
	struct layer *state_z_layer;

	struct layer *input_r_layer;
	struct layer *state_r_layer;

	struct layer *input_h_layer;
	struct layer *state_h_layer;

	float *z_cpu;
	float *r_cpu;
	float *h_cpu;

	float *binary_input;
	uint32_t *bin_re_packed_input;
	char *t_bit_input;

	size_t workspace_size;
};

void free_layer(layer);

typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;

typedef struct network{
	int quantized;
    int n;
    int batch;
    size_t *seen;
    int *t;
    float epoch;
    int subdivisions;
    layer *layers;
    float *output;
    learning_rate_policy policy;

    float learning_rate;
    float momentum;
    float decay;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int outputs;
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    float max_ratio;
    float min_ratio;
    int center;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
    int random;

    tree *hierarchy;

    float *input;
	int8_t *input_int8;
    float *truth;
    float *delta;
    float *workspace;
    int train;
    int index;
    float *cost;
    float clip;


} network;

// for Quantization
typedef struct network_q {
	int quantized;
	float *workspace;
	int n;
	int batch;
	float *input_calibration;
	int input_calibration_size;
	uint64_t *seen;
	float epoch;
	int subdivisions;
	float momentum;
	float decay;
	layer_q *layers;
	int outputs;
	float *output;
	learning_rate_policy policy;

	float learning_rate;
	float gamma;
	float scale;
	float power;
	int time_steps;
	int step;
	int max_batches;
	float *scales;
	int   *steps;
	int num_steps;
	int burn_in;

	int adam;
	float B1;
	float B2;
	float eps;

	int inputs;
	int h, w, c;
	int max_crop;
	int min_crop;
	float angle;
	float aspect;
	float exposure;
	float saturation;
	float hue;

	tree *hierarchy;
	int do_input_calibration;
} network_q;

// for Quantization
typedef struct network_state {
	float *truth;
	float *input;
	int8_t *input_int8;
	float *delta;
	float *workspace;
	int train;
	int index;
	network_q net;
} network_state;

typedef struct {
    int w;
    int h;
    float scale;
    float rad;
    float dx;
    float dy;
    float aspect;
} augment_args;

typedef struct {
    int w;
    int h;
    int c;
    float *data;
} image;

typedef struct{
    float x, y, w, h;
} box;

typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} detection;

typedef struct matrix{
    int rows, cols;
    float **vals;
} matrix;


typedef struct{
    int w, h;
    matrix X;
    matrix y;
    int shallow;
    int *num_boxes;
    box **boxes;
} data;

typedef enum {
    CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA, COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA, STUDY_DATA, DET_DATA, SUPER_DATA, LETTERBOX_DATA, REGRESSION_DATA, SEGMENTATION_DATA, INSTANCE_DATA, ISEG_DATA
} data_type;

typedef struct load_args{
    int threads;
    char **paths;
    char *path;
    int n;
    int m;
    char **labels;
    int h;
    int w;
    int out_w;
    int out_h;
    int nh;
    int nw;
    int num_boxes;
    int min, max, size;
    int classes;
    int background;
    int scale;
    int center;
    int coords;
    float jitter;
    float angle;
    float aspect;
    float saturation;
    float exposure;
    float hue;
    data *d;
    image *im;
    image *resized;
    data_type type;
    tree *hierarchy;
} load_args;

typedef struct{
    int id;
    float x,y,w,h;
    float left, right, top, bottom;
} box_label;


typedef struct{
    char *key;
    char *val;
    int used;
} kvp;

typedef struct cfg_section {
	LAYER_TYPE type;
	kvp *section;
	int size;
} cfg_section;

network *load_network(char *cfg, char *weights, int clear);
#ifdef TEST
network *embed_load_network(cfg_section *cfg, int size, int clear, int quantized);
#endif
load_args get_base_args(network *net);

void free_data(data d);

typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

typedef struct list{
    int size;
    node *front;
    node *back;
} list;

list *read_data_cfg(char *filename);
list *read_cfg(char *filename);
unsigned char *read_file(char *filename);
data resize_data(data orig, int w, int h);
data *tile_data(data orig, int divs, int size);
data select_data(data *orig, int *inds);

void forward_network(network *net);
void backward_network(network *net);
void update_network(network *net);


float dot_cpu(int N, float *X, int INCX, float *Y, int INCY);
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
void scal_cpu(int N, float ALPHA, float *X, int INCX);
void fill_cpu(int N, float ALPHA, float * X, int INCX);
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
void softmax(float *input, int n, float temp, int stride, float *output);

int best_3d_shift_r(image a, image b, int min, int max);
#ifdef GPU
void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY);
void fill_gpu(int N, float ALPHA, float * X, int INCX);
void scal_gpu(int N, float ALPHA, float * X, int INCX);
void copy_gpu(int N, float * X, int INCX, float * Y, int INCY);

void cuda_set_device(int n);
void cuda_free(float *x_gpu);
float *cuda_make_array(float *x, size_t n);
void cuda_pull_array(float *x_gpu, float *x, size_t n);
float cuda_mag_array(float *x_gpu, size_t n);
void cuda_push_array(float *x_gpu, float *x, size_t n);

void forward_network_gpu(network *net);
void backward_network_gpu(network *net);
void update_network_gpu(network *net);

float train_networks(network **nets, int n, data d, int interval);
void sync_nets(network **nets, int n, int interval);
void harmless_update_network_gpu(network *net);
#endif
image get_label(image **characters, char *string, int size);
void draw_label(image a, int r, int c, image label, const float *rgb);
void save_image(image im, const char *name);
void save_image_options(image im, const char *name, IMTYPE f, int quality);
void get_next_batch(data d, int n, int offset, float *X, float *y);
void grayscale_image_3c(image im);
void normalize_image(image p);
void matrix_to_csv(matrix m);
float train_network_sgd(network *net, data d, int n);
void rgbgr_image(image im);
data copy_data(data d);
data concat_data(data d1, data d2);
data load_cifar10_data(char *filename);
float matrix_topk_accuracy(matrix truth, matrix guess, int k);
void matrix_add_matrix(matrix from, matrix to);
void scale_matrix(matrix m, float scale);
matrix csv_to_matrix(char *filename);
float *network_accuracies(network *net, data d, int n);
float train_network_datum(network *net);
image make_random_image(int w, int h, int c);

void denormalize_connected_layer(layer l);
void denormalize_convolutional_layer(layer l);
void statistics_connected_layer(layer l);
void rescale_weights(layer l, float scale, float trans);
void rgbgr_weights(layer l);
image *get_weights(layer l);

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, int avg, float hier_thresh, int w, int h, int fps, int fullscreen);
void get_detection_detections(layer l, int w, int h, float thresh, detection *dets);

char *option_find_str(list *l, char *key, char *def);
int option_find_int(list *l, char *key, int def);
int option_find_int_quiet(list *l, char *key, int def);

#ifdef TEST
network *embed_parse_network_cfg(cfg_section *cfg, int size, int quantized);
#endif
network *parse_network_cfg(char *filename);
void save_weights(network *net, char *filename);
void load_weights(network *net, char *filename);
void save_weights_upto(network *net, char *filename, int cutoff);
void load_weights_upto(network *net, char *filename, int start, int cutoff);
#ifdef TEST
void test_load_weights_upto(network *net,int start, int cutoff);
#endif

void zero_objectness(layer l);
void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets);
int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets);
void free_network(network *net);
void set_batch_network(network *net, int b);
void set_temp_network(network *net, float t);
image load_image(char *filename, int w, int h, int c);
image load_image_color(char *filename, int w, int h);
image make_image(int w, int h, int c);
image resize_image(image im, int w, int h);
void censor_image(image im, int dx, int dy, int w, int h);
image letterbox_image(image im, int w, int h);
image crop_image(image im, int dx, int dy, int w, int h);
image center_crop_image(image im, int w, int h);
image resize_min(image im, int min);
image resize_max(image im, int max);
image threshold_image(image im, float thresh);
image mask_to_rgb(image mask);
int resize_network(network *net, int w, int h);
void free_matrix(matrix m);
void test_resize(char *filename);
int show_image(image p, const char *name, int ms);
image copy_image(image p);
void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);
float get_current_rate(network *net);
void composite_3d(char *f1, char *f2, char *out, int delta);
data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h);
size_t get_current_batch(network *net);
void constrain_image(image im);
image get_network_image_layer(network *net, int i);
layer get_network_output_layer(network *net);
void top_predictions(network *net, int n, int *index);
void flip_image(image a);
image float_to_image(int w, int h, int c, float *data);
void ghost_image(image source, image dest, int dx, int dy);
float network_accuracy(network *net, data d);
void random_distort_image(image im, float hue, float saturation, float exposure);
void fill_image(image m, float s);
image grayscale_image(image im);
void rotate_image_cw(image im, int times);
double what_time_is_it_now();
image rotate_image(image m, float rad);
void visualize_network(network *net);
float box_iou(box a, box b);
data load_all_cifar10();
box_label *read_boxes(char *filename, int *n);
box float_to_box(float *f, int stride);
void draw_detections(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes);

matrix network_predict_data(network *net, data test);
image **load_alphabet();
image get_network_image(network *net);
float *network_predict(network *net, float *input);

int network_width(network *net);
int network_height(network *net);
float *network_predict_image(network *net, image im);
void network_detect(network *net, image im, float thresh, float hier_thresh, float nms, detection *dets);
detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);
void free_detections(detection *dets, int n);

void reset_network_state(network *net, int b);

char **get_labels(char *filename);
void do_nms_obj(detection *dets, int total, int classes, float thresh);
void do_nms_sort(detection *dets, int total, int classes, float thresh);

matrix make_matrix(int rows, int cols);


void free_image(image m);
float train_network(network *net, data d);
void load_data_blocking(load_args args);
list *get_paths(char *filename);
void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves, int stride);
void change_leaves(tree *t, char *leaf_list);

int find_int_arg(int argc, char **argv, char *arg, int def);
float find_float_arg(int argc, char **argv, char *arg, float def);
int find_arg(int argc, char* argv[], char *arg);
char *find_char_arg(int argc, char **argv, char *arg, char *def);
char *basecfg(char *cfgfile);
void find_replace(char *str, char *orig, char *rep, char *output);
void free_ptrs(void **ptrs, int n);
char *fgetl(FILE *fp);
void strip(char *s);
float sec(clock_t clocks);
void **list_to_array(list *l);
void top_k(float *a, int n, int k, int *index);
int *read_map(char *filename);
void error(const char *s);
int max_index(float *a, int n);
int max_int_index(int *a, int n);
int sample_array(float *a, int n);
int *random_index_order(int min, int max);
void free_list(list *l);
float mse_array(float *a, int n);
float variance_array(float *a, int n);
float mag_array(float *a, int n);
void scale_array(float *a, int n, float s);
float mean_array(float *a, int n);
float sum_array(float *a, int n);
void normalize_array(float *a, int n);
int *read_intlist(char *s, int *n, int d);
size_t rand_size_t();
float rand_normal();
float rand_uniform(float min, float max);


void darknet_main();
#ifdef __cplusplus
}
#endif
#endif
