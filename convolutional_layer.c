#include "convolutional_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include "debug.h"
#include "math.h"

#ifdef AI2
#include "xnor_layer.h"
#endif

#define I_MAX_VAL (256/2 - 1)    // 7-bit (1-bit sign)
#define R_MULT (32)    // 4 - 32

void swap_binary(convolutional_layer *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;

#ifdef GPU
    swap = l->weights_gpu;
    l->weights_gpu = l->binary_weights_gpu;
    l->binary_weights_gpu = swap;
#endif
}


void binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

/*
void binarize_input(float *input, int n, int size, float *binary)
{
    int i, s;
    for(s = 0; s < size; ++s){
        float mean = 0;
        for(i = 0; i < n; ++i){
            mean += fabs(input[i*size + s]);
        }
        mean = mean / n;
        for(i = 0; i < n; ++i){
            binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
        }
    }
}
*/

int convolutional_out_height(convolutional_layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width(convolutional_layer l)
{
//	printf("[%s] %d %d %d %d\n", __func__,l.w, l.pad,l.size, l.stride);
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}

static size_t get_workspace_size(layer l){
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c/l.groups*sizeof(float);
}


convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, int quantization)
{
//    int i;
    convolutional_layer l = {0};
    l.type = CONVOLUTIONAL;

    l.groups = groups;
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

//   l.weights = calloc(c/groups*n*size*size, sizeof(float));
//   l.weight_updates = calloc(c/groups*n*size*size, sizeof(float));
	l.weights_int8 = (int8_t *)embed_calloc(c/groups*n*size*size, sizeof(int8_t));

//    l.biases = calloc(n, sizeof(float));
//    l.bias_updates = calloc(n, sizeof(float));

    l.nweights = c/groups*n*size*size;
    l.nbiases = n;

    // float scale = 1./sqrt(size*size*c);
//    float scale = sqrt(2./(size*size*c/l.groups));
    //printf("convscale %f\n", scale);
    //scale = .02;
    //for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
//    for(i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_normal();
    int out_w = convolutional_out_width(l);
    int out_h = convolutional_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
	//printf("[%s] %d %d %d\n", __func__,l.out_h, l.out_w,l.out_c);
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = embed_calloc(l.batch*l.outputs, sizeof(float));
//    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

	if (!quantization) {
		l.forward = forward_convolutional_layer;
	} else {
		l.forward = forward_convolutional_layer_q;
	}


    l.update = update_convolutional_layer;
	
	//printf("[%s] %d %d %d %d\n", __func__,binary, xnor, batch_normalize, adam);
	/*
    if(binary){
        l.binary_weights = calloc(l.nweights, sizeof(float));
        l.cweights = calloc(l.nweights, sizeof(char));
        l.scales = calloc(n, sizeof(float));
    }
    if(xnor){
        l.binary_weights = calloc(l.nweights, sizeof(float));
        l.binary_input = calloc(l.inputs*l.batch, sizeof(float));
    }
	*/

    if(batch_normalize){
        //l.scales = calloc(n, sizeof(float));
        //l.scale_updates = calloc(n, sizeof(float));
		/*
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }
		*/

        //l.mean = calloc(n, sizeof(float));
        //l.variance = calloc(n, sizeof(float));

        //l.mean_delta = calloc(n, sizeof(float));
        //l.variance_delta = calloc(n, sizeof(float));

        //l.rolling_mean = calloc(n, sizeof(float));
        //l.rolling_variance = calloc(n, sizeof(float));
        l.x = embed_calloc(l.batch*l.outputs, sizeof(float));
       // l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }
	/*
    if(adam){
        l.m = calloc(l.nweights, sizeof(float));
        l.v = calloc(l.nweights, sizeof(float));
        l.bias_m = calloc(n, sizeof(float));
        l.scale_m = calloc(n, sizeof(float));
        l.bias_v = calloc(n, sizeof(float));
        l.scale_v = calloc(n, sizeof(float));
    }
	*/

    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    printf("conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\r\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);

    return l;
}


/*
void test_convolutional_layer()
{
    convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0, 0, 0);
    l.batch_normalize = 1;
    float data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3};
    //net.input = data;
    //forward_convolutional_layer(l);
}
*/

void resize_convolutional_layer(convolutional_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    int out_w = convolutional_out_width(*l);
    int out_h = convolutional_out_height(*l);

    l->out_w = out_w;
    l->out_h = out_h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = realloc(l->delta,  l->batch*l->outputs*sizeof(float));
    if(l->batch_normalize){
        l->x = realloc(l->x, l->batch*l->outputs*sizeof(float));
        l->x_norm  = realloc(l->x_norm, l->batch*l->outputs*sizeof(float));
    }

    l->workspace_size = get_workspace_size(*l);
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}


void forward_convolutional_layer(convolutional_layer l, network net)
{
	xil_printf("\n[%s]\r\n",__func__);
    int i, j;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
//    xil_printf("\n[%s] 1 \r\n",__func__);

    if(l.xnor){
        binarize_weights(l.weights, l.n, l.c/l.groups*l.size*l.size, l.binary_weights);
        swap_binary(&l);
        binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
        net.input = l.binary_input;
    }

//    xil_printf("\n[%s] 2 \r\n",__func__);
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output + (i*l.groups + j)*n*m;
            float *im =  net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1) {
                b = im;
            } else {
                im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }

    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
    }

    activate_array(l.output, l.outputs*l.batch, l.activation);
    if(l.binary || l.xnor) swap_binary(&l);
}

void forward_convolutional_layer_q(layer l, network net)
{
	xil_printf("\n[%s]\r\n",__func__);

    int out_h = (l.h + 2 * l.pad - l.size) / l.stride + 1;    // output_height=input_height for stride=1 and pad=1
    int out_w = (l.w + 2 * l.pad - l.size) / l.stride + 1;    // output_width=input_width for stride=1 and pad=1
    int i, f, j;
    int const out_size = out_h*out_w;
    size_t const weights_size = l.size*l.size*l.c*l.n;

    // fill zero (ALPHA)
    //for (i = 0; i < l.outputs; ++i) l.output[i] = 0;

    // l.n - number of filters on this layer
    // l.c - channels of input-array
    // l.h - height of input-array
    // l.w - width of input-array
    // l.size - width and height of filters (the same size for all filters)


    //draw_distribution(l.weights, weights_size, "weights");
    //draw_distribution(state.input, l.inputs, "input");

    //typedef int32_t conv_t;    // l.output
    typedef int16_t conv_t;    // l.output
    conv_t *output_q = embed_calloc(l.outputs, sizeof(conv_t));


    net.input_int8 = (int8_t *)embed_calloc(l.inputs, sizeof(int8_t));
    int z;
    for (z = 0; z < l.inputs; ++z) {
        //int16_t src = lround(state.input[k] * net.layers[0].input_quant_multipler);
        int16_t src = net.input[z] * l.input_quant_multipler;
        net.input_int8[z] = max_abs(src, I_MAX_VAL);
    }

    ////////////////////////////////////
    // cudnnConvolutionBiasActivationForward()
    // y = act ( alpha1 * conv(x) + alpha2 * z + bias )
    // int8 = activation( float * conv(int8) + float * int8 + float )
    // int8 = activation( conv(input_int8) + bias_float ) // X_INT8x4 or X_INT8
    // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBiasActivationForward
    ///////////////////////////////////


    // 1. Convolution !!!
    int fil;

    // cuDNN: y = conv(x)
    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = out_h*out_w;
    int8_t *a = l.weights_int8;
    int8_t *b = (int8_t *)net.workspace;
    conv_t *c = output_q;    // int16_t

    // convolution as GEMM (as part of BLAS)
    //for (i = 0; i < l.batch; ++i) {
    im2col_cpu_int8(net.input_int8, l.c, l.h, l.w, l.size, l.stride, l.pad, b);    // here
    //gemm_nn_int8_int16(m, n, k, 1, a, k, b, n, c, n);    // single-thread gemm

    int t;    // multi-thread gemm
    for (t = 0; t < m; ++t) {
        gemm_nn_int8_int16(1, n, k, 1, a + t*k, k, b, n, c + t*n, n);
        //gemm_nn_int8_int16_conv16(1, n, k, 1, a + t*k, k, b, n, c + t*n, n);
        //gemm_nn_int8_int32(1, n, k, 1, a + t*k, k, b, n, c + t*n, n); // conv_t should be int32_t
    }
    //}

    embed_free(net.input_int8);

    float ALPHA1 = R_MULT / (l.input_quant_multipler * l.weights_quant_multipler);

    // cuDNN: y = alpha1 * conv(x)
    for (i = 0; i < l.outputs; ++i) {
        l.output[i] = output_q[i] * ALPHA1;    // cuDNN: alpha1
    }
    embed_free(output_q);

    // cuDNN: y = alpha1 * conv(x) + bias
    for (fil = 0; fil < l.n; ++fil) {
        for (j = 0; j < out_size; ++j) {
            l.output[fil*out_size + j] += l.biases[fil];
        }
    }

    //draw_distribution(l.output, l.outputs, "output");

    activate_array(l.output, l.outputs*l.batch, l.activation);

}

void update_convolutional_layer(convolutional_layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.nweights, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.nweights, momentum, l.weight_updates, 1);
}





