#include "darknet.h"
//#include "xtime_l.h"

static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};
#define COUNTS_PER_USECOND (866666687/(2*1000000))


extern char *coco_name_array[];


#define YOLO_LITE_MOBILENET

#if defined(YOLOV3_TINY_24_LAYER)
extern cfg_section yolov3_tiny_cfg[];
extern int yolov3_tiny_cfg_size;
#define CONFIGURE yolov3_tiny_cfg
#define CONFIGURE_SIZE yolov3_tiny_cfg_size
#elif defined(YOLOV3_TINY_14_LAYER)
extern cfg_section test_1_cfg[];
extern int test_1_cfg_size;
#define CONFIGURE test_1_cfg
#define CONFIGURE_SIZE test_1_cfg_size
#elif defined(YOLOV3_TINY_13_LAYER)
extern cfg_section test_2_cfg[];
extern int test_2_cfg_size;
#define CONFIGURE test_2_cfg
#define CONFIGURE_SIZE test_2_cfg_size
#elif defined(YOLOV3_TINY_12_LAYER)
extern cfg_section test_3_cfg[];
extern int test_3_cfg_size;
#define CONFIGURE test_3_cfg
#define CONFIGURE_SIZE test_3_cfg_size
#elif defined(YOLO_LITE)
extern cfg_section yolo_lite_cfg[];
extern int yolo_lite_cfg_size;
#define CONFIGURE yolo_lite_cfg
#define CONFIGURE_SIZE yolo_lite_cfg_size
#elif defined(YOLO_LITE_MOBILENET)
extern cfg_section yolo_lite_mobilenet_cfg[];
extern int yolo_lite_mobilenet_cfg_size;
#define CONFIGURE yolo_lite_mobilenet_cfg
#define CONFIGURE_SIZE yolo_lite_mobilenet_cfg_size
#endif






extern image dog_im;


unsigned int yolov3_tiny_weights_len = 35434956;
#define YOLOV3_TINY_WEIGHTS_LEN 35434956
#ifdef OPENSSD
unsigned char *yolov3_tiny_weights = RESERVED0_START_ADDR;
#else
unsigned char weights[YOLOV3_TINY_WEIGHTS_LEN];
unsigned char *yolov3_tiny_weights = weights ;
#endif


void test_detector(float thresh, float hier_thresh, int quantized)
{
//    char **names = (char **)coco_name_array;
    //image **alphabet = load_alphabet();
//    memset(yolov3_tiny_weights,0, yolov3_tiny_weights_len);

    network *net = embed_load_network(CONFIGURE, CONFIGURE_SIZE, 0, quantized);
	double start, end;

    set_batch_network(net, 1);
    xil_printf("\nset_batch_network\r\n",__func__);
//    srand(2222222);
//    XTime start,end;
//    char buff[256];
//    char *input = buff;
//    float nms=.45;
	image im = dog_im;

//	layer l = net->layers[net->n-1];
	//float *X = sized.data;
	float *X = im.data;
#ifdef OPENSSD
	XTime_GetTime(&start);
#else
    start = what_time_is_it_now();
#endif
	network_predict(net, X);
#ifdef OPENSSD
	XTime_GetTime(&end);
#else
    end = what_time_is_it_now();
	xil_printf("predicted in %f seconds.\r\n",(double)(end-start));
#endif

}

