#include "darknet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "xil_io.h"
#include "xil_cache.h"
#include "xil_mmu.h"
#include <stdio.h>
#include <stdlib.h>
#include "sleep.h"
#include "xparameters.h"
#include "xil_types.h"
#include "xil_assert.h"
#include "xil_io.h"
#include "xil_exception.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "xscugic.h"
#include "xdmaps.h"
#include "xil_mmu.h"
#include "xil_cache.h"
#include <time.h>

// static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};

//static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20};

#define COUNTS_PER_USECOND (866666687/(2*1000000))


extern char *coco_name_array[];


#define YOLO_EMBED	//define which network you want to use

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
#elif defined(YOLO_EMBED)
extern cfg_section YOLO_Embed_cfg[];
extern int YOLO_Embed_cfg_size;
#define CONFIGURE YOLO_Embed_cfg
#define CONFIGURE_SIZE YOLO_Embed_cfg_size
#endif


metadata meta;

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
	meta.classes = 20;
	int numboxes;
	int *pnum = &numboxes;
	int a,b,c=0;
	float results[100][20]={0};	//[number of boxes][classes]
	float temp[20]={0};
	int pred[20];


	detection *dets;

    //Test image
	image im = dog_im;
	float *X = im.data;

	//loading the network
    network *net = embed_load_network(CONFIGURE, CONFIGURE_SIZE, 0, quantized);

    //setting batch network to a single image
    set_batch_network(net, 1);


	//performing Inference
	network_predict(net, X);

	//getting the bounding boxes
	dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, NULL, 0, pnum); //added new

	//Making sure that the number of boxes doesn't go above 100
	if (numboxes>100)
		c=100;
	else
		c=numboxes;

	//getting the object probability from each box
	for (a=0; a<c; a++) {//number of boxes
		for (b=0; b<20; b++) { //number fo classes
			if(dets[a].prob[b] > 0 && dets[a].prob[b]<100) {
				results[a][b] = dets[a].prob[b];
			}
			else
				results[a][b] = 0;
		}
	}
	//taking the maximum value for each class from all the boxes
	for(a=0;a<20;a++){
		for(b=0;b<c;b++){
			if (temp[a] < results[b][a])
				temp[a] = results[b][a];
		}
	}
	//printing the prediction results
	for(a=0;a<20;a++){
		pred[a] = (uint)(temp[a]);
		printf("class[%d]: %d\n\r",a,pred[a]);
	}
}




