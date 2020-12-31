#include "darknet.h"

//#include <time.h>
#include <stdlib.h>
#include <stdio.h>

extern void test_detector(float thresh, float hier_thresh);


void darknet_main()
{
	float thresh = 0.5f;
	xil_printf("thresh %f\n",thresh);
	test_detector(thresh, .5);
}

