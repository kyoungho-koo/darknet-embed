#include "darknet.h"


static void *heap = HEAP_STAR_ADDR;

void *embed_calloc(int num_val, int size_val) 
{
	void *ret = heap;
	heap += num_val*size_val;
	if (heap >= RESERVED0_END_ADDR) {
		printf("out of memory\r\n", __func__);
		return NULL;
	}
	printf("total dynamic alloc %dKB\n", ((int)heap - HEAP_STAR_ADDR)>>10);
	return ret;
}

void embed_free (void *p) 
{
//	free(p);
}
