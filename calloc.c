#include "darknet.h"
#include "debug.h"


//#define ARM
#ifdef OPENSSD
static void *heap = HEAP_STAR_ADDR;
#define MAX_HEAP_ADDR RESERVED0_END_ADDR
#else
#define MAX_HEAP_ADDR (110*1024*1024)
static char heap_area[MAX_HEAP_ADDR];
static void *heap = heap_area;
#define HEAP_START_ADDR heap_area
#endif

void *embed_calloc(int num_val, int size_val) 
{
	void *ret = heap;
	heap += num_val*size_val;
#if defined(OPENSSD) || defined(ARM)
	if (heap >= MAX_HEAP_ADDR) {
#else
	if (heap <= MAX_HEAP_ADDR) {
#endif
		printf("out of memory\r\n", __func__);
		return NULL;
	}
	LOG("total dynamic alloc %dKB\n", ((int)heap - (int)HEAP_START_ADDR)>>10);
	return ret;
}

void embed_free (void *p) 
{
	heap = p;
//	free(p);
}
