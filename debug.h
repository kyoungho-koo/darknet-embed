
#include <stdio.h>

#ifdef OPENSSD
#include "xil_printf.h"

#define printf(fmt, ...)	xil_printf(fmt, ##__VA_ARGS__)

#if defined(DEBUG)
#define LOG_ARG(fmt, ...)	xil_printf("[%s] " #__VA_ARGS__ "  (" fmt ") \r\n", __func__, ##__VA_ARGS__)
#define LOG(fmt, ...)	xil_printf("[%s] " fmt "\r\n", __func__, ##__VA_ARGS__)
#else
#define LOG_ARG(fmt, ...)	
#define LOG(fmt, ...)	
#endif


#else

#define xil_printf(fmt, ...)	printf(fmt, ##__VA_ARGS__)

#if defined(DEBUG)
#define LOG_ARG(fmt, ...)	printf("[%s] " #__VA_ARGS__ "  (" fmt ") \n", __func__, ##__VA_ARGS__)
#define LOG(fmt, ...)	printf("[%s] " fmt "\n", __func__, ##__VA_ARGS__)
#else
#define LOG_ARG(fmt, ...)	
#define LOG(fmt, ...)	
#endif

#endif
