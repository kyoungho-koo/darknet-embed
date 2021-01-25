#include "math.h"
#include <math.h>



int max_abs(int src, int max_val)
{
    if (abs(src) > abs(max_val)) src = (src > 0) ? max_val : -max_val;
    return src;
}


fixed sqrtF2F (fixed x)
{
	int32_t t, q, b, r;
	r = x;
	b = 0x40000000;
	q = 0;

    while (b > 0x40)
    {
		t = q + b;
		if (r >= t)
		{
			r -= t;
			q = t + b; // equivalent to q += 2*b
		}
		r <<= 1;
		b >>= 1;
	}
    q >>= 8;
    return q;
}
