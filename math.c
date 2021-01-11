#include "math.h"
#include <math.h>



int max_abs(int src, int max_val)
{
    if (abs(src) > abs(max_val)) src = (src > 0) ? max_val : -max_val;
    return src;
}
