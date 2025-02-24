#ifndef MVAU_THRESH10_H
#define MVAU_THRESH10_H

#include "../activations.hpp"

extern ThresholdsActivation<1,8,15,ap_int<12>,ap_uint<4>,0,comp::less_equal<ap_int<12>, ap_int<12>>> mvau_threshs10;

#endif
