#ifndef MVAU_THRESH6_H
#define MVAU_THRESH6_H

#include "../activations.hpp"

extern ThresholdsActivation<1,32,15,ap_int<12>,ap_uint<4>,0,comp::less_equal<ap_int<12>, ap_int<12>>> mvau_threshs6;

#endif
