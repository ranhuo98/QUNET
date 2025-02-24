#ifndef MVAU_THRESH9_H
#define MVAU_THRESH9_H

#include "../activations.hpp"

extern ThresholdsActivation<1,16,15,ap_int<12>,ap_uint<4>,0,comp::less_equal<ap_int<12>, ap_int<12>>> mvau_threshs9;

#endif
