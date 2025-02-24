#ifndef MVAU_THRESH5_H
#define MVAU_THRESH5_H

#include "../activations.hpp"

extern ThresholdsActivation<2,32,15,ap_int<12>,ap_uint<4>,0,comp::less_equal<ap_int<12>, ap_int<12>>> mvau_threshs5;

#endif
