#ifndef MVAU_THRESH3_H
#define MVAU_THRESH3_H

#include "../activations.hpp"

extern ThresholdsActivation<1,32,15,ap_int<12>,ap_uint<4>,0,comp::less_equal<ap_int<12>, ap_int<12>>> mvau_threshs3;

#endif
