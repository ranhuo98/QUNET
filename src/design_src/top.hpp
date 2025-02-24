//#ifndef TOP_HPP
//#define TOP_HPP
//
//#include "function_wrapper.hpp"
//
//#include "MVAU_params/MVAU_thresh0.h"
//
//#include "Thresholding_params/Thresholding_thresh0.h"
//
//
//void top(
////		ap_uint<32> *in,
////		ap_uint<32> *out,
////		ap_uint<8*32*4> *w_in,
//		hls::stream<ap_uint<2*Input_precision>> &in_0,
//		hls::stream<ap_uint<SIMD*Input_precision>> &out_0,
//		hls::stream<ap_uint<8*32*4>> &weight_stream_1,
//		hls::stream<ap_uint<8*32*4>> &weight_stream_2,
//		ap_uint<8> block,
//		ap_uint<4> IFMDim_arg,
//		ap_uint<4> OFMDim_arg,
//		ap_uint<8> IFMChannel_arg, // use actual value
//		ap_uint<8> MVAU_OFMChannel_arg, // maximum 32, use actual value
//		ap_uint<8> weight_in_simd_arg,
//		ap_uint<8> MVAU_Tiles_arg,
//		ap_uint<8> UpS_Tiles_arg,
//		ap_uint<8> OUPChannel_arg,
//		ap_uint<2> nf_compute, // = pe / 32
//		ap_uint<8> scale_factor_arg,
//		ap_uint<8> Padding_arg,
//		ap_uint<1> MaxPooling_en,
//		ap_uint<1> Upsampling_en,
//		ap_uint<2> buf_index
//);
//
//
//
//#endif

#ifndef TOP_HPP
#define TOP_HPP

#include "function_wrapper.hpp"

//#include "MVAU_params/MVAU_params0.h"
//#include "MVAU_params/MVAU_params1.h"
//#include "MVAU_params/MVAU_params2.h"
//#include "MVAU_params/MVAU_params3.h"
//#include "MVAU_params/MVAU_params4.h"
//#include "MVAU_params/MVAU_params5.h"
//#include "MVAU_params/MVAU_params6.h"
//#include "MVAU_params/MVAU_params7.h"
//#include "MVAU_params/MVAU_params8.h"
//#include "MVAU_params/MVAU_params9.h"
//#include "MVAU_params/MVAU_params10.h"
//#include "MVAU_params/MVAU_params11.h"
#include "MVAU_params/MVAU_params12.h"
#include "MVAU_params/MVAU_thresh0.h"
#include "MVAU_params/MVAU_thresh1.h"
#include "MVAU_params/MVAU_thresh2.h"
#include "MVAU_params/MVAU_thresh3.h"
#include "MVAU_params/MVAU_thresh4.h"
#include "MVAU_params/MVAU_thresh5.h"
#include "MVAU_params/MVAU_thresh6.h"
#include "MVAU_params/MVAU_thresh7.h"
#include "MVAU_params/MVAU_thresh8.h"
#include "MVAU_params/MVAU_thresh9.h"
#include "MVAU_params/MVAU_thresh10.h"
#include "MVAU_params/MVAU_thresh11.h"
#include "Thresholding_params/Thresholding_thresh0.h"
#include "Thresholding_params/Thresholding_thresh1.h"
#include "Thresholding_params/Thresholding_thresh2.h"
#include "Thresholding_params/Thresholding_thresh3.h"
#include "Thresholding_params/Thresholding_thresh4.h"
#include "Thresholding_params/Thresholding_thresh5.h"
#include "Thresholding_params/Thresholding_thresh6.h"

void top(
//		ap_uint<32> *in,
//		ap_uint<32> *out,
//		ap_uint<8*32*4> *w_in,
		hls::stream<ap_uint<2*Input_precision>> &in_0,
		hls::stream<ap_uint<32>> &out_0,
		hls::stream<ap_uint<8*32*4>> &weight_stream_1,
		hls::stream<ap_uint<8*32*4>> &weight_stream_2,
		ap_uint<8> block
//		ap_uint<4> IFMDim_arg,
//		ap_uint<4> OFMDim_arg,
//		ap_uint<8> IFMChannel_arg, // use actual value
//		ap_uint<8> MVAU_OFMChannel_arg, // maximum 32, use actual value
//		ap_uint<8> weight_in_simd_arg,
//		ap_uint<8> MVAU_Tiles_arg,
//		ap_uint<8> UpS_Tiles_arg,
//		ap_uint<8> OUPChannel_arg,
//		ap_uint<2> nf_compute, // = pe / 32
//		ap_uint<8> scale_factor_arg,
//		ap_uint<8> Padding_arg,
//		ap_uint<1> MaxPooling_en,
//		ap_uint<1> Upsampling_en,
//		ap_uint<2> buf_index
);



#endif



