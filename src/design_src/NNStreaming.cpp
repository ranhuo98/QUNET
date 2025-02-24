#include "top.hpp"

void top(
		hls::stream<ap_uint<2*Input_precision>> &in_0,
		hls::stream<ap_uint<32>> &out_0,
		hls::stream<ap_uint<8*32*4>> &weight_stream_1,
		hls::stream<ap_uint<8*32*4>> &weight_stream_2,
		ap_uint<8> block
)
{

#pragma HLS INTERFACE axis port=in_0
#pragma HLS INTERFACE axis port=out_0
#pragma HLS INTERFACE axis port=weight_stream_1
#pragma HLS INTERFACE axis port=weight_stream_2
#pragma HLS INTERFACE s_axilite port=block bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control


#pragma HLS ARRAY_PARTITION variable=thresB_threshs0.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=thresB_threshs0.m_thresholds complete dim=3


#pragma HLS ARRAY_PARTITION variable=thresB_threshs1.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=thresB_threshs1.m_thresholds complete dim=3

#pragma HLS ARRAY_PARTITION variable=thresB_threshs2.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=thresB_threshs2.m_thresholds complete dim=3

#pragma HLS ARRAY_PARTITION variable=thresB_threshs3.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=thresB_threshs3.m_thresholds complete dim=3

#pragma HLS ARRAY_PARTITION variable=thresB_threshs4.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=thresB_threshs4.m_thresholds complete dim=3

#pragma HLS ARRAY_PARTITION variable=thresB_threshs5.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=thresB_threshs5.m_thresholds complete dim=3

#pragma HLS ARRAY_PARTITION variable=thresB_threshs6.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=thresB_threshs6.m_thresholds complete dim=3

// params passing in to the layer function, before which they should get values from pad function
	FixedPointWeights<SIMD,ap_int<Input_precision>,PE,TILES> mvau_weights_tile;
	ThresholdsActivation<TMEM,PE,NUM_THRES,ap_int<12>,ap_uint<Input_precision>,0,comp::less_equal<ap_int<12>, ap_int<12>>> mvau_threshs;


#pragma HLS ARRAY_PARTITION variable=mvau_weights_tile.m_weights complete dim=1
#pragma HLS bind_storage variable=mvau_weights_tile.m_weights type=RAM_2P impl=bram
#pragma HLS ARRAY_PARTITION variable=mvau_threshs.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=mvau_threshs.m_thresholds complete dim=3


	ap_uint<4> IFMDim_arg;
	ap_uint<4> OFMDim_arg;
	ap_uint<8> IFMChannel_arg;
	ap_uint<8> MVAU_OFMChannel_arg;
	ap_uint<8> weight_in_simd_arg;
	ap_uint<8> MVAU_Tiles_arg;
	ap_uint<8> UpS_Tiles_arg;
	ap_uint<8> OUPChannel_arg;
	ap_uint<2> nf_compute;
	ap_uint<8> scale_factor_arg;
	ap_uint<8> Padding_arg;
	ap_uint<1> MaxPooling_en;
	ap_uint<1> Upsampling_en;
	ap_uint<2> buf_index;

	hls::stream<ap_uint<SIMD*Input_precision>> inter_1("inter_1");
#pragma HLS STREAM variable=inter_1 depth=64 type=fifo
	hls::stream<ap_uint<SIMD*Input_precision>> inter_2("inter_2");
#pragma HLS STREAM variable=inter_2 depth=324 type=fifo // 324
	hls::stream<ap_uint<PE*Input_precision>> inter_3("inter_3");
#pragma HLS STREAM variable=inter_3 depth=72 type=fifo // 72
	hls::stream<ap_uint<SIMD*Input_precision>> inter_4("inter_4");
#pragma HLS STREAM variable=inter_4 depth=36 type=fifo // 36
	hls::stream<ap_uint<SIMD*Input_precision>> inter_5("inter_5");
#pragma HLS STREAM variable=inter_5 depth=36 type=fifo // 36
	hls::stream<ap_uint<PE*Input_precision>> inter_7("inter_7");
#pragma HLS STREAM variable=inter_7 depth=18 type=fifo // 18
	hls::stream<ap_uint<SIMD*Input_precision>> inter_7_simd("inter_7_simd");
#pragma HLS STREAM variable=inter_7_simd depth=9 type=fifo // 9
	hls::stream<ap_uint<SIMD*Input_precision>> inter_8("inter_8");
#pragma HLS STREAM variable=inter_8 depth=36 type=fifo // 36
	hls::stream<ap_uint<SIMD*Input_precision>> inter_9("inter_9");
#pragma HLS STREAM variable=inter_9 depth=36 type=fifo // 36
	hls::stream<ap_uint<Thresholding_PE*Input_precision>> inter_10("inter_10");
#pragma HLS STREAM variable=inter_10 depth=36 type=fifo // 36
	hls::stream<ap_uint<Thresholding_PE*Input_precision>> inter_11("inter_11");
#pragma HLS STREAM variable=inter_11 depth=36 type=fifo // 36
	hls::stream<ap_uint<SIMD*(Input_precision+1)>> inter_12("inter_12");
#pragma HLS STREAM variable=inter_12 depth=36 type=fifo // 36
	hls::stream<ap_uint<SIMD*Input_precision>> from_buf("from_buf");
#pragma HLS STREAM variable=from_buf depth=36 type=fifo // 36
	hls::stream<ap_uint<SIMD*Input_precision>> to_buf("to_buf");
#pragma HLS STREAM variable=to_buf depth=36 type=fifo // 36
	hls::stream<ap_uint<SIMD*Input_precision>> to_buf_1("to_buf_1");
#pragma HLS STREAM variable=to_buf_1 depth=36 type=fifo // 36
	hls::stream<ap_uint<Input_precision*32>> inter_13("inter_13");
#pragma HLS STREAM variable=inter_13 depth=36 type=fifo // 36


// Buffer BRAM
	static ap_uint<SIMD*Input_precision> buf_concat[2][MAX_BUFSIZE];
#pragma HLS ARRAY_PARTITION variable=buf_concat type=complete dim=2

	ap_uint<SIMD*Input_precision> buf_reload[MAX_BUFSIZE];
#pragma HLS ARRAY_PARTITION variable=buf_reload type=complete dim=1


	switch(block) {
		case 1: 
			pad_mvau_thresh(mvau_threshs0, mvau_threshs);
			break;
		case 2: 
			pad_mvau_thresh(mvau_threshs1, mvau_threshs);
			break;
		case 3: 
			pad_mvau_thresh(mvau_threshs2, mvau_threshs);
			break;
		case 4: 
			pad_mvau_thresh(mvau_threshs3, mvau_threshs);
			break;
		case 5: 
			pad_mvau_thresh(mvau_threshs4, mvau_threshs);
			break;
		case 6: 
			pad_mvau_thresh(mvau_threshs5, mvau_threshs);
			break;
		case 7: 
			pad_mvau_thresh(mvau_threshs7, mvau_threshs);
			break;
		case 8: 
			pad_mvau_thresh(mvau_threshs8, mvau_threshs);
			break;
		case 9: 
			pad_mvau_thresh(mvau_threshs10, mvau_threshs);
			break;
		case 10: 
			pad_mvau_thresh(mvau_threshs11, mvau_threshs);
			break;
	}

	configure_top_parameters(block, IFMDim_arg, OFMDim_arg, IFMChannel_arg,
	                         MVAU_OFMChannel_arg, weight_in_simd_arg,
	                         MVAU_Tiles_arg, UpS_Tiles_arg, OUPChannel_arg,
	                         nf_compute, scale_factor_arg, Padding_arg,
	                         MaxPooling_en, Upsampling_en, buf_index);
	pad_weights_Stream_to_FixedPointWeights<ap_int<4>, 18>(weight_stream_1, mvau_weights_tile, weight_in_simd_arg, 32, MVAU_Tiles_arg);

	if(block == 1)
	{
		Thresholding_Batch_initial(in_0, from_buf, thresB_threshs0);
	}
	else
	{
		Read_Buf_reloadToStream(buf_reload, from_buf, IFMDim_arg*IFMDim_arg);
	}

	// from idma or the data from last round in BRAM
	FMPadding_Batch_edge_6to8(from_buf, inter_1, IFMDim_arg+2, 1, 1, 1, 1);
	ConvolutionInputGenerator_3by3(inter_1, inter_2, 3, 1, IFMDim_arg+2, IFMDim_arg);
	if(block == 7)
		MatrixVectorActivation_9by64_block7(inter_2, inter_3, mvau_weights_tile, mvau_threshs, IFMChannel_arg, MVAU_OFMChannel_arg, IFMDim_arg*IFMDim_arg, nf_compute, 9);
	else if(block == 9)
		MatrixVectorActivation_9by64_block9(inter_2, inter_3, mvau_weights_tile, mvau_threshs, IFMChannel_arg, MVAU_OFMChannel_arg, IFMDim_arg*IFMDim_arg, nf_compute, 9);
	else
		MatrixVectorActivation_9by64(inter_2, inter_3, mvau_weights_tile, mvau_threshs, IFMChannel_arg, MVAU_OFMChannel_arg, IFMDim_arg*IFMDim_arg, nf_compute, 9);
	DataWidthConverter_PEtoSIMD(inter_3, to_buf, IFMDim_arg*IFMDim_arg*nf_compute, nf_compute);
	if(MaxPooling_en)
	{
		Write_StreamToBuf_concat(buf_index, to_buf, buf_concat, inter_4, IFMDim_arg*IFMDim_arg);
		ConvolutionInputGenerator_for_maxpool(inter_4, inter_5, 2, 2, IFMDim_arg, OFMDim_arg);
		Pool_Batch_1in4(inter_5, to_buf_1, OFMDim_arg*OFMDim_arg);
	}
	else if(Upsampling_en)
	{
		if(block == 6)
		{
			pad_mvau_thresh(mvau_threshs6, mvau_threshs);
		}
		else if(block == 8)
		{
			pad_mvau_thresh(mvau_threshs9, mvau_threshs);
		}
		pad_weights_Stream_to_FixedPointWeights<ap_int<4>, 1>(weight_stream_2, mvau_weights_tile, weight_in_simd_arg, 32, UpS_Tiles_arg);
		MatrixVectorActivation_9by64(to_buf, inter_7, mvau_weights_tile, mvau_threshs, IFMChannel_arg, OUPChannel_arg, IFMDim_arg*IFMDim_arg, 1, 1);
		DataWidthConverter_PEtoSIMD(inter_7, inter_7_simd, IFMDim_arg*IFMDim_arg*1, 1);
		UpsampleNearestNeighbour_Batch_3to6(inter_7_simd, inter_8, scale_factor_arg, Padding_arg, IFMDim_arg, OFMDim_arg); //inter_8 to add
		Read_Buf_concatToStream(buf_index, buf_concat, inter_9, OFMDim_arg*OFMDim_arg); // infer_9 to add
		if(OUPChannel_arg == 32){
			Thresholding_Batch_preAdd_2(inter_9, inter_11, thresB_threshs2);
			Thresholding_Batch_preAdd_3(inter_8, inter_10, thresB_threshs3);
			AddStreams_Batch_01(inter_10, inter_11, inter_12, 9, 32);
			Thresholding_Batch_postAdd_4(inter_12, to_buf_1, thresB_threshs4); // "out" save to BRAM in preparation for the next layer or to odma
		}
		else if(OUPChannel_arg == 16){
			Thresholding_Batch_preAdd_1(inter_9, inter_11, thresB_threshs1);
			Thresholding_Batch_preAdd_5(inter_8, inter_10, thresB_threshs5);
			AddStreams_Batch_01(inter_10, inter_11, inter_12, 36, 16);
			Thresholding_Batch_postAdd_6(inter_12, to_buf_1, thresB_threshs6); // "out" save to BRAM in preparation for the next layer or to odma
		}

	}
	if(block == 10) // to odma
	{
		MatrixVectorActivation_out(to_buf, inter_13, mvau_weights12, 8, 4, 36, 1, 1);
		LabelSelect(inter_13, out_0);
	}
	else // to BRAM
	{
		// assign DoubleConvOut to out interface
		if(MaxPooling_en || Upsampling_en){
			Write_StreamToBuf_reload(to_buf_1, buf_reload, OFMDim_arg*OFMDim_arg);
		}
		else
		{
			Write_StreamToBuf_reload(to_buf, buf_reload, OFMDim_arg*OFMDim_arg);
		}

	}

}



