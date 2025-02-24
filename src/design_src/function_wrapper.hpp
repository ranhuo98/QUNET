#ifndef FUNCTION_WRAPPER_HPP
#define FUNCTION_WRAPPER_HPP

#include "function_test.hpp"

void configure_top_parameters(
	ap_uint<8> block,
    ap_uint<4> &IFMDim_arg,
    ap_uint<4> &OFMDim_arg,
    ap_uint<8> &IFMChannel_arg,
    ap_uint<8> &MVAU_OFMChannel_arg,
    ap_uint<8> &weight_in_simd_arg,
    ap_uint<8> &MVAU_Tiles_arg,
    ap_uint<8> &UpS_Tiles_arg,
    ap_uint<8> &OUPChannel_arg,
    ap_uint<2> &nf_compute,
    ap_uint<8> &scale_factor_arg,
    ap_uint<8> &Padding_arg,
    ap_uint<1> &MaxPooling_en,
    ap_uint<1> &Upsampling_en,
    ap_uint<2> &buf_index
);

/**
 * \brief   FM Padding - Padds the input of multiple frames with zeroes
 *          for when the sliding window is centered on border pixels
 *
 * Used to add padding with zeroes to multiple inputs in case the sliding window is
 * centered on border pixels
 *
 * \tparam	ImgDim			<ignored>
 * \tparam	OutputDim		Size of the output feature map
 * \tparam	PaddingBefore	Top / left padding
 * \tparam	PaddingBehind	Bottom / right padding
 * \tparam	NumChannels		Number of channels of the input feature map
 * \tparam	SIMD			Input parallelism
 * \tparam	In_t			Input datatype
 *
 * \param	in	Input stream
 * \param	out	Output stream
 * \param	numReps	Number of frames / images
 */
#define ImgDim 6
#define OutputDim ImgDim+2
#define PaddingBefore (OutputDim-ImgDim)/2
#define PaddingBehind (OutputDim-ImgDim)/2
#define Input_NumChannels 64
#define Input_precision 4
#define SIMD Input_NumChannels
#define Padding_numReps 1

void FMPadding_Batch_edge_6to8(hls::stream<ap_uint<SIMD*Input_precision> > &in0_V,
		hls::stream<ap_uint<SIMD*Input_precision> > &out_V,
		ap_uint<4> outputdim,
		ap_uint<3> paddingleft,
		ap_uint<3> paddingright,
		ap_uint<3> paddingtop,
		ap_uint<3> paddingbottom);



/**
 * \brief Sliding Window unit that produces output vectors for feeding
 * a Matrix_Vector_Activate_Batch, implementing the im2col algorithm. To be used only if
 * ConvKernelDim%Stride = 0
 *
 * For instance, this function will convert a 3 * 3 kernel
 *
 * [0 0 0]
 * [0 1 1]
 * [0 1 1] from a n * n FM to [0 0 0 0 1 1 0 1 1]
 *
 * 	So a n * n input feature map will be divided to (n-2)*(n-2) vectors of size 1 * 9 each,
 * 	which will be fed to Matrix_Vector_Activate_Batch
 */
#define ConvKernelDim 3
#define IFMChannels Input_NumChannels
#define IFMDim OutputDim
#define OFMDim IFMDim-2
#define Stride 1
#define Conv_numReps 1

void ConvolutionInputGenerator_3by3(hls::stream<ap_uint<SIMD*Input_precision>> &in0_V,
                hls::stream<ap_uint<SIMD*Input_precision>> &out_V,
				ap_uint<4> convkerneldim,
				ap_uint<4> stride,
				ap_uint<4> ifmdim,
				ap_uint<4> ofmdim);



/**
 * \brief Sliding Window unit that produces output vectors for feeding
 * a Matrix_Vector_Activate_Batch, implementing the im2col algorithm. To be used only if
 * ConvKernelDim%Stride = 0
 *
 * For instance, this function will convert a 3 * 3 kernel
 *
 * [0 0 0]
 * [0 1 1]
 * [0 1 1] from a n * n FM to [0 0 0 0 1 1 0 1 1]
 *
 * 	So a n * n input feature map will be divided to (n-2)*(n-2) vectors of size 1 * 9 each,
 * 	which will be fed to Matrix_Vector_Activate_Batch
 */
#define ConvKernelDim_mp 2
#define Stride_mp 2

void ConvolutionInputGenerator_for_maxpool(hls::stream<ap_uint<SIMD*Input_precision>> &in0_V,
                hls::stream<ap_uint<SIMD*Input_precision>> &out_V,
				ap_uint<4> convkerneldim,
				ap_uint<4> stride,
				ap_uint<4> ifmdim,
				ap_uint<4> ofmdim);



/**
 * Padding function for weights in Matrix_Vector_Activate_Batch
 */
#define WS_SIMD 8
#define WS_PE 32

template<typename in_WT, unsigned max_TILES,
		 unsigned out_SIMD, typename out_WT, unsigned out_PE, unsigned out_TILES>
void pad_weights_Stream_to_FixedPointWeights(
			hls::stream<ap_uint<WS_PE*WS_SIMD*in_WT::width>> &weight,
			FixedPointWeights<out_SIMD,out_WT,out_PE,out_TILES>& out,
			int in_simd,
			int in_pe, // always 32
			int in_tiles
		)
{
	const int fold_simd_compute = in_simd/WS_SIMD;
	const int fold_pe_compute = in_pe/WS_PE;
	const int fold_simd = out_SIMD/WS_SIMD;
	const int fold_pe = out_PE/WS_PE;
	const int fold_factor_compute = fold_simd_compute * fold_pe_compute;
	const int fold_factor = fold_simd * fold_pe;
	const int num_total = fold_factor_compute * in_tiles;
	const int NUM_TOTAL = fold_factor * out_TILES;

	unsigned  nf   = 0;
	unsigned  sf   = 0;
	unsigned  tile = 0; // invariant: tile = nf*SF + sf

	for (unsigned i = 0; i < NUM_TOTAL; i++) {
			ap_uint<WS_PE * WS_SIMD * in_WT::width> W_packed = 0;
			if((sf < fold_simd_compute) && (tile < in_tiles) && (nf < fold_pe_compute)){
				W_packed = weight.read();
			}
			for (unsigned pe = 0; pe < WS_PE; pe++) {
#pragma HLS UNROLL
				out.m_weights[WS_PE * nf + pe][tile] = out.m_weights[WS_PE * nf + pe][tile] >> WS_SIMD*in_WT::width;
				out.m_weights[WS_PE * nf + pe][tile](out_SIMD * in_WT::width - 1, (out_SIMD - WS_SIMD) * in_WT::width) = W_packed(WS_SIMD * in_WT::width - 1, 0);
				W_packed = W_packed >> WS_SIMD*in_WT::width;
			}
			sf++;
			if(sf == fold_simd){
				tile++;
				sf = 0;
			}

			if(tile == out_TILES){
				nf++;
				tile = 0;
			}
		}
}



/**
 * Padding function for weights in Matrix_Vector_Activate_Batch
 */
template<unsigned in_SIMD, typename in_WT, unsigned in_PE, unsigned in_TILES,
		 unsigned out_SIMD, typename out_WT, unsigned out_PE, unsigned out_TILES>
void pad_weights(
			const FixedPointWeights<in_SIMD,in_WT,in_PE,in_TILES>& in,
			FixedPointWeights<out_SIMD,out_WT,out_PE,out_TILES>& out
		)
{
	for (unsigned pe = 0; pe < in_PE; pe++) {
#pragma HLS UNROLL
		for (unsigned tile = 0; tile < in_TILES; tile++) {
			ap_uint<in_SIMD*in_WT::width> local_temp = in.m_weights[pe][tile];
			for (unsigned simd = 0; simd < in_SIMD; simd++) {
				ap_int<in_WT::width> weight = local_temp(in_SIMD*in_WT::width - 1, (in_SIMD-1)*in_WT::width);
				local_temp = local_temp << in_WT::width;
				out.m_weights[pe][tile] = out.m_weights[pe][tile] << in_WT::width;
				out.m_weights[pe][tile](in_WT::width - 1, 0) = weight;
			}
		}
	}

}



/**
 * Padding function for thresholds in Matrix_Vector_Activate_Batch
 */
#define NUM_THRES (1<<Input_precision) - 1 // used in Activations
template<unsigned in_TMEM, unsigned in_PE,
		 unsigned out_TMEM, unsigned out_PE>
void pad_mvau_thresh(
			ThresholdsActivation<in_TMEM,in_PE,NUM_THRES,ap_int<12>,ap_uint<Input_precision>,0,comp::less_equal<ap_int<12>, ap_int<12>>>& in,
			ThresholdsActivation<out_TMEM,out_PE,NUM_THRES,ap_int<12>,ap_uint<Input_precision>,0,comp::less_equal<ap_int<12>, ap_int<12>>>& out
		)
{
	for (unsigned pe = 0; pe < in_PE; pe++) {
#pragma HLS UNROLL
		for (unsigned nf = 0; nf < in_TMEM; nf++) {
			for (unsigned numth = 0; numth < NUM_THRES; numth++) {
#pragma HLS UNROLL
				ap_int<12> temp = in.m_thresholds[pe][nf][numth];
				out.m_thresholds[pe][nf][numth] = temp;
			}
		}
	}
}



/**
 * Padding function for thresholds in Thresholding_Batch
 */
template<unsigned in_TMEM, unsigned in_PE, typename in_TA,
		 unsigned out_TMEM, unsigned out_PE>
void pad_thresB_thresh(
			ThresholdsActivation<in_TMEM,in_PE,NUM_THRES,in_TA,ap_int<Input_precision>,-8,comp::less_equal<in_TA, in_TA>>& in,
			ThresholdsActivation<out_TMEM,out_PE,NUM_THRES,ap_int<10>,ap_int<Input_precision>,-8,comp::less_equal<ap_int<10>, ap_int<10>>>& out
		)
{
	for (unsigned pe = 0; pe < in_PE; pe++) {
		 for (unsigned nf = 0; nf < in_TMEM; nf++) {
			 for (unsigned numth = 0; numth < NUM_THRES; numth++) {
#pragma HLS PIPELINE II=1
				 in_TA temp = in.m_thresholds[pe][nf][numth];
				 out.m_thresholds[pe][nf][numth] = temp;
			 }
		}
	}
}



/**
 * \brief Matrix vector activate function
 *
 * The function performs the multiplication between a weigth matrix and the input activation vector,
 * accumulating the results and then applying an activation function on the accumulated result.
 *
 *
 * \tparam MatrixW    Width of the input matrix
 * \tparam MatrixH    Heigth of the input matrix
 * \tparam SIMD       Number of input columns computed in parallel
 * \tparam PE         Number of output rows computed in parallel
 * \tparam MMV        Number of output pixels computed in parallel
 * \tparam TSrcI      DataType of the input activation (as used in the MAC)
 * \tparam TDstI      DataType of the output activation (as generated by the activation)
 * \tparam TWeightI   DataType of the weights and how to access them in the array
 * \tparam TI         DataType of the input stream - safely deducible from the paramaters
 * \tparam TO         DataType of the output stream - safely deducible from the paramaters
 * \tparam TW         DataType of the weights matrix - safely deducible from the paramaters
 * \tparam TA         DataType of the activation class (e.g. thresholds) - safely deducible from the paramaters
 * \tparam R          Datatype for the resource used for FPGA implementation of the MAC  - safely deducible from the paramaters
 *
 * \param in          Input stream
 * \param out         Output stream
 * \param weights     Weights matrix (currently supports BinaryWeights or FixedPointWeights)
 * \param activation  Activation class
 * \param reps        Number of time the function has to be repeatedly executed (e.g. number of images)
 * \param r           Resource type for the hardware implementation of the MAC block
 *
 * \output dimension: MVA_NUMREPS * TMEM
 */
#define Output_NumChannels 64 // conv output channels
#define MW Input_NumChannels*ConvKernelDim*ConvKernelDim
#define MH Output_NumChannels
#define MVA_SIMD SIMD
#define PE 32
#define TILES (MW/MVA_SIMD)*(MH/PE) // TOTAL_FOLD
#define WMEM TILES // TOTAL_FOLD
#define TMEM MH/PE // NF
#define MVA_NUMREPS ImgDim*ImgDim // related to the number of vectors that the input FM is divided into

void MatrixVectorActivation_9by64(hls::stream<ap_uint<SIMD*Input_precision>> &in0_V,
                    hls::stream<ap_uint<PE*Input_precision>> &out_V,
					FixedPointWeights<SIMD,ap_int<Input_precision>,PE,TILES> &weights,
					ThresholdsActivation<TMEM,PE,NUM_THRES,ap_int<12>,ap_uint<Input_precision>,0,comp::less_equal<ap_int<12>, ap_int<12>>> &threshs,
					ap_uint<8> const simd_compute,
				    ap_uint<8> const pe_compute,
				    int const  reps_compute,
					unsigned const nf_compute,
					unsigned const sf_compute
				);



void MatrixVectorActivation_9by64_block7(hls::stream<ap_uint<SIMD*Input_precision>> &in0_V,
                    hls::stream<ap_uint<PE*Input_precision>> &out_V,
					FixedPointWeights<SIMD,ap_int<Input_precision>,PE,TILES> &weights,
					ThresholdsActivation<TMEM,PE,NUM_THRES,ap_int<12>,ap_uint<Input_precision>,0,comp::less_equal<ap_int<12>, ap_int<12>>> &threshs,
					ap_uint<8> const simd_compute,
				    ap_uint<8> const pe_compute,
				    int const  reps_compute,
					unsigned const nf_compute,
					unsigned const sf_compute);



void MatrixVectorActivation_9by64_block9(hls::stream<ap_uint<SIMD*Input_precision>> &in0_V,
                    hls::stream<ap_uint<PE*Input_precision>> &out_V,
					FixedPointWeights<SIMD,ap_int<Input_precision>,PE,TILES> &weights,
					ThresholdsActivation<TMEM,PE,NUM_THRES,ap_int<12>,ap_uint<Input_precision>,0,comp::less_equal<ap_int<12>, ap_int<12>>> &threshs,
					ap_uint<8> const simd_compute,
				    ap_uint<8> const pe_compute,
				    int const  reps_compute,
					unsigned const nf_compute,
					unsigned const sf_compute);



#define ConvKernalDim_preUpS 1
#define MW_preUpS Input_NumChannels*ConvKernalDim_preUpS*ConvKernalDim_preUpS

void MatrixVectorActivation_1by64(hls::stream<ap_uint<SIMD*Input_precision>> &in0_V,
                    hls::stream<ap_uint<PE*32>> &out_V,
					FixedPointWeights<SIMD,ap_int<Input_precision>,PE,1> &weights,
					ap_uint<8> const simd_compute,
				    ap_uint<8> const pe_compute,
				    int const  reps_compute,
					unsigned const nf_compute,
					unsigned const sf_compute);



void MatrixVectorActivation_out(hls::stream<ap_uint<SIMD*Input_precision>> &in0_V,
                    hls::stream<ap_uint<4*32>> &out_V,
					const FixedPointWeights<8,ap_int<Input_precision>,4,1> &weights,
					ap_uint<8> const simd_compute,
				    ap_uint<8> const pe_compute,
				    int const  reps_compute,
					unsigned const nf_compute,
					unsigned const sf_compute);



#define InWidth PE*Input_precision
#define OutWidth SIMD*Input_precision
#define NumInWords MVA_NUMREPS * TMEM // total InWidth-bit data that need processing
#define numReps_dwc 1

void DataWidthConverter_PEtoSIMD(
		hls::stream<ap_uint<PE*Input_precision> > &in0_V,
		hls::stream<ap_uint<SIMD*Input_precision> > &out_V,
		unsigned int numinwords,
		unsigned int nf_compute);



/**
 * \brief Pool_batch function
 *
 * The function performs a generic pool function (defined in pool.hpp) and works in conjuction
 * with a sliding window unit performing im2col on the input data, allowing
 * generic kernel and stride values
 *
 * \tparam Channels   Number of channels in the pool layer
 * \tparam PE         Number of channels in the pool layer computed in parallel
 * \tparam TotalK     Total kernel size of pooling (e.g. 3x3=9)
 * \tparam TSrcI      DataType of the input value (Slice)
 * \tparam TDstI      DataType of the output value (Slice)
 * \tparam TI         DataType of the input stream - safely deducible from the paramaters
 * \tparam TO         DataType of the output stream - safely deducible from the paramaters
 * \tparam TA         DataType of the function class (e.g. Max, Avg, Sum) - safely deducible from the paramaters
 *
 * \param in          Input stream
 * \param out         Output stream
 * \param function    Function class in the pool (Max, Avg, Sum)
 * \param reps        Number of time the function has to be repeatedly executed (e.g. number of images)
 *
 * This function will select the max value in each channel
 * For instance, input is 0x0c 0x0d 0x12 0x13, the max value of first channel is 0xd, the max value of second channel is 0x1,
 * then put two channels together, the output is 0x1d
 */
#define POOL_REPS 9

void Pool_Batch_1in4(hls::stream<ap_uint<SIMD*Input_precision> > &in0_V,
			hls::stream<ap_uint<SIMD*Input_precision> > &out_V,
			int const  reps_compute);


/**
 * \brief Upsampling with the Nearest Neighbour algorithm. Works with square feature maps on multiple images
 *
 * \tparam 	OFMDim 		Size of the output feature map
 * \tparam 	IFMDim 		Size of the input feature map
 * \tparam 	NumChannels 	Amount of channels of the input feature map
 * \tparam 	In_t		 	Input datatype
 *
 * \param 	in 			Input stream
 * \param 	out 			Output stream
 * \param     numReps      Number of time the function has to be repeatedly executed (e.g. number of images)
 */
#define IFMDim_UpS_3to6 3
#define OFMDim_UpS_3to6 6
#define numReps 1

void UpsampleNearestNeighbour_Batch_3to6(hls::stream<ap_uint<SIMD*Input_precision> > &in0_V,
			hls::stream<ap_uint<SIMD*Input_precision> > &out_V,
			ap_uint<4> scale_factor_argu,
			ap_uint<4> Padding_argu,
			ap_uint<4> ifmdim,
			ap_uint<4> ofmdim);


/**
 * \brief Stream to Buf
 */
#define MAX_BUFSIZE (ImgDim * ImgDim)

void Write_StreamToBuf_concat(
			int buf_index,
			hls::stream<ap_uint<SIMD*Input_precision> > &in0_V,
			ap_uint<SIMD*Input_precision> (&buf)[2][MAX_BUFSIZE],
			hls::stream<ap_uint<SIMD*Input_precision> > &out_V,
			int const data_size
		);


void Read_Buf_concatToStream(
			int buf_index,
			ap_uint<SIMD*Input_precision> (&buf)[2][MAX_BUFSIZE],
			hls::stream<ap_uint<SIMD*Input_precision> > &out_V,
			int const data_size
		);


void Write_StreamToBuf_reload(
			hls::stream<ap_uint<SIMD*Input_precision> > &in0_V,
			ap_uint<SIMD*Input_precision> (&buf)[MAX_BUFSIZE],
			int const data_size
		);


void Read_Buf_reloadToStream(
			ap_uint<SIMD*Input_precision> (&buf)[MAX_BUFSIZE],
			hls::stream<ap_uint<SIMD*Input_precision> > &out_V,
			int const data_size
		);


/*!
 * \brief Thresholding function for multiple images
 *
 * The function performs thresholds comparison with input activation vector,
 * and generating output based on the comparison results
 *
 * \tparam ImgDim         Total spatial size of input feature map
 * \tparam NumChannels    Number of channels in input feature map
 * \tparam PE             Number of output rows computed in parallel
 * \tparam TSrcI          DataType of the input activation (as used in the MAC)
 * \tparam TDstI          DataType of the output activation (as generated by the activation)
 * \tparam TI             DataType of the input stream - safely deducible from the paramaters
 * \tparam TO             DataType of the output stream - safely deducible from the paramaters
 * \tparam TA             DataType of the activation class (e.g. thresholds) - safely deducible from the paramaters
 *
 * \param in              Input stream
 * \param out             Output stream
 * \param activation      Activation class
 * \param reps            Number of time the function has to be repeatedly executed (e.g. number of images)
 */
#define Thresholding_Reps 1
#define Total_spatial_size ImgDim*ImgDim
#define Thresholding_PE 64

void Thresholding_Batch_preAdd(
				hls::stream<ap_uint<SIMD*Input_precision>> &in0_V,
				hls::stream<ap_uint<Thresholding_PE*Input_precision>> &out_V,
				ThresholdsActivation<1,PE,NUM_THRES,ap_int<10>,ap_int<Input_precision>,-8,comp::less_equal<ap_int<10>, ap_int<10>>> &threshs,
				ap_uint<8> const pe_compute,
				unsigned const imgdim_compute,
				unsigned const nf_compute
			);


// NumTotal is either 32*9=288 or 16*36=576
#define NumTotal_Add 576
void AddStreams_Batch_01(
			hls::stream<ap_uint<Thresholding_PE*Input_precision>> &in0_V,
			hls::stream<ap_uint<Thresholding_PE*Input_precision>> &in1_V,
			hls::stream<ap_uint<SIMD*(Input_precision+1)>> &out_V,
			ap_uint<32> const data_compute, // numTotal
			ap_uint<8> const simd_compute
		);


void Thresholding_Batch_postAdd(
				hls::stream<ap_uint<SIMD*(Input_precision+1)>> &in0_V,
				hls::stream<ap_uint<Thresholding_PE*Input_precision>> &out_V,
				ThresholdsActivation<1,PE,NUM_THRES,ap_int<10>,ap_int<Input_precision>,-8,comp::less_equal<ap_int<10>, ap_int<10>>> &threshs,
				ap_uint<8> const pe_compute,
				unsigned const imgdim_compute,
				unsigned const nf_compute
			);


void Thresholding_Batch_initial(
				hls::stream<ap_uint<2*Input_precision>> &in0_V,
				hls::stream<ap_uint<Thresholding_PE*Input_precision>> &out_V,
				ThresholdsActivation<1,2,15,ap_int<8>,ap_int<4>,-8,comp::less_equal<ap_int<8>, ap_int<8>>> const &threshs
			);


void Thresholding_Batch_preAdd_2(
				hls::stream<ap_uint<SIMD*Input_precision>> &in0_V,
				hls::stream<ap_uint<Thresholding_PE*Input_precision>> &out_V,
				ThresholdsActivation<1,32,15,ap_int<8>,ap_int<4>,-8,comp::less_equal<ap_int<8>, ap_int<8>>> const &threshs
			);


void Thresholding_Batch_preAdd_3(
				hls::stream<ap_uint<SIMD*Input_precision>> &in0_V,
				hls::stream<ap_uint<Thresholding_PE*Input_precision>> &out_V,
				ThresholdsActivation<1,32,15,ap_int<8>,ap_int<4>,-8,comp::less_equal<ap_int<8>, ap_int<8>>> const &threshs
			);


void Thresholding_Batch_postAdd_4(
				hls::stream<ap_uint<SIMD*(Input_precision+1)>> &in0_V,
				hls::stream<ap_uint<Thresholding_PE*Input_precision>> &out_V,
				ThresholdsActivation<1,32,15,ap_int<8>,ap_int<4>,-8,comp::less_equal<ap_int<8>, ap_int<8>>> const &threshs
			);


void Thresholding_Batch_preAdd_1(
				hls::stream<ap_uint<SIMD*Input_precision>> &in0_V,
				hls::stream<ap_uint<Thresholding_PE*Input_precision>> &out_V,
				ThresholdsActivation<1,16,15,ap_int<8>,ap_int<4>,-8,comp::less_equal<ap_int<8>, ap_int<8>>> const &threshs
			);


void Thresholding_Batch_preAdd_5(
				hls::stream<ap_uint<SIMD*Input_precision>> &in0_V,
				hls::stream<ap_uint<Thresholding_PE*Input_precision>> &out_V,
				ThresholdsActivation<1,16,15,ap_int<8>,ap_int<4>,-8,comp::less_equal<ap_int<8>, ap_int<8>>> const &threshs
			);


void Thresholding_Batch_postAdd_6(
				hls::stream<ap_uint<SIMD*(Input_precision+1)>> &in0_V,
				hls::stream<ap_uint<Thresholding_PE*Input_precision>> &out_V,
				ThresholdsActivation<1,16,15,ap_int<8>,ap_int<4>,-8,comp::less_equal<ap_int<8>, ap_int<8>>> const &threshs
			);



#define TopK_class 4
#define TopK_PE 4
#define K 1
#define TopK_reps ImgDim*ImgDim

void LabelSelect(
		hls::stream<ap_uint<4*32>> &in0_V,
		hls::stream<ap_uint<32>> &out_V
	);



#endif
