#define AP_INT_MAX_W 1024

#include "top.hpp"


template<unsigned const dim>
void buf_reload_print(ap_uint<SIMD*Input_precision> (&buf)[MAX_BUFSIZE], int const block)
{
	int count = 1;
	std::cout << std::endl;
	std::cout << "/***********************************************block_" << block << " : buf_reload ***********************************************/" << std::endl;
	for (int i = 0; i < dim*dim; i++) {
		std::cout << std::hex << std::setw(32) << buf[i] << "   ";
		if(i == count * dim - 1){
			std::cout << std::endl;
			count++;
		}
	}
}

template<unsigned const dim>
void buf_concat_print(ap_uint<SIMD*Input_precision> (&buf)[2][MAX_BUFSIZE], int const block, int const index)
{
	int count = 1;
	std::cout << std::endl;
	std::cout << "/**********************************************block_" << block << " : buf_concat[" << index << "]**********************************************/" << std::endl;
	for (int i = 0; i < dim*dim; i++) {
		std::cout << std::hex << std::setw(32) << buf[index][i] << "  ";
		if(i == count * dim - 1){
			std::cout << std::endl;
			count++;
		}
	}
}


template<unsigned LEN, unsigned DIM>
void duplicate_stream(
    hls::stream<ap_uint<LEN*Input_precision>>& in_stream,
    hls::stream<ap_uint<LEN*Input_precision>>& out_stream
) {
	hls::stream<ap_uint<LEN*Input_precision>> out_print;
	while (!in_stream.empty()) {
        ap_uint<LEN*Input_precision> data = in_stream.read();
        out_stream.write(data);
        out_print.write(data);
    }

    int count_channel = 0;
	int count_line = 0;

	while (!out_print.empty()) {
		ap_uint<LEN*Input_precision> out_data = out_print.read();
		count_channel++;
		if(count_channel == Input_NumChannels/SIMD) {
			std::cout << std::hex << std::setw(32) << out_data.range(LEN*Input_precision-1,0).to_string(16) << "  ";
			count_channel = 0;
			count_line++;
		}
		else {
			std::cout << std::hex << std::setw(32) << out_data.range(LEN*Input_precision-1,0).to_string(16) << ",";
		}

		if (count_line == DIM) {
			std::cout << std::endl;
			count_line = 0;
		}
	}

}

template<unsigned LEN>
void duplicate_stream_01(
    hls::stream<ap_uint<LEN*(Input_precision+1)>>& in_stream,
    hls::stream<ap_uint<LEN*(Input_precision+1)>>& out_stream1,
    hls::stream<ap_uint<LEN*(Input_precision+1)>>& out_stream2
) {
    while (!in_stream.empty()) {
        ap_uint<LEN*(Input_precision+1)> data = in_stream.read();
        out_stream1.write(data);
        out_stream2.write(data);
    }
}

// Define the width of the data in the stream
const int STREAM_WIDTH = 8 * 32 * 4;

// Function to convert hex string to ap_uint<128>
ap_uint<STREAM_WIDTH> hex_to_ap_uint_128(const std::string &hex_string) {
    ap_uint<STREAM_WIDTH> result = 0;
    for (size_t i = 0; i < hex_string.length(); i++) {
        char c = hex_string[i];
        result <<= 4; // Shift left by 4 bits (equivalent to one hex digit)
        if (c >= '0' && c <= '9') {
            result |= (ap_uint<STREAM_WIDTH>)(c - '0');
        } else if (c >= 'a' && c <= 'f') {
            result |= (ap_uint<STREAM_WIDTH>)(c - 'a' + 10);
        } else if (c >= 'A' && c <= 'F') {
            result |= (ap_uint<STREAM_WIDTH>)(c - 'A' + 10);
        }
    }
    return result;
}

// Function to read the .dat file and stream the data
void load_weights(hls::stream<ap_uint<STREAM_WIDTH>> &weight, const char *filename) {
    std::ifstream infile(filename);
    std::string line;

    // Check if file is open
    if (!infile.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }

    // Read each line from the .dat file
    while (std::getline(infile, line)) {
        // Convert hex string line to ap_uint<128>
        ap_uint<STREAM_WIDTH> weight_value = hex_to_ap_uint_128(line);

        // Push the value into the HLS stream
        weight.write(weight_value);
    }

    infile.close();
}


// top: initialize and sum of function
int main()
{
	// Define the input and output streams
	hls::stream<ap_uint<SIMD*Input_precision>> in0_V("input_stream");
	hls::stream<ap_uint<SIMD*Input_precision>> out_V("output_stream");
	hls::stream<ap_uint<SIMD*Input_precision>> out_V1("out_V1 output_stream");
	hls::stream<ap_uint<SIMD*Input_precision>> out_V2("out_V2 output_stream");
	hls::stream<ap_uint<SIMD*Input_precision>> out_V3("out_V3 output_stream");
	hls::stream<ap_uint<SIMD*Input_precision>> out_V4("out_V4 output_stream");
	hls::stream<ap_uint<SIMD*Input_precision>> out_V5("out_V5 output_stream");
	hls::stream<ap_uint<SIMD*Input_precision>> out_V6("out_V6 output_stream");
	hls::stream<ap_uint<SIMD*Input_precision>> out_V7("out_V7 output_stream");
	hls::stream<ap_uint<SIMD*Input_precision>> out_V8("out_V8 output_stream");
	hls::stream<ap_uint<SIMD*Input_precision>> out_V9("out_V9 output_stream");
	hls::stream<ap_uint<SIMD*Input_precision>> out_V10("out_V10 output_stream");

	hls::stream<ap_uint<2*Input_precision>> in_0;
	hls::stream<ap_uint<32>> out_0;
	hls::stream<ap_uint<8*32*4>> weight_stream_1;
	hls::stream<ap_uint<8*32*4>> weight_stream_2;

//	ap_uint<SIMD*Input_precision> buf_concat[2][MAX_BUFSIZE];
//	ap_uint<SIMD*Input_precision> buf_reload[MAX_BUFSIZE];


/************************************** Input ******************************************/

	// Initialize input data
	const int size_input = 6;
	const int input_data_size = size_input * size_input * Input_NumChannels/SIMD;

	int size = 36;

	ap_uint<SIMD*Input_precision> input_data[size];


	for (int i = 0; i < size; ++i) {
		// each bit represents a channel
		input_data[i] = 0;  // You can modify this to set specific test values
		//in0_V.write(input_data[i]);
	}
	input_data[8] = 0x01;
	input_data[26] = 0x10;
	input_data[29] = 0x01;

	int pos_i = 1;
	for (int i = 0; i < size; ++i) {
		std::cout << std::hex << input_data[i] << "   ";
		if(i == pos_i * size_input - 1){
			std::cout << std::endl;
			pos_i++;
		}
	}

	int pos = 0;
	for(int i = 0; i < input_data_size; ++i)
	{
		//if(i < 18 && i % 6 < 3)
			//in0_V.write(input_data[pos++] + 1);
		//else
			//in0_V.write(0);
		in_0.write(input_data[i]);
//		in0_V.write(i+1);
	}

	const char *filename1 = "/home/ranhuo/ranhuo98/tb_3level/src/MVAU_memstream_dat/memstream_0.dat";
	const char *filename2 = "/home/ranhuo/ranhuo98/tb_3level/src/MVAU_memstream_dat/memstream_1.dat";
	const char *filename3 = "/home/ranhuo/ranhuo98/tb_3level/src/MVAU_memstream_dat/memstream_2.dat";
	const char *filename4 = "/home/ranhuo/ranhuo98/tb_3level/src/MVAU_memstream_dat/memstream_3.dat";
	const char *filename5 = "/home/ranhuo/ranhuo98/tb_3level/src/MVAU_memstream_dat/memstream_4.dat";
	const char *filename6 = "/home/ranhuo/ranhuo98/tb_3level/src/MVAU_memstream_dat/memstream_5.dat";
	const char *filename7 = "/home/ranhuo/ranhuo98/tb_3level/src/MVAU_memstream_dat/memstream_6.dat";
	const char *filename8 = "/home/ranhuo/ranhuo98/tb_3level/src/MVAU_memstream_dat/memstream_7.dat";
	const char *filename9 = "/home/ranhuo/ranhuo98/tb_3level/src/MVAU_memstream_dat/memstream_8.dat";
	const char *filename10 = "/home/ranhuo/ranhuo98/tb_3level/src/MVAU_memstream_dat/memstream_9.dat";
	const char *filename11 = "/home/ranhuo/ranhuo98/tb_3level/src/MVAU_memstream_dat/memstream_10.dat";
	const char *filename12 = "/home/ranhuo/ranhuo98/tb_3level/src/MVAU_memstream_dat/memstream_11.dat";


///************************************** Main Loop ******************************************/

//	hls::stream<ap_uint<2*Input_precision>> &in_0,
//	hls::stream<ap_uint<32>> &out_0,
//	hls::stream<ap_uint<8*32*4>> &weight_stream_1,
//	hls::stream<ap_uint<8*32*4>> &weight_stream_2,
//	ap_uint<8> block,
//	ap_uint<4> IFMDim_arg,
//	ap_uint<4> OFMDim_arg,
//	ap_uint<8> IFMChannel_arg, // use actual value
//	ap_uint<8> MVAU_OFMChannel_arg, // maximum 32, use actual value
//	ap_uint<8> weight_in_simd_arg,
//	ap_uint<8> MVAU_Tiles_arg,
//	ap_uint<8> UpS_Tiles_arg,
//	ap_uint<8> OUPChannel_arg,
//	ap_uint<2> nf_compute, // = pe / 32
//	ap_uint<8> scale_factor_arg,
//	ap_uint<8> Padding_arg,
//	ap_uint<1> MaxPooling_en,
//	ap_uint<1> Upsampling_en,
//	ap_uint<2> buf_index


/**
 *  ap_uint<8> block			= 1,	|	ap_uint<2> nf_compute		= 1,
	ap_uint<4> IFMDim_arg		= 6,	|	ap_uint<1> MaxPooling_en	= 0,
	ap_uint<4> OFMDim_arg		= 6,	|	ap_uint<1> Upsampling_en	= 0,
	ap_uint<8> IFMChannel_arg	= 2,	|	ap_uint<4> scale_factor_arg	= 0,
	ap_uint<8> OFMChannel_arg	= 16,	|	ap_uint<4> Padding_arg		= 0,
	ap_uint<8> OUPChannel_arg	= 0,	|	ap_uint<2> buf_index		= 0
 */

	load_weights(weight_stream_1, filename1);
//	top(in_0,out_0,weight_stream_1,weight_stream_2,1,6,6,2,16,8,9,0,0,1,0,0,0,0,0);
	top(in_0,out_0,weight_stream_1,weight_stream_2,1);

//	buf_reload_print<6>(buf_reload, 1);


/**
 *  ap_uint<8> block			= 2,	|	ap_uint<2> nf_compute		= 1,
	ap_uint<4> IFMDim_arg		= 6,	|	ap_uint<1> MaxPooling_en	= 1,
	ap_uint<4> OFMDim_arg		= 3,	|	ap_uint<1> Upsampling_en	= 0,
	ap_uint<8> IFMChannel_arg	= 16,	|	ap_uint<4> scale_factor_arg	= 0,
	ap_uint<8> OFMChannel_arg	= 16,	|	ap_uint<4> Padding_arg		= 0,
	ap_uint<8> OUPChannel_arg	= 0,	|	ap_uint<2> buf_index		= 0
 */
	load_weights(weight_stream_1, filename2);
//	top(in_0,out_0,weight_stream_1,weight_stream_2,2,6,3,16,16,16,9,0,0,1,0,0,1,0,0);
	top(in_0,out_0,weight_stream_1,weight_stream_2,2);


/**
 *  ap_uint<8> block			= 3,	|	ap_uint<2> nf_compute		= 1,
	ap_uint<4> IFMDim_arg		= 3,	|	ap_uint<1> MaxPooling_en	= 0,
	ap_uint<4> OFMDim_arg		= 3,	|	ap_uint<1> Upsampling_en	= 0,
	ap_uint<8> IFMChannel_arg	= 16,	|	ap_uint<4> scale_factor_arg	= 0,
	ap_uint<8> OFMChannel_arg	= 32,	|	ap_uint<4> Padding_arg		= 0,
	ap_uint<8> OUPChannel_arg	= 0,	|	ap_uint<2> buf_index		= 0
 */
	load_weights(weight_stream_1, filename3);
//	top(in_0,out_0,weight_stream_1,weight_stream_2,3,3,3,16,32,16,9,0,0,1,0,0,0,0,0);
	top(in_0,out_0,weight_stream_1,weight_stream_2,3);


/**
 *  ap_uint<8> block			= 4,	|	ap_uint<2> nf_compute		= 1,
	ap_uint<4> IFMDim_arg		= 3,	|	ap_uint<1> MaxPooling_en	= 1,
	ap_uint<4> OFMDim_arg		= 1,	|	ap_uint<1> Upsampling_en	= 0,
	ap_uint<8> IFMChannel_arg	= 32,	|	ap_uint<4> scale_factor_arg	= 0,
	ap_uint<8> OFMChannel_arg	= 32,	|	ap_uint<4> Padding_arg		= 0,
	ap_uint<8> OUPChannel_arg	= 0,	|	ap_uint<2> buf_index		= 1
 */
	load_weights(weight_stream_1, filename4);
//	top(in_0,out_0,weight_stream_1,weight_stream_2,4,3,1,32,32,32,9,0,0,1,0,0,1,0,1);
	top(in_0,out_0,weight_stream_1,weight_stream_2,4);


/**
 *  ap_uint<8> block			= 5,	|	ap_uint<2> nf_compute		= 2,
	ap_uint<4> IFMDim_arg		= 1,	|	ap_uint<1> MaxPooling_en	= 0,
	ap_uint<4> OFMDim_arg		= 1,	|	ap_uint<1> Upsampling_en	= 0,
	ap_uint<8> IFMChannel_arg	= 32,	|	ap_uint<4> scale_factor_arg	= 0,
	ap_uint<8> OFMChannel_arg	= 64,	|	ap_uint<4> Padding_arg		= 0,
	ap_uint<8> OUPChannel_arg	= 0,	|	ap_uint<2> buf_index		= 0
 */
	load_weights(weight_stream_1, filename5);
//	top(in_0,out_0,weight_stream_1,weight_stream_2,5,1,1,32,64,32,18,0,0,2,0,0,0,0,0);
	top(in_0,out_0,weight_stream_1,weight_stream_2,5);


/**
 *  ap_uint<8> block			= 6,	|	ap_uint<2> nf_compute		= 2,
	ap_uint<4> IFMDim_arg		= 1,	|	ap_uint<1> MaxPooling_en	= 0,
	ap_uint<4> OFMDim_arg		= 3,	|	ap_uint<1> Upsampling_en	= 1,
	ap_uint<8> IFMChannel_arg	= 64,	|	ap_uint<4> scale_factor_arg	= 3,
	ap_uint<8> OFMChannel_arg	= 64,	|	ap_uint<4> Padding_arg		= 0,
	ap_uint<8> OUPChannel_arg	= 32,	|	ap_uint<2> buf_index		= 1
 */
	load_weights(weight_stream_1, filename6);
	load_weights(weight_stream_2, filename7);
//	top(in_0,out_0,weight_stream_1,weight_stream_2,6,1,3,64,64,64,18,1,32,2,3,0,0,1,1);
	top(in_0,out_0,weight_stream_1,weight_stream_2,6);


/**
 *  ap_uint<8> block			= 7,	|	ap_uint<2> nf_compute		= 1,
	ap_uint<4> IFMDim_arg		= 3,	|	ap_uint<1> MaxPooling_en	= 0,
	ap_uint<4> OFMDim_arg		= 3,	|	ap_uint<1> Upsampling_en	= 0,
	ap_uint<8> IFMChannel_arg	= 32,	|	ap_uint<4> scale_factor_arg	= 0,
	ap_uint<8> OFMChannel_arg	= 16,	|	ap_uint<4> Padding_arg		= 0,
	ap_uint<8> OUPChannel_arg	= 0,	|	ap_uint<2> buf_index		= 0
 */
	load_weights(weight_stream_1, filename8);
//	top(in_0,out_0,weight_stream_1,weight_stream_2,7,3,3,32,16,32,9,0,0,1,0,0,0,0,0);
	top(in_0,out_0,weight_stream_1,weight_stream_2,7);


/**
 *  ap_uint<8> block			= 8,	|	ap_uint<2> nf_compute		= 1,
	ap_uint<4> IFMDim_arg		= 3,	|	ap_uint<1> MaxPooling_en	= 0,
	ap_uint<4> OFMDim_arg		= 6,	|	ap_uint<1> Upsampling_en	= 1,
	ap_uint<8> IFMChannel_arg	= 16,	|	ap_uint<4> scale_factor_arg	= 2,
	ap_uint<8> OFMChannel_arg	= 16,	|	ap_uint<4> Padding_arg		= 0,
	ap_uint<8> OUPChannel_arg	= 16,	|	ap_uint<2> buf_index		= 0
 */
	load_weights(weight_stream_1, filename9);
	load_weights(weight_stream_2, filename10);
//	top(in_0,out_0,weight_stream_1,weight_stream_2,8,3,6,16,16,16,9,1,16,1,2,0,0,1,0);
	top(in_0,out_0,weight_stream_1,weight_stream_2,8);


/**
 *  ap_uint<8> block			= 9,	|	ap_uint<2> nf_compute		= 1,
	ap_uint<4> IFMDim_arg		= 6,	|	ap_uint<1> MaxPooling_en	= 0,
	ap_uint<4> OFMDim_arg		= 6,	|	ap_uint<1> Upsampling_en	= 0,
	ap_uint<8> IFMChannel_arg	= 16,	|	ap_uint<4> scale_factor_arg	= 0,
	ap_uint<8> OFMChannel_arg	= 8,	|	ap_uint<4> Padding_arg		= 0,
	ap_uint<8> OUPChannel_arg	= 0,	|	ap_uint<2> buf_index		= 0
 */
	load_weights(weight_stream_1, filename11);
//	top(in_0,out_0,weight_stream_1,weight_stream_2,9,6,6,16,8,16,9,0,0,1,0,0,0,0,0);
	top(in_0,out_0,weight_stream_1,weight_stream_2,9);



/**
 *  ap_uint<8> block			= 10,	|	ap_uint<2> nf_compute		= 1,
	ap_uint<4> IFMDim_arg		= 6,	|	ap_uint<1> MaxPooling_en	= 0,
	ap_uint<4> OFMDim_arg		= 6,	|	ap_uint<1> Upsampling_en	= 0,
	ap_uint<8> IFMChannel_arg	= 8,	|	ap_uint<4> scale_factor_arg	= 0,
	ap_uint<8> OFMChannel_arg	= 8,	|	ap_uint<4> Padding_arg		= 0,
	ap_uint<8> OUPChannel_arg	= 0,	|	ap_uint<2> buf_index		= 0
 */
	load_weights(weight_stream_1, filename12);
//	top(in_0,out_0,weight_stream_1,weight_stream_2,10,6,6,8,8,8,9,0,0,1,0,0,0,0,0);
	top(in_0,out_0,weight_stream_1,weight_stream_2,10);


/************************************** Print ********************************************/

	int buf_index = 0;

	// Print output data
	std::cout << "\nOutput data:" << std::endl;
	int count_channel = 0;
	int count_line = 0;
	const int length = 32;

	while (!out_0.empty()) {
		ap_uint<length> out_data = out_0.read();
//		ap_uint<4*32> out_data = out_1024.read();
		count_channel++;
		if(count_channel == Input_NumChannels/SIMD) {
			std::cout << std::hex << std::setw(32) << out_data.range(length - 1, 0).to_string(16) << "  ";
			count_channel = 0;
			count_line++;
		}
		else {
			std::cout << std::hex << std::setw(32) << out_data.range(length - 1, 0).to_string(16) << ",";
		}

		if (count_line == 6) {
			std::cout << std::endl;
			count_line = 0;
		}
	}
	if (count_line != 0) {
		std::cout << std::endl; // Print a newline if the last line was not complete
	}

	int count_in_v = 0;
	while(!in_0.empty())
	{
		ap_uint<SIMD*Input_precision> in_data = in_0.read();
		std::cout << in_data << " ";
		count_in_v++;
	}
	std::cout << "\n" << std::dec << count_in_v << std::endl;

	return 0;

}
