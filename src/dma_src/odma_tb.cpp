#include "odma.hpp"


int main(){
	hls::stream<ap_uint<32> > in0_V;
	ap_uint<32> out_V[36];

	for(int i = 0; i < 36; i++){
		in0_V.write(i);
	}

	odma(in0_V, out_V);
}
