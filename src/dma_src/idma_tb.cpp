#include "idma.hpp"

int main(){
	ap_uint<DataWidth> in0_V[9];
	hls::stream<ap_uint<2*precision> > out_V;

	for(int i = 0; i < 9; i++)
	{
		in0_V[i] = 0x87654321;
	}

	idma(in0_V, out_V);

	return 0;
}
