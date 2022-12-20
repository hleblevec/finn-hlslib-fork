#include <iostream>
#include <cmath>
#include <ctime>
#include <cstring>
#include <hls_stream.h>
#include <cstdlib>
#define AP_INT_MAX_W 8191
#include "ap_int.h"
#include "weights.hpp"
#include "activations.hpp"
#include "interpret.hpp"
#include "tc.hpp"
#include "data/config_strd.h"
#include "data/memdata_strd.h"

using namespace hls;
using namespace std;

#define MAX_IMAGES 1
void Testbench_strd(stream<ap_uint<IFM_Channels1*INPUT_PRECISION>> &in, stream<ap_uint<OFM_Channels1*ACTIVATION_PRECISION>> &out, unsigned int numReps);

int main()
{
    //create_memdata();
	static	ap_uint<INPUT_PRECISION> IMAGE[MAX_IMAGES][IFMDim1][IFMDim1][IFM_Channels1];
	static	ap_uint<ACTIVATION_PRECISION> TEST[MAX_IMAGES][OFMDim3][OFMDim3][OFM_Channels1];
	stream<ap_uint<IFM_Channels1*INPUT_PRECISION> > input_stream("input_stream");
	stream<ap_uint<OFM_Channels1*ACTIVATION_PRECISION> > output_stream("output_stream");
	unsigned int counter = 0;
	for (unsigned int n_image = 0; n_image < MAX_IMAGES; n_image++) {
			for (unsigned int oy = 0; oy < IFMDim1; oy++) {
				for (unsigned int ox = 0; ox < IFMDim1; ox++) {
					ap_uint<INPUT_PRECISION*IFM_Channels1> input_channel = 0;
					for(unsigned int channel = 0; channel < IFM_Channels1; channel++)
					{
						ap_uint<INPUT_PRECISION> input = (ap_uint<INPUT_PRECISION>)(counter);
						IMAGE[n_image][ox][oy][channel]= input;
						input_channel = input_channel >> INPUT_PRECISION;
						input_channel(IFM_Channels1*INPUT_PRECISION-1,(IFM_Channels1-1)*INPUT_PRECISION)=input;
						counter++;
					}
					input_stream.write(input_channel);
				}
			}
		}
	static	ap_uint<WEIGHT_PRECISION> W1[OFM_Channels1][KERNEL][KERNEL][IFM_Channels1];
	// initialize the weights
	constexpr int TX = (IFM_Channels1*KERNEL*KERNEL) / SIMD1;
	constexpr int TY = OFM_Channels1 / PE1;
	unsigned int kx=0;
	unsigned int ky=0;
	unsigned int chan_count=0;
	unsigned int out_chan_count=0;

	for (unsigned int oy = 0; oy < TY; oy++) {
		for(unsigned int pe=0;pe <PE1;pe++){
			for (unsigned int ox = 0; ox <TX; ox++) {
				for(unsigned int simd=0;simd<SIMD1;simd++){
					W1[out_chan_count][kx][ky][chan_count] = PARAM::tc_weights.weights(oy*TX + ox)[pe][simd];
					chan_count++;
				    if (chan_count==IFM_Channels1){
				    	chan_count=0;
						kx++;
						if (kx==KERNEL){
							kx=0;
							ky++;
							if (ky==KERNEL){
								ky=0;
						    	out_chan_count++;
							    if (out_chan_count==OFM_Channels1){
							    	out_chan_count=0;
								}
						    }
					    }
					}
				}
			}
		}
	}

	transposed_conv<MAX_IMAGES,IFMDim1,OFMDim3,IFM_Channels1,OFM_Channels1, KERNEL, STRIDE, TCPADDING, ap_uint<INPUT_PRECISION>, ap_uint<WEIGHT_PRECISION>, ap_uint<ACTIVATION_PRECISION>>(IMAGE, W1, TEST);
	Testbench_strd(input_stream, output_stream, MAX_IMAGES);
	int err_counter = 0, err_perimage=0;
	ap_int<ACTIVATION_PRECISION> out_chan;
	for (unsigned int n_image = 0; n_image < MAX_IMAGES; n_image++) {
		for (unsigned int oy = 0; oy < OFMDim3; oy++) {
			for (unsigned int ox = 0; ox < OFMDim3; ox++) {
				ap_uint<OFM_Channels1*ACTIVATION_PRECISION> outElem = output_stream.read();
				for(unsigned int channel = 0; channel < OFM_Channels1; channel++){
					ap_int<ACTIVATION_PRECISION> EXP = TEST[n_image][ox][oy][channel];
					out_chan(ACTIVATION_PRECISION-1,0) = outElem((channel + 1)*ACTIVATION_PRECISION-1,channel*ACTIVATION_PRECISION);

					if (EXP != out_chan){
						std::cout << "ERROR: Expected["<<oy <<"]["<<ox<<"]["<<channel<<"]=" << EXP << " actual " <<  out_chan << std::endl;
						err_counter ++;
						err_perimage++;
					}else{
						std::cout << "Expected["<<oy <<"]["<<ox<<"]["<<channel<<"]=" << EXP << " actual " <<  out_chan << std::endl;
					}
				}
			}
		}
		if(err_perimage == 0){
			std::cout << "Image # " << n_image << " passed the testing."<< std::endl;
		}
		else{
			err_perimage=0;
			std::cout << "Image # " << n_image << " failed the testing."<< std::endl;
		}
	}
	if(err_counter == 0){
		return 0;
	}
	else{
		return 1;
	}
}