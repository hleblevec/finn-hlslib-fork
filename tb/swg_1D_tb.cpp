/******************************************************************************
 *  Copyright (c) 2021, Xilinx, Inc.
 *  Copyright (c) 2022, Advanced Micro Devices, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
/******************************************************************************
 *
 *  Authors: Giulio Gambardella <giuliog@xilinx.com>
 *           Felix Jentzsch <felix.jentzsch@upb.de>
 *
 *  \file swg_1D_tb.cpp
 *
 *  Testbench for the sliding window generator HLS block for 1D convolutions
 *
 *****************************************************************************/
#define AP_INT_MAX_W 8191
#include <hls_stream.h>
#include "ap_int.h"
#include <iostream>
#include <string>
#include "bnn-library.h"

#include "data/input_gen_1d.h"

#include "math.h"
using namespace hls;
using namespace std;

#define MAX_IMAGES 1

void Testbench(stream<ap_uint<SIMD1*INPUT_PRECISION1> > & in, stream<ap_uint<SIMD1*INPUT_PRECISION1> > & out); //, unsigned int numReps)

int main()
{
	stream<ap_uint<IFM_Channels1*INPUT_PRECISION1> > input_stream("input_stream");
	stream<ap_uint<IFM_Channels1*INPUT_PRECISION1> > output_stream("output_stream");
	stream<ap_uint<SIMD1*INPUT_PRECISION1> > in_simd("in_simd");
	stream<ap_uint<SIMD1*INPUT_PRECISION1> > out_simd("out_simd");

	static	ap_int<INPUT_PRECISION1> IMAGE[MAX_IMAGES][IFMDim_x][IFMDim_y][IFM_Channels1];
	int counter = 0;
	ap_uint<IFM_Channels1*INPUT_PRECISION1> input_channel = 0;
	for(unsigned int n_image = 0; n_image < MAX_IMAGES; n_image++) {
		for(unsigned int x = 0; x < IFMDim_x; x++) {
			for(unsigned int y = 0; y < IFMDim_y; y++) {
				input_channel = 0;
				for(unsigned int c = 0; c < IFM_Channels1; c++) {
					ap_int<INPUT_PRECISION1> input = (ap_int<INPUT_PRECISION1>)(counter);
					IMAGE[n_image][x][y][c]= input;
					input_channel = input_channel >> INPUT_PRECISION1;
					input_channel(IFM_Channels1*INPUT_PRECISION1-1,(IFM_Channels1-1)*INPUT_PRECISION1)=input;
					counter++;
				}
				input_stream.write(input_channel);
			}
		}
	}
	StreamingDataWidthConverter_Batch<IFM_Channels1*INPUT_PRECISION1, SIMD1*INPUT_PRECISION1, IFMDim_x>(input_stream, in_simd, 1);
	Testbench(in_simd, out_simd);
	StreamingDataWidthConverter_Batch<SIMD1*INPUT_PRECISION1, IFM_Channels1*INPUT_PRECISION1, KERNEL_DIM_x*OFMDim_x*IFM_Channels1/SIMD1>(out_simd, output_stream, 1);

	ap_int<INPUT_PRECISION1> out_chan;
	int expected_value;
	for(unsigned int n_image = 0; n_image < MAX_IMAGES; n_image++) {
		for(unsigned int ox = 0; ox < OFMDim_x; ox++) {
			for(unsigned int oy = 0; oy < OFMDim_y; oy++) {
				for(unsigned int kx = 0; kx < KERNEL_DIM_x; kx++) {
					for(unsigned int ky = 0; ky < KERNEL_DIM_y; ky++) {
						ap_uint<INPUT_PRECISION1*IFM_Channels1> outElem = output_stream.read();
						for(unsigned int chan = 0; chan < IFM_Channels1; chan++) {
							out_chan(INPUT_PRECISION1-1,0) = outElem((chan + 1)*INPUT_PRECISION1-1,chan*INPUT_PRECISION1);
							int output_value = (ap_int<INPUT_PRECISION1>) out_chan;
							expected_value = (ap_int<INPUT_PRECISION1>) IMAGE[n_image][ox*STRIDE_x+kx*DILATION_x][oy+ky*DILATION_y][chan];
							if (output_value != expected_value){
								std::cout << "ERROR: Expected " << expected_value << " actual " <<  output_value << std::endl;
								std::cout << "Position: OFMDim_x " << ox << " OFMDim_y " << oy <<  " KERNEL_DIM_x " <<  kx << " KERNEL_DIM_y " << ky << " IFM_Channels1 " <<  chan << " Image " << n_image << std::endl;
								return 1;
							}
						}
					}
				}
			}
		}
	}

	return 0;
}
