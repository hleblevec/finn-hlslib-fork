#ifndef TC_H
#define TC_H

#include "ap_axi_sdata.h"

/**
 * \brief   FM Padding for fractionally strided convolution - Padds the input of the transposed 
 * convolution with zeroes to prepare for equivalent convolution as per the
 * fractionally strided convolution algorithm. 
 *
 * \tparam	OutputDim_x	Width of the output feature map (padded)
 * \tparam	OutputDim_y	Height of the output feature map (padded)
 * \tparam	Stride  	Transposed Convolution Stride
 * \tparam	NumChannels	Amount of channels of the input feature map
 * \tparam	SIMD			Input parallelism 
 * \tparam	In_t			Input datatype

 * \param		in			Input stream
 * \param		out			Output stream
 *
 */
template<
	unsigned int OutputDim_x,
	unsigned int OutputDim_y,
	unsigned int Stride,
	unsigned int NumChannels,
	unsigned int SIMD,
	typename In_t
>
void FMPadding_strd(
	hls::stream<ap_uint<SIMD*In_t::width>> &in,
	hls::stream<ap_uint<SIMD*In_t::width>> &out
){
	static_assert(NumChannels%SIMD == 0, "Channel count must be a SIMD multiple.");
	constexpr unsigned int Folding = NumChannels/SIMD;	
	for(unsigned int y = 0; y<OutputDim_y; y++){
		for(unsigned int x=0; x < OutputDim_x; x++){
			for(unsigned int simd=0; simd < Folding; simd++) {
#pragma HLS pipeline style=flp II=1
				ap_uint<SIMD*In_t::width> outData;
				outData = 0;
				if(x % Stride == 0 && y % Stride == 0){
					outData = in.read();
				}
				out.write(outData);
			}
		}
	}
}

/**
 * \brief   FM Padding for fractionally strided convolution - Padds the input of the transposed 
 * convolution with zeroes to prepare for equivalent convolution as per the
 * fractionally strided convolution algorithm. 
 *
 * \tparam	OutputDim_x	Width of the output feature map (padded)
 * \tparam	OutputDim_y	Height of the output feature map (padded)
 * \tparam	Stride  	Transposed Convolution Stride
 * \tparam	NumChannels	Amount of channels of the input feature map
 * \tparam	SIMD			Input parallelism 
 * \tparam	In_t			Input datatype

 * \param		in			Input stream
 * \param		out			Output stream
 *
 */
template<	
		unsigned int OutputDim_x,
		unsigned int OutputDim_y,
		unsigned int Stride,
		unsigned int NumChannels,
		unsigned int SIMD,			
		typename In_t
>	
void FMPadding_strd_Batch(
	hls::stream<ap_uint<SIMD*In_t::width>> &in,
	hls::stream<ap_uint<SIMD*In_t::width>> &out,
	const unsigned int numReps
){
	for (unsigned int rep = 0; rep<numReps; rep++) {
		FMPadding_strd<OutputDim_x, OutputDim_y, Stride, NumChannels, SIMD, In_t>(in, out);
	}
}

#endif
