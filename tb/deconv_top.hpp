#ifndef DECONV_TOP_HPP
#define DECONV_TOP_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>

// TODO: PE
constexpr unsigned  SIMD = 1;
constexpr unsigned  CI = 1;	// input channels
constexpr unsigned  CO = 2;	// output channels
constexpr unsigned  H = 6;	// height
constexpr unsigned  W = 6;	// Width
constexpr unsigned  K = 4;	// Kernel Size
constexpr unsigned  S = 2; 	// Stride

constexpr unsigned  D = CI/SIMD;
using  TW = ap_uint< 8>;
using  TI = ap_uint< 4>;
using  TO = ap_uint<16>;

void deconv_top(
	hls::stream<hls::vector<TI, SIMD>> &src,
	hls::stream<TO> &dst
);

#endif
