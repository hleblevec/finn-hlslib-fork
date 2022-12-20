#include <hls_stream.h>
#include <ap_int.h>

#include "tc.h"
// #include "convlayer.h"
#include "weights.hpp"
#include "activations.hpp"
#include "interpret.hpp"
#include "data/config_strd.h"
#include "data/memdata_strd.h"
#include "streamtools.h"
#include "slidingwindow.h"
#include "mvau.hpp"

using namespace hls;

void Testbench_strd(stream<ap_uint<IFM_Channels1*INPUT_PRECISION>> &in, stream<ap_uint<OFM_Channels1*ACTIVATION_PRECISION>> &out, unsigned int numReps){
#pragma HLS INTERFACE port=return ap_ctrl_none
#pragma HLS DATAFLOW disable_start_propagation
    constexpr unsigned MatrixW = KERNEL * KERNEL * IFM_Channels1;
    constexpr unsigned MatrixH = OFM_Channels1;
    constexpr unsigned InpPerImage = IFMDim1 * IFMDim1;
    hls::stream<ap_uint<SIMD1*INPUT_PRECISION>> padIn("padIn");
    hls::stream<ap_uint<SIMD1*INPUT_PRECISION>> padOut("padOut");
    hls::stream<ap_uint<SIMD1*INPUT_PRECISION>> cigInp("cigInp");
    hls::stream<ap_uint<SIMD1*INPUT_PRECISION>> convInp("convInp");
    hls::stream<ap_uint<PE1*ACTIVATION_PRECISION>> mvOut("mvOut");
    StreamingDataWidthConverter_Batch<IFM_Channels1*INPUT_PRECISION, SIMD1*INPUT_PRECISION, InpPerImage>(in, padIn, numReps);
    FMPadding_strd_Batch<OFMDim1, OFMDim1, STRIDE, IFM_Channels1, SIMD1, ap_uint<INPUT_PRECISION>>(padIn, padOut, numReps);
    FMPadding_Batch<OFMDim1, OFMDim2, ConvPADDING, ConvPADDING, IFM_Channels1, SIMD1, ap_uint<INPUT_PRECISION>>(padOut, cigInp, numReps);
    ConvolutionInputGenerator<KERNEL, IFM_Channels1, INPUT_PRECISION, OFMDim2,
                OFMDim3, SIMD1,1>(cigInp, convInp, numReps, ap_resource_dflt());
    Matrix_Vector_Activate_Batch<MatrixW, MatrixH, SIMD1, PE1, 1, Slice<ap_uint<INPUT_PRECISION>>, Slice<ap_uint<ACTIVATION_PRECISION>>, Identity>
        (convInp,
        mvOut,
        PARAM::conv_weights, PassThroughActivation<ap_uint<ACTIVATION_PRECISION>>(), numReps* OFMDim3 * OFMDim3, ap_resource_dsp());
    StreamingDataWidthConverter_Batch<PE1*ACTIVATION_PRECISION, OFM_Channels1*ACTIVATION_PRECISION, OFMDim3 * OFMDim3 * (OFM_Channels1 / PE1)>(mvOut, out, numReps);
}
