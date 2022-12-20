import numpy as np

outFileWeights = open("memdata_strd.h" , "wt")
outFileConfig = open("config_strd.h" , "wt")

kernel_dim = 4
stride = 2
tcpadding = 1
input_precision = 8
ifm_channels = 1
ofm_channels = 3
ifm_dimension = 2

activation_precision = 8
simd = 1
pe = 1
w_precision = 8

convpadding = kernel_dim - tcpadding - 1
ofm_dim1 = ifm_dimension + (ifm_dimension - 1) * (stride - 1)
ofm_dim2 = ofm_dim1 + 2 * convpadding
ofm_dimension = (ifm_dimension - 1) * stride - 2 * tcpadding + (kernel_dim - 1) + 1

tile = ifm_channels *kernel_dim*kernel_dim * ofm_channels // (simd*pe)

outFileConfig.write("constexpr int unsigned IFMDim1 = %d; \n" % ifm_dimension)
outFileConfig.write("constexpr int unsigned STRIDE = %d; \n" % stride)
outFileConfig.write("constexpr int unsigned OFMDim1 = %d; \n" % ofm_dim1)
outFileConfig.write("constexpr int unsigned IFM_Channels1 = %d; \n" % ifm_channels)
outFileConfig.write("constexpr int unsigned OFM_Channels1 = %d; \n" % ofm_channels)
outFileConfig.write("constexpr int unsigned TCPADDING = %d; \n" % tcpadding)
outFileConfig.write("constexpr int unsigned KERNEL = %d; \n" % kernel_dim)
outFileConfig.write("constexpr int unsigned ConvPADDING = %d; \n" % convpadding)
outFileConfig.write("constexpr int unsigned OFMDim2 = %d; \n" % ofm_dim2)
outFileConfig.write("constexpr int unsigned OFMDim3 = %d; \n" % ofm_dimension)
outFileConfig.write("constexpr int unsigned INPUT_PRECISION = %d; \n" % input_precision)
outFileConfig.write("constexpr int unsigned ACTIVATION_PRECISION = %d; \n" % activation_precision)
outFileConfig.write("constexpr int unsigned WEIGHT_PRECISION = %d; \n" % w_precision)
outFileConfig.write("constexpr int unsigned SIMD1 = %d; \n" % simd)
outFileConfig.write("constexpr int unsigned PE1 = %d; \n" % pe)

outFileConfig.close()

weights = np.random.randint(0, 1<<w_precision-1, (ofm_channels, ifm_channels, kernel_dim, kernel_dim))
np.save("weights.npy", weights)

# tc_weights = np.reshape(np.transpose(weights, (0,2,3,1)), (pe, simd*tile))
tc_weights = np.reshape(weights, (pe, simd*tile))


# conv_weights = np.reshape(np.transpose(np.moveaxis(np.rot90(weights, 2, (2,3)), 0, 1),(0,2,3,1)), (pe, simd*tile))
conv_weights = np.reshape(np.moveaxis(np.rot90(weights, 2, (2,3)), 0, 1), (pe, simd*tile))



outFileWeights.write("#ifndef PARAMS_HPP\n")
outFileWeights.write("#define PARAMS_HPP\n")

outFileWeights.write("namespace PARAM{ \n")
outFileWeights.write("static FixedPointWeights<%d,ap_int<%d>,%d,%d> tc_weights= {\n{\n" %(simd,w_precision,pe,tile))

for p in range(pe):
    outFileWeights.write("{ \n")
    for t in range(tile):
        val = 0
        if simd != 1:
            for s in reversed(range(simd)):
                val = tc_weights[p,t*simd+s] << w_precision | val
        else:
            val = tc_weights[p,t]
        outFileWeights.write("%s" % hex(val))
        if t!=tile-1:
            outFileWeights.write(",\n")
        else:
            outFileWeights.write("} \n")
    if p!=pe-1:
        outFileWeights.write(",")

outFileWeights.write("}\n};\n \n")

outFileWeights.write("static FixedPointWeights<%d,ap_int<%d>,%d,%d> conv_weights= {\n{\n" %(simd,w_precision,pe,tile))

for p in range(pe):
    outFileWeights.write("{ \n")
    for t in range(tile):
        val = 0
        if simd != 1:
            for s in reversed(range(simd)):
                val = conv_weights[p,t*simd+s] << w_precision | val
        else:
            val = conv_weights[p,t]
        outFileWeights.write("%s" % hex(val))
        if t!=tile-1:
            outFileWeights.write(",\n")
        else:
            outFileWeights.write("} \n")
    if p!=pe-1:
        outFileWeights.write(",")

outFileWeights.write("}\n};\n}\n")
outFileWeights.write("#endif \n")
outFileWeights.close()

