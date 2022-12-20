#ifndef TC_TB_H
#define TC_TB_H

template<
	int MAX_IMAGE,
	int IFMDim,
	int OFMDim,
	int IFMCh,
	int OFMCh,
	int kernel,
	int stride,
	int padding,
	typename TI,
	typename TW,
	typename TO
>
void transposed_conv(TI const img[MAX_IMAGE][IFMDim][IFMDim][IFMCh], TW const weights[OFMCh][kernel][kernel][IFMCh], TO out[MAX_IMAGE][OFMDim][OFMDim][OFMCh]){
	int oh = 0;
	int ow = 0;
	for(int n=0;n<MAX_IMAGE;n++)
		for(int oc=0;oc<OFMCh;oc++)
			for(int ic=0;ic<IFMCh;ic++)
				for(int kw=0;kw<kernel;kw++)
					for(int kh=0;kh<kernel;kh++)
						for(int ih=0;ih<IFMDim;ih++)
							for(int iw=0;iw<IFMDim;iw++){
								oh = (stride * ih) + kh - padding;
								ow = (stride * iw) + kw - padding;
								if (oh < OFMDim && ow < OFMDim && ow >= 0 && oh >=0){
									out[n][oh][ow][oc] += img[n][ih][iw][ic] * weights[oc][kh][kw][ic];
								}
							}
								
}

#endif
