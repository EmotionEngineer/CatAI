#include <math.h>

#include "catnext_activation.h"

LIB_HIDDEN void node_model_dense_4_MatMul( const float A[1][300], const float B[300][300], float Y[1][300] )
{
	/* MatMul */
	for( uint32_t r=0; r<1; r++ )
		for( uint32_t c=0; c<300; c++ ) {
			Y[r][c] = 0;
			for( uint32_t i=0; i<300; i++ )
				Y[r][c] += A[r][i] * B[i][c];
		}
}

/*
 * Operand:           Add
 * Name: model/dense_4/BiasAdd
 */
LIB_HIDDEN void node_model_dense_4_BiasAdd( const float A[1][300], const float B[300], float C[1][300] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<300; i1++) {
		C[i0][i1] = A[0][i1]+B[i1];;
	}
	}
}

/*
 * Operand:           Transpose
 * Name: model/permute/transpose
 */
LIB_HIDDEN void node_model_permute_transpose( const float input[1][13][384][384][3], float output[1][384][384][13][3] )
{
	/* Transpose
	 * perm = 0 2 3 1 4 
	 */
	for( uint32_t i0=0; i0<1; i0++ ) {
	for( uint32_t i1=0; i1<13; i1++ ) {
	for( uint32_t i2=0; i2<384; i2++ ) {
	for( uint32_t i3=0; i3<384; i3++ ) {
	for( uint32_t i4=0; i4<3; i4++ ) {
		output[i0][i2][i3][i1][i4] = input[i0][i1][i2][i3][i4];
	}
	}
	}
	}
	}
}

/*
 * Operand:           Reshape
 * Name: model/reshape/Reshape
 */
LIB_HIDDEN void node_model_reshape_Reshape( const float data[1][384][384][13][3], const int64_t shape[4], float reshaped[1][384][384][39] )
{
	/*Reshape*/
	float *data_ptr = (float*)data;
	float *reshaped_ptr = (float*)reshaped;
	for( uint32_t i=0; i<5750784; i++ )
		reshaped_ptr[i] = data_ptr[i];

}

/*
 * Operand:           Transpose
 * Name: catnext/stem_conv/Conv2D__6
 */
LIB_HIDDEN void node_model_catnext_stem_conv_Conv2D__6( const float input[1][384][384][39], float output[1][39][384][384] )
{
	/* Transpose
	 * perm = 0 3 1 2 
	 */
	for( uint32_t i0=0; i0<1; i0++ ) {
	for( uint32_t i1=0; i1<384; i1++ ) {
	for( uint32_t i2=0; i2<384; i2++ ) {
	for( uint32_t i3=0; i3<39; i3++ ) {
		output[i0][i3][i1][i2] = input[i0][i1][i2][i3];
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stem_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stem_conv_Conv2D( const float x[1][39][384][384], const float w[32][39][3][3], const float bias[32], float y[1][32][192][192] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 2 2 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<32; m++) {
		for( int32_t o0=0, i0=-1; o0<192; o0++, i0+=2) {
		for( int32_t o1=0, i1=-1; o1<192; o1++, i1+=2) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<39; c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=384) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=384) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stem_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stem_swish_Sigmoid( const float X[1][32][192][192], float Y[1][32][192][192] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<32; i1++) {
	for (unsigned i2=0; i2<192; i2++) {
	for (unsigned i3=0; i3<192; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stem_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stem_swish_mul_1( const float A[1][32][192][192], const float B[1][32][192][192], float C[1][32][192][192] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<32; i1++) {
	for (unsigned i2=0; i2<192; i2++) {
	for (unsigned i3=0; i3<192; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack1_block1_deep_1_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack1_block1_deep_1_conv_Conv2D( const float x[1][32][192][192], const float w[128][32][1][1], const float bias[128], float y[1][128][192][192] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<128; m++) {
		for( int32_t o0=0, i0=0; o0<192; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<192; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<32; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=192) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=192) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack1_block1_deep_1_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack1_block1_deep_1_swish_Sigmoid( const float X[1][128][192][192], float Y[1][128][192][192] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<128; i1++) {
	for (unsigned i2=0; i2<192; i2++) {
	for (unsigned i3=0; i3<192; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack1_block1_deep_1_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack1_block1_deep_1_swish_mul_1( const float A[1][128][192][192], const float B[1][128][192][192], float C[1][128][192][192] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<128; i1++) {
	for (unsigned i2=0; i2<192; i2++) {
	for (unsigned i3=0; i3<192; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack1_block1_deep_2_GC_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack1_block1_deep_2_GC_conv_Conv2D( const float x[1][128][192][192], const float w[128][16][3][3], const float bias[128], float y[1][128][96][96] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 8
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 2 2 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 16; // output group size, i.e. maps/group
	uint32_t gi = 16; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<8; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<96; o0++, i0+=2) {
		for( int32_t o1=0, i1=-1; o1<96; o1++, i1+=2) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=192) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=192) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c-(gi*g)][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack1_block1_deep_2_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack1_block1_deep_2_swish_Sigmoid( const float X[1][128][96][96], float Y[1][128][96][96] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<128; i1++) {
	for (unsigned i2=0; i2<96; i2++) {
	for (unsigned i3=0; i3<96; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack1_block1_deep_2_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack1_block1_deep_2_swish_mul_1( const float A[1][128][96][96], const float B[1][128][96][96], float C[1][128][96][96] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<128; i1++) {
	for (unsigned i2=0; i2<96; i2++) {
	for (unsigned i3=0; i3<96; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           GlobalAveragePool
 * Name: catnext/tf.math.reduce_mean/Mean
 */
LIB_HIDDEN void node_model_catnext_tf_math_reduce_mean_Mean( const float input[1][128][96][96], float output[1][128][1][1] )
{
	/* GlobalAveragePool */
	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<128; c++ ) {
		float dimsum=0.0f;
		for( int32_t d0 = 0; d0<96; d0++ ) {
		for( int32_t d1 = 0; d1<96; d1++ ) {
			dimsum +=  input[b][c][d0][d1];
		}
		}
		output[b][c][0][0] = dimsum / 9216;
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack1_block1_deep_2_se_1_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack1_block1_deep_2_se_1_conv_BiasAdd( const float x[1][128][1][1], const float w[32][128][1][1], const float bias[32], float y[1][32][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<32; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<128; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Relu
 * Name: catnext/stack1_block1_deep_2_se_relu/Relu
 */
LIB_HIDDEN void node_model_catnext_stack1_block1_deep_2_se_relu_Relu( const float X[1][32][1][1], float Y[1][32][1][1] )
{
	/*Relu*/
	float *X_ptr = (float*)X;
	float *Y_ptr = (float*)Y;
	for( uint32_t i=0; i<32; i++ )
		Y_ptr[i] = X_ptr[i] > 0 ? X_ptr[i] : 0;

}

/*
 * Operand:           Conv
 * Name: catnext/stack1_block1_deep_2_se_2_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack1_block1_deep_2_se_2_conv_BiasAdd( const float x[1][32][1][1], const float w[128][32][1][1], const float bias[128], float y[1][128][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<128; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<32; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack1_block1_deep_2_se_sigmoid/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack1_block1_deep_2_se_sigmoid_Sigmoid( const float X[1][128][1][1], float Y[1][128][1][1] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<128; i1++) {
	for (unsigned i2=0; i2<1; i2++) {
	for (unsigned i3=0; i3<1; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack1_block1_deep_2_se_out/mul
 */
LIB_HIDDEN void node_model_catnext_stack1_block1_deep_2_se_out_mul( const float A[1][128][96][96], const float B[1][128][1][1], float C[1][128][96][96] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<128; i1++) {
	for (unsigned i2=0; i2<96; i2++) {
	for (unsigned i3=0; i3<96; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][0][0];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack1_block1_deep_3_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack1_block1_deep_3_conv_Conv2D( const float x[1][128][96][96], const float w[48][128][1][1], const float bias[48], float y[1][48][96][96] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<48; m++) {
		for( int32_t o0=0, i0=0; o0<96; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<96; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<128; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=96) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=96) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Conv
 * Name: catnext/stack1_block2_deep_1_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack1_block2_deep_1_conv_Conv2D( const float x[1][48][96][96], const float w[192][48][1][1], float y[1][192][96][96] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<96; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<96; o1++, i1+=1) {
			y[b][m][o0][o1] = 0;
			for( int32_t c=0; c<48; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=96) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=96) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           BatchNormalization
 * Name: catnext/stack1_block2_deep_1_bn/FusedBatchNormV3
 */
LIB_HIDDEN void node_model_catnext_stack1_block2_deep_1_bn_FusedBatchNormV3( const float X[1][192][96][96], const float scale[192], const float bias[192], const float mean[192], const float var[192], float output[1][192][96][96] )
{
	/* BatchNormalization
	 * epsilon = 1.0009999641624744982e-05
	 * momentum = 0.89999997615814208984
	 */

	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<192; c++ ) {
	for( uint32_t i2=0; i2<96; i2++ ) {
	for( uint32_t i3=0; i3<96; i3++ ) {
		float tmp_X = ( X[b][c][i2][i3] - mean[c] ) / ( var[c] );
		output[b][c][i2][i3] = tmp_X * scale[c] + bias[c];
	}
	}
	}
	}
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack1_block2_deep_1_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack1_block2_deep_1_swish_Sigmoid( const float X[1][192][96][96], float Y[1][192][96][96] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<96; i2++) {
	for (unsigned i3=0; i3<96; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack1_block2_deep_1_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack1_block2_deep_1_swish_mul_1( const float A[1][192][96][96], const float B[1][192][96][96], float C[1][192][96][96] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<96; i2++) {
	for (unsigned i3=0; i3<96; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack1_block2_deep_2_GC_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack1_block2_deep_2_GC_conv_Conv2D( const float x[1][192][96][96], const float w[192][16][3][3], const float bias[192], float y[1][192][96][96] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 12
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 16; // output group size, i.e. maps/group
	uint32_t gi = 16; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<12; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<96; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<96; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=96) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=96) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c-(gi*g)][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack1_block2_deep_2_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack1_block2_deep_2_swish_Sigmoid( const float X[1][192][96][96], float Y[1][192][96][96] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<96; i2++) {
	for (unsigned i3=0; i3<96; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack1_block2_deep_2_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack1_block2_deep_2_swish_mul_1( const float A[1][192][96][96], const float B[1][192][96][96], float C[1][192][96][96] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<96; i2++) {
	for (unsigned i3=0; i3<96; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           GlobalAveragePool
 * Name: catnext/tf.math.reduce_mean_1/Mean
 */
LIB_HIDDEN void node_model_catnext_tf_math_reduce_mean_1_Mean( const float input[1][192][96][96], float output[1][192][1][1] )
{
	/* GlobalAveragePool */
	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<192; c++ ) {
		float dimsum=0.0f;
		for( int32_t d0 = 0; d0<96; d0++ ) {
		for( int32_t d1 = 0; d1<96; d1++ ) {
			dimsum +=  input[b][c][d0][d1];
		}
		}
		output[b][c][0][0] = dimsum / 9216;
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack1_block2_deep_2_se_1_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack1_block2_deep_2_se_1_conv_BiasAdd( const float x[1][192][1][1], const float w[48][192][1][1], const float bias[48], float y[1][48][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<48; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Relu
 * Name: catnext/stack1_block2_deep_2_se_relu/Relu
 */
LIB_HIDDEN void node_model_catnext_stack1_block2_deep_2_se_relu_Relu( const float X[1][48][1][1], float Y[1][48][1][1] )
{
	/*Relu*/
	float *X_ptr = (float*)X;
	float *Y_ptr = (float*)Y;
	for( uint32_t i=0; i<48; i++ )
		Y_ptr[i] = X_ptr[i] > 0 ? X_ptr[i] : 0;

}

/*
 * Operand:           Conv
 * Name: catnext/stack1_block2_deep_2_se_2_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack1_block2_deep_2_se_2_conv_BiasAdd( const float x[1][48][1][1], const float w[192][48][1][1], const float bias[192], float y[1][192][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<48; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack1_block2_deep_2_se_sigmoid/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack1_block2_deep_2_se_sigmoid_Sigmoid( const float X[1][192][1][1], float Y[1][192][1][1] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<1; i2++) {
	for (unsigned i3=0; i3<1; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack1_block2_deep_2_se_out/mul
 */
LIB_HIDDEN void node_model_catnext_stack1_block2_deep_2_se_out_mul( const float A[1][192][96][96], const float B[1][192][1][1], float C[1][192][96][96] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<96; i2++) {
	for (unsigned i3=0; i3<96; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][0][0];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack1_block2_deep_3_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack1_block2_deep_3_conv_Conv2D( const float x[1][192][96][96], const float w[48][192][1][1], const float bias[48], float y[1][48][96][96] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<48; m++) {
		for( int32_t o0=0, i0=0; o0<96; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<96; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=96) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=96) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name: catnext/stack1_block2_add/add
 */
LIB_HIDDEN void node_model_catnext_stack1_block2_add_add( const float A[1][48][96][96], const float B[1][48][96][96], float C[1][48][96][96] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<48; i1++) {
	for (unsigned i2=0; i2<96; i2++) {
	for (unsigned i3=0; i3<96; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block1_deep_1_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack2_block1_deep_1_conv_Conv2D( const float x[1][48][96][96], const float w[192][48][1][1], const float bias[192], float y[1][192][96][96] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<96; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<96; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<48; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=96) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=96) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack2_block1_deep_1_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack2_block1_deep_1_swish_Sigmoid( const float X[1][192][96][96], float Y[1][192][96][96] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<96; i2++) {
	for (unsigned i3=0; i3<96; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack2_block1_deep_1_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack2_block1_deep_1_swish_mul_1( const float A[1][192][96][96], const float B[1][192][96][96], float C[1][192][96][96] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<96; i2++) {
	for (unsigned i3=0; i3<96; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block1_deep_2_GC_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack2_block1_deep_2_GC_conv_Conv2D( const float x[1][192][96][96], const float w[192][16][3][3], const float bias[192], float y[1][192][48][48] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 12
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 2 2 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 16; // output group size, i.e. maps/group
	uint32_t gi = 16; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<12; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<48; o0++, i0+=2) {
		for( int32_t o1=0, i1=-1; o1<48; o1++, i1+=2) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=96) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=96) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c-(gi*g)][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack2_block1_deep_2_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack2_block1_deep_2_swish_Sigmoid( const float X[1][192][48][48], float Y[1][192][48][48] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack2_block1_deep_2_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack2_block1_deep_2_swish_mul_1( const float A[1][192][48][48], const float B[1][192][48][48], float C[1][192][48][48] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           GlobalAveragePool
 * Name: catnext/tf.math.reduce_mean_2/Mean
 */
LIB_HIDDEN void node_model_catnext_tf_math_reduce_mean_2_Mean( const float input[1][192][48][48], float output[1][192][1][1] )
{
	/* GlobalAveragePool */
	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<192; c++ ) {
		float dimsum=0.0f;
		for( int32_t d0 = 0; d0<48; d0++ ) {
		for( int32_t d1 = 0; d1<48; d1++ ) {
			dimsum +=  input[b][c][d0][d1];
		}
		}
		output[b][c][0][0] = dimsum / 2304;
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block1_deep_2_se_1_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack2_block1_deep_2_se_1_conv_BiasAdd( const float x[1][192][1][1], const float w[48][192][1][1], const float bias[48], float y[1][48][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<48; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Relu
 * Name: catnext/stack2_block1_deep_2_se_relu/Relu
 */
LIB_HIDDEN void node_model_catnext_stack2_block1_deep_2_se_relu_Relu( const float X[1][48][1][1], float Y[1][48][1][1] )
{
	/*Relu*/
	float *X_ptr = (float*)X;
	float *Y_ptr = (float*)Y;
	for( uint32_t i=0; i<48; i++ )
		Y_ptr[i] = X_ptr[i] > 0 ? X_ptr[i] : 0;

}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block1_deep_2_se_2_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack2_block1_deep_2_se_2_conv_BiasAdd( const float x[1][48][1][1], const float w[192][48][1][1], const float bias[192], float y[1][192][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<48; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack2_block1_deep_2_se_sigmoid/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack2_block1_deep_2_se_sigmoid_Sigmoid( const float X[1][192][1][1], float Y[1][192][1][1] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<1; i2++) {
	for (unsigned i3=0; i3<1; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack2_block1_deep_2_se_out/mul
 */
LIB_HIDDEN void node_model_catnext_stack2_block1_deep_2_se_out_mul( const float A[1][192][48][48], const float B[1][192][1][1], float C[1][192][48][48] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][0][0];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block1_deep_3_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack2_block1_deep_3_conv_Conv2D( const float x[1][192][48][48], const float w[96][192][1][1], const float bias[96], float y[1][96][48][48] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<96; m++) {
		for( int32_t o0=0, i0=0; o0<48; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<48; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=48) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=48) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block2_deep_1_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack2_block2_deep_1_conv_Conv2D( const float x[1][96][48][48], const float w[384][96][1][1], float y[1][384][48][48] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<384; m++) {
		for( int32_t o0=0, i0=0; o0<48; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<48; o1++, i1+=1) {
			y[b][m][o0][o1] = 0;
			for( int32_t c=0; c<96; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=48) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=48) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           BatchNormalization
 * Name: catnext/stack2_block2_deep_1_bn/FusedBatchNormV3
 */
LIB_HIDDEN void node_model_catnext_stack2_block2_deep_1_bn_FusedBatchNormV3( const float X[1][384][48][48], const float scale[384], const float bias[384], const float mean[384], const float var[384], float output[1][384][48][48] )
{
	/* BatchNormalization
	 * epsilon = 1.0009999641624744982e-05
	 * momentum = 0.89999997615814208984
	 */

	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<384; c++ ) {
	for( uint32_t i2=0; i2<48; i2++ ) {
	for( uint32_t i3=0; i3<48; i3++ ) {
		float tmp_X = ( X[b][c][i2][i3] - mean[c] ) / ( var[c] );
		output[b][c][i2][i3] = tmp_X * scale[c] + bias[c];
	}
	}
	}
	}
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack2_block2_deep_1_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack2_block2_deep_1_swish_Sigmoid( const float X[1][384][48][48], float Y[1][384][48][48] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack2_block2_deep_1_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack2_block2_deep_1_swish_mul_1( const float A[1][384][48][48], const float B[1][384][48][48], float C[1][384][48][48] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block2_deep_2_GC_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack2_block2_deep_2_GC_conv_Conv2D( const float x[1][384][48][48], const float w[384][16][3][3], const float bias[384], float y[1][384][48][48] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 24
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 16; // output group size, i.e. maps/group
	uint32_t gi = 16; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<24; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<48; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<48; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=48) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=48) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c-(gi*g)][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack2_block2_deep_2_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack2_block2_deep_2_swish_Sigmoid( const float X[1][384][48][48], float Y[1][384][48][48] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack2_block2_deep_2_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack2_block2_deep_2_swish_mul_1( const float A[1][384][48][48], const float B[1][384][48][48], float C[1][384][48][48] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           GlobalAveragePool
 * Name: catnext/tf.math.reduce_mean_3/Mean
 */
LIB_HIDDEN void node_model_catnext_tf_math_reduce_mean_3_Mean( const float input[1][384][48][48], float output[1][384][1][1] )
{
	/* GlobalAveragePool */
	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<384; c++ ) {
		float dimsum=0.0f;
		for( int32_t d0 = 0; d0<48; d0++ ) {
		for( int32_t d1 = 0; d1<48; d1++ ) {
			dimsum +=  input[b][c][d0][d1];
		}
		}
		output[b][c][0][0] = dimsum / 2304;
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block2_deep_2_se_1_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack2_block2_deep_2_se_1_conv_BiasAdd( const float x[1][384][1][1], const float w[96][384][1][1], const float bias[96], float y[1][96][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<96; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<384; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Relu
 * Name: catnext/stack2_block2_deep_2_se_relu/Relu
 */
LIB_HIDDEN void node_model_catnext_stack2_block2_deep_2_se_relu_Relu( const float X[1][96][1][1], float Y[1][96][1][1] )
{
	/*Relu*/
	float *X_ptr = (float*)X;
	float *Y_ptr = (float*)Y;
	for( uint32_t i=0; i<96; i++ )
		Y_ptr[i] = X_ptr[i] > 0 ? X_ptr[i] : 0;

}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block2_deep_2_se_2_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack2_block2_deep_2_se_2_conv_BiasAdd( const float x[1][96][1][1], const float w[384][96][1][1], const float bias[384], float y[1][384][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<384; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<96; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack2_block2_deep_2_se_sigmoid/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack2_block2_deep_2_se_sigmoid_Sigmoid( const float X[1][384][1][1], float Y[1][384][1][1] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<1; i2++) {
	for (unsigned i3=0; i3<1; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack2_block2_deep_2_se_out/mul
 */
LIB_HIDDEN void node_model_catnext_stack2_block2_deep_2_se_out_mul( const float A[1][384][48][48], const float B[1][384][1][1], float C[1][384][48][48] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][0][0];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block2_deep_3_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack2_block2_deep_3_conv_Conv2D( const float x[1][384][48][48], const float w[96][384][1][1], const float bias[96], float y[1][96][48][48] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<96; m++) {
		for( int32_t o0=0, i0=0; o0<48; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<48; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<384; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=48) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=48) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name: catnext/stack2_block2_add/add
 */
LIB_HIDDEN void node_model_catnext_stack2_block2_add_add( const float A[1][96][48][48], const float B[1][96][48][48], float C[1][96][48][48] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<96; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block3_deep_1_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack2_block3_deep_1_conv_Conv2D( const float x[1][96][48][48], const float w[384][96][1][1], const float bias[384], float y[1][384][48][48] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<384; m++) {
		for( int32_t o0=0, i0=0; o0<48; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<48; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<96; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=48) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=48) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack2_block3_deep_1_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack2_block3_deep_1_swish_Sigmoid( const float X[1][384][48][48], float Y[1][384][48][48] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack2_block3_deep_1_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack2_block3_deep_1_swish_mul_1( const float A[1][384][48][48], const float B[1][384][48][48], float C[1][384][48][48] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block3_deep_2_GC_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack2_block3_deep_2_GC_conv_Conv2D( const float x[1][384][48][48], const float w[384][16][3][3], const float bias[384], float y[1][384][48][48] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 24
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 16; // output group size, i.e. maps/group
	uint32_t gi = 16; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<24; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<48; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<48; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=48) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=48) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c-(gi*g)][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack2_block3_deep_2_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack2_block3_deep_2_swish_Sigmoid( const float X[1][384][48][48], float Y[1][384][48][48] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack2_block3_deep_2_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack2_block3_deep_2_swish_mul_1( const float A[1][384][48][48], const float B[1][384][48][48], float C[1][384][48][48] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           GlobalAveragePool
 * Name: catnext/tf.math.reduce_mean_4/Mean
 */
LIB_HIDDEN void node_model_catnext_tf_math_reduce_mean_4_Mean( const float input[1][384][48][48], float output[1][384][1][1] )
{
	/* GlobalAveragePool */
	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<384; c++ ) {
		float dimsum=0.0f;
		for( int32_t d0 = 0; d0<48; d0++ ) {
		for( int32_t d1 = 0; d1<48; d1++ ) {
			dimsum +=  input[b][c][d0][d1];
		}
		}
		output[b][c][0][0] = dimsum / 2304;
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block3_deep_2_se_1_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack2_block3_deep_2_se_1_conv_BiasAdd( const float x[1][384][1][1], const float w[96][384][1][1], const float bias[96], float y[1][96][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<96; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<384; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Relu
 * Name: catnext/stack2_block3_deep_2_se_relu/Relu
 */
LIB_HIDDEN void node_model_catnext_stack2_block3_deep_2_se_relu_Relu( const float X[1][96][1][1], float Y[1][96][1][1] )
{
	/*Relu*/
	float *X_ptr = (float*)X;
	float *Y_ptr = (float*)Y;
	for( uint32_t i=0; i<96; i++ )
		Y_ptr[i] = X_ptr[i] > 0 ? X_ptr[i] : 0;

}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block3_deep_2_se_2_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack2_block3_deep_2_se_2_conv_BiasAdd( const float x[1][96][1][1], const float w[384][96][1][1], const float bias[384], float y[1][384][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<384; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<96; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack2_block3_deep_2_se_sigmoid/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack2_block3_deep_2_se_sigmoid_Sigmoid( const float X[1][384][1][1], float Y[1][384][1][1] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<1; i2++) {
	for (unsigned i3=0; i3<1; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack2_block3_deep_2_se_out/mul
 */
LIB_HIDDEN void node_model_catnext_stack2_block3_deep_2_se_out_mul( const float A[1][384][48][48], const float B[1][384][1][1], float C[1][384][48][48] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][0][0];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block3_deep_3_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack2_block3_deep_3_conv_Conv2D( const float x[1][384][48][48], const float w[96][384][1][1], const float bias[96], float y[1][96][48][48] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<96; m++) {
		for( int32_t o0=0, i0=0; o0<48; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<48; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<384; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=48) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=48) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name: catnext/stack2_block3_add/add
 */
LIB_HIDDEN void node_model_catnext_stack2_block3_add_add( const float A[1][96][48][48], const float B[1][96][48][48], float C[1][96][48][48] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<96; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block4_deep_1_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack2_block4_deep_1_conv_Conv2D( const float x[1][96][48][48], const float w[384][96][1][1], const float bias[384], float y[1][384][48][48] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<384; m++) {
		for( int32_t o0=0, i0=0; o0<48; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<48; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<96; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=48) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=48) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack2_block4_deep_1_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack2_block4_deep_1_swish_Sigmoid( const float X[1][384][48][48], float Y[1][384][48][48] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack2_block4_deep_1_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack2_block4_deep_1_swish_mul_1( const float A[1][384][48][48], const float B[1][384][48][48], float C[1][384][48][48] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block4_deep_2_GC_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack2_block4_deep_2_GC_conv_Conv2D( const float x[1][384][48][48], const float w[384][16][3][3], const float bias[384], float y[1][384][48][48] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 24
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 16; // output group size, i.e. maps/group
	uint32_t gi = 16; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<24; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<48; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<48; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=48) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=48) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c-(gi*g)][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack2_block4_deep_2_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack2_block4_deep_2_swish_Sigmoid( const float X[1][384][48][48], float Y[1][384][48][48] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack2_block4_deep_2_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack2_block4_deep_2_swish_mul_1( const float A[1][384][48][48], const float B[1][384][48][48], float C[1][384][48][48] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           GlobalAveragePool
 * Name: catnext/tf.math.reduce_mean_5/Mean
 */
LIB_HIDDEN void node_model_catnext_tf_math_reduce_mean_5_Mean( const float input[1][384][48][48], float output[1][384][1][1] )
{
	/* GlobalAveragePool */
	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<384; c++ ) {
		float dimsum=0.0f;
		for( int32_t d0 = 0; d0<48; d0++ ) {
		for( int32_t d1 = 0; d1<48; d1++ ) {
			dimsum +=  input[b][c][d0][d1];
		}
		}
		output[b][c][0][0] = dimsum / 2304;
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block4_deep_2_se_1_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack2_block4_deep_2_se_1_conv_BiasAdd( const float x[1][384][1][1], const float w[96][384][1][1], const float bias[96], float y[1][96][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<96; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<384; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Relu
 * Name: catnext/stack2_block4_deep_2_se_relu/Relu
 */
LIB_HIDDEN void node_model_catnext_stack2_block4_deep_2_se_relu_Relu( const float X[1][96][1][1], float Y[1][96][1][1] )
{
	/*Relu*/
	float *X_ptr = (float*)X;
	float *Y_ptr = (float*)Y;
	for( uint32_t i=0; i<96; i++ )
		Y_ptr[i] = X_ptr[i] > 0 ? X_ptr[i] : 0;

}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block4_deep_2_se_2_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack2_block4_deep_2_se_2_conv_BiasAdd( const float x[1][96][1][1], const float w[384][96][1][1], const float bias[384], float y[1][384][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<384; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<96; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack2_block4_deep_2_se_sigmoid/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack2_block4_deep_2_se_sigmoid_Sigmoid( const float X[1][384][1][1], float Y[1][384][1][1] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<1; i2++) {
	for (unsigned i3=0; i3<1; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack2_block4_deep_2_se_out/mul
 */
LIB_HIDDEN void node_model_catnext_stack2_block4_deep_2_se_out_mul( const float A[1][384][48][48], const float B[1][384][1][1], float C[1][384][48][48] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][0][0];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block4_deep_3_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack2_block4_deep_3_conv_Conv2D( const float x[1][384][48][48], const float w[96][384][1][1], const float bias[96], float y[1][96][48][48] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<96; m++) {
		for( int32_t o0=0, i0=0; o0<48; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<48; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<384; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=48) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=48) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name: catnext/stack2_block4_add/add
 */
LIB_HIDDEN void node_model_catnext_stack2_block4_add_add( const float A[1][96][48][48], const float B[1][96][48][48], float C[1][96][48][48] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<96; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block5_deep_1_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack2_block5_deep_1_conv_Conv2D( const float x[1][96][48][48], const float w[384][96][1][1], const float bias[384], float y[1][384][48][48] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<384; m++) {
		for( int32_t o0=0, i0=0; o0<48; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<48; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<96; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=48) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=48) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack2_block5_deep_1_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack2_block5_deep_1_swish_Sigmoid( const float X[1][384][48][48], float Y[1][384][48][48] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack2_block5_deep_1_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack2_block5_deep_1_swish_mul_1( const float A[1][384][48][48], const float B[1][384][48][48], float C[1][384][48][48] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block5_deep_2_GC_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack2_block5_deep_2_GC_conv_Conv2D( const float x[1][384][48][48], const float w[384][16][3][3], const float bias[384], float y[1][384][48][48] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 24
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 16; // output group size, i.e. maps/group
	uint32_t gi = 16; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<24; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<48; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<48; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=48) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=48) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c-(gi*g)][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack2_block5_deep_2_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack2_block5_deep_2_swish_Sigmoid( const float X[1][384][48][48], float Y[1][384][48][48] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack2_block5_deep_2_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack2_block5_deep_2_swish_mul_1( const float A[1][384][48][48], const float B[1][384][48][48], float C[1][384][48][48] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           GlobalAveragePool
 * Name: catnext/tf.math.reduce_mean_6/Mean
 */
LIB_HIDDEN void node_model_catnext_tf_math_reduce_mean_6_Mean( const float input[1][384][48][48], float output[1][384][1][1] )
{
	/* GlobalAveragePool */
	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<384; c++ ) {
		float dimsum=0.0f;
		for( int32_t d0 = 0; d0<48; d0++ ) {
		for( int32_t d1 = 0; d1<48; d1++ ) {
			dimsum +=  input[b][c][d0][d1];
		}
		}
		output[b][c][0][0] = dimsum / 2304;
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block5_deep_2_se_1_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack2_block5_deep_2_se_1_conv_BiasAdd( const float x[1][384][1][1], const float w[96][384][1][1], const float bias[96], float y[1][96][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<96; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<384; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Relu
 * Name: catnext/stack2_block5_deep_2_se_relu/Relu
 */
LIB_HIDDEN void node_model_catnext_stack2_block5_deep_2_se_relu_Relu( const float X[1][96][1][1], float Y[1][96][1][1] )
{
	/*Relu*/
	float *X_ptr = (float*)X;
	float *Y_ptr = (float*)Y;
	for( uint32_t i=0; i<96; i++ )
		Y_ptr[i] = X_ptr[i] > 0 ? X_ptr[i] : 0;

}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block5_deep_2_se_2_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack2_block5_deep_2_se_2_conv_BiasAdd( const float x[1][96][1][1], const float w[384][96][1][1], const float bias[384], float y[1][384][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<384; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<96; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack2_block5_deep_2_se_sigmoid/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack2_block5_deep_2_se_sigmoid_Sigmoid( const float X[1][384][1][1], float Y[1][384][1][1] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<1; i2++) {
	for (unsigned i3=0; i3<1; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack2_block5_deep_2_se_out/mul
 */
LIB_HIDDEN void node_model_catnext_stack2_block5_deep_2_se_out_mul( const float A[1][384][48][48], const float B[1][384][1][1], float C[1][384][48][48] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][0][0];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block5_deep_3_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack2_block5_deep_3_conv_Conv2D( const float x[1][384][48][48], const float w[96][384][1][1], const float bias[96], float y[1][96][48][48] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<96; m++) {
		for( int32_t o0=0, i0=0; o0<48; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<48; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<384; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=48) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=48) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name: catnext/stack2_block5_add/add
 */
LIB_HIDDEN void node_model_catnext_stack2_block5_add_add( const float A[1][96][48][48], const float B[1][96][48][48], float C[1][96][48][48] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<96; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block6_deep_1_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack2_block6_deep_1_conv_Conv2D( const float x[1][96][48][48], const float w[384][96][1][1], const float bias[384], float y[1][384][48][48] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<384; m++) {
		for( int32_t o0=0, i0=0; o0<48; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<48; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<96; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=48) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=48) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack2_block6_deep_1_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack2_block6_deep_1_swish_Sigmoid( const float X[1][384][48][48], float Y[1][384][48][48] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack2_block6_deep_1_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack2_block6_deep_1_swish_mul_1( const float A[1][384][48][48], const float B[1][384][48][48], float C[1][384][48][48] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block6_deep_2_GC_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack2_block6_deep_2_GC_conv_Conv2D( const float x[1][384][48][48], const float w[384][16][3][3], const float bias[384], float y[1][384][48][48] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 24
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 16; // output group size, i.e. maps/group
	uint32_t gi = 16; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<24; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<48; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<48; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=48) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=48) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c-(gi*g)][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack2_block6_deep_2_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack2_block6_deep_2_swish_Sigmoid( const float X[1][384][48][48], float Y[1][384][48][48] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack2_block6_deep_2_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack2_block6_deep_2_swish_mul_1( const float A[1][384][48][48], const float B[1][384][48][48], float C[1][384][48][48] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           GlobalAveragePool
 * Name: catnext/tf.math.reduce_mean_7/Mean
 */
LIB_HIDDEN void node_model_catnext_tf_math_reduce_mean_7_Mean( const float input[1][384][48][48], float output[1][384][1][1] )
{
	/* GlobalAveragePool */
	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<384; c++ ) {
		float dimsum=0.0f;
		for( int32_t d0 = 0; d0<48; d0++ ) {
		for( int32_t d1 = 0; d1<48; d1++ ) {
			dimsum +=  input[b][c][d0][d1];
		}
		}
		output[b][c][0][0] = dimsum / 2304;
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block6_deep_2_se_1_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack2_block6_deep_2_se_1_conv_BiasAdd( const float x[1][384][1][1], const float w[96][384][1][1], const float bias[96], float y[1][96][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<96; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<384; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Relu
 * Name: catnext/stack2_block6_deep_2_se_relu/Relu
 */
LIB_HIDDEN void node_model_catnext_stack2_block6_deep_2_se_relu_Relu( const float X[1][96][1][1], float Y[1][96][1][1] )
{
	/*Relu*/
	float *X_ptr = (float*)X;
	float *Y_ptr = (float*)Y;
	for( uint32_t i=0; i<96; i++ )
		Y_ptr[i] = X_ptr[i] > 0 ? X_ptr[i] : 0;

}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block6_deep_2_se_2_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack2_block6_deep_2_se_2_conv_BiasAdd( const float x[1][96][1][1], const float w[384][96][1][1], const float bias[384], float y[1][384][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<384; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<96; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack2_block6_deep_2_se_sigmoid/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack2_block6_deep_2_se_sigmoid_Sigmoid( const float X[1][384][1][1], float Y[1][384][1][1] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<1; i2++) {
	for (unsigned i3=0; i3<1; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack2_block6_deep_2_se_out/mul
 */
LIB_HIDDEN void node_model_catnext_stack2_block6_deep_2_se_out_mul( const float A[1][384][48][48], const float B[1][384][1][1], float C[1][384][48][48] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][0][0];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack2_block6_deep_3_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack2_block6_deep_3_conv_Conv2D( const float x[1][384][48][48], const float w[96][384][1][1], const float bias[96], float y[1][96][48][48] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<96; m++) {
		for( int32_t o0=0, i0=0; o0<48; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<48; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<384; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=48) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=48) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name: catnext/stack2_block6_add/add
 */
LIB_HIDDEN void node_model_catnext_stack2_block6_add_add( const float A[1][96][48][48], const float B[1][96][48][48], float C[1][96][48][48] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<96; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block1_deep_1_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block1_deep_1_conv_Conv2D( const float x[1][96][48][48], const float w[384][96][1][1], const float bias[384], float y[1][384][48][48] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<384; m++) {
		for( int32_t o0=0, i0=0; o0<48; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<48; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<96; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=48) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=48) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block1_deep_1_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block1_deep_1_swish_Sigmoid( const float X[1][384][48][48], float Y[1][384][48][48] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block1_deep_1_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack3_block1_deep_1_swish_mul_1( const float A[1][384][48][48], const float B[1][384][48][48], float C[1][384][48][48] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<48; i2++) {
	for (unsigned i3=0; i3<48; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block1_deep_2_GC_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block1_deep_2_GC_conv_Conv2D( const float x[1][384][48][48], const float w[384][16][3][3], const float bias[384], float y[1][384][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 24
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 2 2 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 16; // output group size, i.e. maps/group
	uint32_t gi = 16; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<24; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<24; o0++, i0+=2) {
		for( int32_t o1=0, i1=-1; o1<24; o1++, i1+=2) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=48) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=48) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c-(gi*g)][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block1_deep_2_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block1_deep_2_swish_Sigmoid( const float X[1][384][24][24], float Y[1][384][24][24] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block1_deep_2_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack3_block1_deep_2_swish_mul_1( const float A[1][384][24][24], const float B[1][384][24][24], float C[1][384][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           GlobalAveragePool
 * Name: catnext/tf.math.reduce_mean_8/Mean
 */
LIB_HIDDEN void node_model_catnext_tf_math_reduce_mean_8_Mean( const float input[1][384][24][24], float output[1][384][1][1] )
{
	/* GlobalAveragePool */
	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<384; c++ ) {
		float dimsum=0.0f;
		for( int32_t d0 = 0; d0<24; d0++ ) {
		for( int32_t d1 = 0; d1<24; d1++ ) {
			dimsum +=  input[b][c][d0][d1];
		}
		}
		output[b][c][0][0] = dimsum / 576;
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block1_deep_2_se_1_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack3_block1_deep_2_se_1_conv_BiasAdd( const float x[1][384][1][1], const float w[96][384][1][1], const float bias[96], float y[1][96][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<96; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<384; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Relu
 * Name: catnext/stack3_block1_deep_2_se_relu/Relu
 */
LIB_HIDDEN void node_model_catnext_stack3_block1_deep_2_se_relu_Relu( const float X[1][96][1][1], float Y[1][96][1][1] )
{
	/*Relu*/
	float *X_ptr = (float*)X;
	float *Y_ptr = (float*)Y;
	for( uint32_t i=0; i<96; i++ )
		Y_ptr[i] = X_ptr[i] > 0 ? X_ptr[i] : 0;

}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block1_deep_2_se_2_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack3_block1_deep_2_se_2_conv_BiasAdd( const float x[1][96][1][1], const float w[384][96][1][1], const float bias[384], float y[1][384][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<384; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<96; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block1_deep_2_se_sigmoid/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block1_deep_2_se_sigmoid_Sigmoid( const float X[1][384][1][1], float Y[1][384][1][1] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<1; i2++) {
	for (unsigned i3=0; i3<1; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block1_deep_2_se_out/mul
 */
LIB_HIDDEN void node_model_catnext_stack3_block1_deep_2_se_out_mul( const float A[1][384][24][24], const float B[1][384][1][1], float C[1][384][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<384; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][0][0];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block1_deep_3_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block1_deep_3_conv_Conv2D( const float x[1][384][24][24], const float w[192][384][1][1], const float bias[192], float y[1][192][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<384; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block2_deep_1_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block2_deep_1_conv_Conv2D( const float x[1][192][24][24], const float w[768][192][1][1], float y[1][768][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<768; m++) {
		for( int32_t o0=0, i0=0; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = 0;
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           BatchNormalization
 * Name: catnext/stack3_block2_deep_1_bn/FusedBatchNormV3
 */
LIB_HIDDEN void node_model_catnext_stack3_block2_deep_1_bn_FusedBatchNormV3( const float X[1][768][24][24], const float scale[768], const float bias[768], const float mean[768], const float var[768], float output[1][768][24][24] )
{
	/* BatchNormalization
	 * epsilon = 1.0009999641624744982e-05
	 * momentum = 0.89999997615814208984
	 */

	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<768; c++ ) {
	for( uint32_t i2=0; i2<24; i2++ ) {
	for( uint32_t i3=0; i3<24; i3++ ) {
		float tmp_X = ( X[b][c][i2][i3] - mean[c] ) / ( var[c] );
		output[b][c][i2][i3] = tmp_X * scale[c] + bias[c];
	}
	}
	}
	}
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block2_deep_1_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block2_deep_1_swish_Sigmoid( const float X[1][768][24][24], float Y[1][768][24][24] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block2_deep_1_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack3_block2_deep_1_swish_mul_1( const float A[1][768][24][24], const float B[1][768][24][24], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block2_deep_2_GC_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block2_deep_2_GC_conv_Conv2D( const float x[1][768][24][24], const float w[768][16][3][3], const float bias[768], float y[1][768][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 48
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 16; // output group size, i.e. maps/group
	uint32_t gi = 16; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<48; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c-(gi*g)][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block2_deep_2_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block2_deep_2_swish_Sigmoid( const float X[1][768][24][24], float Y[1][768][24][24] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block2_deep_2_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack3_block2_deep_2_swish_mul_1( const float A[1][768][24][24], const float B[1][768][24][24], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           GlobalAveragePool
 * Name: catnext/tf.math.reduce_mean_9/Mean
 */
LIB_HIDDEN void node_model_catnext_tf_math_reduce_mean_9_Mean( const float input[1][768][24][24], float output[1][768][1][1] )
{
	/* GlobalAveragePool */
	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<768; c++ ) {
		float dimsum=0.0f;
		for( int32_t d0 = 0; d0<24; d0++ ) {
		for( int32_t d1 = 0; d1<24; d1++ ) {
			dimsum +=  input[b][c][d0][d1];
		}
		}
		output[b][c][0][0] = dimsum / 576;
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block2_deep_2_se_1_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack3_block2_deep_2_se_1_conv_BiasAdd( const float x[1][768][1][1], const float w[192][768][1][1], const float bias[192], float y[1][192][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<768; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Relu
 * Name: catnext/stack3_block2_deep_2_se_relu/Relu
 */
LIB_HIDDEN void node_model_catnext_stack3_block2_deep_2_se_relu_Relu( const float X[1][192][1][1], float Y[1][192][1][1] )
{
	/*Relu*/
	float *X_ptr = (float*)X;
	float *Y_ptr = (float*)Y;
	for( uint32_t i=0; i<192; i++ )
		Y_ptr[i] = X_ptr[i] > 0 ? X_ptr[i] : 0;

}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block2_deep_2_se_2_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack3_block2_deep_2_se_2_conv_BiasAdd( const float x[1][192][1][1], const float w[768][192][1][1], const float bias[768], float y[1][768][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<768; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block2_deep_2_se_sigmoid/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block2_deep_2_se_sigmoid_Sigmoid( const float X[1][768][1][1], float Y[1][768][1][1] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<1; i2++) {
	for (unsigned i3=0; i3<1; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block2_deep_2_se_out/mul
 */
LIB_HIDDEN void node_model_catnext_stack3_block2_deep_2_se_out_mul( const float A[1][768][24][24], const float B[1][768][1][1], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][0][0];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block2_deep_3_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block2_deep_3_conv_Conv2D( const float x[1][768][24][24], const float w[192][768][1][1], const float bias[192], float y[1][192][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<768; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name: catnext/stack3_block2_add/add
 */
LIB_HIDDEN void node_model_catnext_stack3_block2_add_add( const float A[1][192][24][24], const float B[1][192][24][24], float C[1][192][24][24] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block3_deep_1_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block3_deep_1_conv_Conv2D( const float x[1][192][24][24], const float w[768][192][1][1], const float bias[768], float y[1][768][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<768; m++) {
		for( int32_t o0=0, i0=0; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block3_deep_1_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block3_deep_1_swish_Sigmoid( const float X[1][768][24][24], float Y[1][768][24][24] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block3_deep_1_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack3_block3_deep_1_swish_mul_1( const float A[1][768][24][24], const float B[1][768][24][24], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block3_deep_2_GC_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block3_deep_2_GC_conv_Conv2D( const float x[1][768][24][24], const float w[768][16][3][3], const float bias[768], float y[1][768][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 48
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 16; // output group size, i.e. maps/group
	uint32_t gi = 16; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<48; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c-(gi*g)][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block3_deep_2_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block3_deep_2_swish_Sigmoid( const float X[1][768][24][24], float Y[1][768][24][24] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block3_deep_2_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack3_block3_deep_2_swish_mul_1( const float A[1][768][24][24], const float B[1][768][24][24], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           GlobalAveragePool
 * Name: catnext/tf.math.reduce_mean_10/Mean
 */
LIB_HIDDEN void node_model_catnext_tf_math_reduce_mean_10_Mean( const float input[1][768][24][24], float output[1][768][1][1] )
{
	/* GlobalAveragePool */
	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<768; c++ ) {
		float dimsum=0.0f;
		for( int32_t d0 = 0; d0<24; d0++ ) {
		for( int32_t d1 = 0; d1<24; d1++ ) {
			dimsum +=  input[b][c][d0][d1];
		}
		}
		output[b][c][0][0] = dimsum / 576;
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block3_deep_2_se_1_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack3_block3_deep_2_se_1_conv_BiasAdd( const float x[1][768][1][1], const float w[192][768][1][1], const float bias[192], float y[1][192][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<768; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Relu
 * Name: catnext/stack3_block3_deep_2_se_relu/Relu
 */
LIB_HIDDEN void node_model_catnext_stack3_block3_deep_2_se_relu_Relu( const float X[1][192][1][1], float Y[1][192][1][1] )
{
	/*Relu*/
	float *X_ptr = (float*)X;
	float *Y_ptr = (float*)Y;
	for( uint32_t i=0; i<192; i++ )
		Y_ptr[i] = X_ptr[i] > 0 ? X_ptr[i] : 0;

}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block3_deep_2_se_2_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack3_block3_deep_2_se_2_conv_BiasAdd( const float x[1][192][1][1], const float w[768][192][1][1], const float bias[768], float y[1][768][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<768; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block3_deep_2_se_sigmoid/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block3_deep_2_se_sigmoid_Sigmoid( const float X[1][768][1][1], float Y[1][768][1][1] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<1; i2++) {
	for (unsigned i3=0; i3<1; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block3_deep_2_se_out/mul
 */
LIB_HIDDEN void node_model_catnext_stack3_block3_deep_2_se_out_mul( const float A[1][768][24][24], const float B[1][768][1][1], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][0][0];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block3_deep_3_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block3_deep_3_conv_Conv2D( const float x[1][768][24][24], const float w[192][768][1][1], const float bias[192], float y[1][192][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<768; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name: catnext/stack3_block3_add/add
 */
LIB_HIDDEN void node_model_catnext_stack3_block3_add_add( const float A[1][192][24][24], const float B[1][192][24][24], float C[1][192][24][24] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block4_deep_1_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block4_deep_1_conv_Conv2D( const float x[1][192][24][24], const float w[768][192][1][1], const float bias[768], float y[1][768][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<768; m++) {
		for( int32_t o0=0, i0=0; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block4_deep_1_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block4_deep_1_swish_Sigmoid( const float X[1][768][24][24], float Y[1][768][24][24] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block4_deep_1_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack3_block4_deep_1_swish_mul_1( const float A[1][768][24][24], const float B[1][768][24][24], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block4_deep_2_GC_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block4_deep_2_GC_conv_Conv2D( const float x[1][768][24][24], const float w[768][16][3][3], const float bias[768], float y[1][768][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 48
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 16; // output group size, i.e. maps/group
	uint32_t gi = 16; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<48; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c-(gi*g)][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block4_deep_2_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block4_deep_2_swish_Sigmoid( const float X[1][768][24][24], float Y[1][768][24][24] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block4_deep_2_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack3_block4_deep_2_swish_mul_1( const float A[1][768][24][24], const float B[1][768][24][24], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           GlobalAveragePool
 * Name: catnext/tf.math.reduce_mean_11/Mean
 */
LIB_HIDDEN void node_model_catnext_tf_math_reduce_mean_11_Mean( const float input[1][768][24][24], float output[1][768][1][1] )
{
	/* GlobalAveragePool */
	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<768; c++ ) {
		float dimsum=0.0f;
		for( int32_t d0 = 0; d0<24; d0++ ) {
		for( int32_t d1 = 0; d1<24; d1++ ) {
			dimsum +=  input[b][c][d0][d1];
		}
		}
		output[b][c][0][0] = dimsum / 576;
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block4_deep_2_se_1_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack3_block4_deep_2_se_1_conv_BiasAdd( const float x[1][768][1][1], const float w[192][768][1][1], const float bias[192], float y[1][192][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<768; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Relu
 * Name: catnext/stack3_block4_deep_2_se_relu/Relu
 */
LIB_HIDDEN void node_model_catnext_stack3_block4_deep_2_se_relu_Relu( const float X[1][192][1][1], float Y[1][192][1][1] )
{
	/*Relu*/
	float *X_ptr = (float*)X;
	float *Y_ptr = (float*)Y;
	for( uint32_t i=0; i<192; i++ )
		Y_ptr[i] = X_ptr[i] > 0 ? X_ptr[i] : 0;

}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block4_deep_2_se_2_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack3_block4_deep_2_se_2_conv_BiasAdd( const float x[1][192][1][1], const float w[768][192][1][1], const float bias[768], float y[1][768][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<768; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block4_deep_2_se_sigmoid/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block4_deep_2_se_sigmoid_Sigmoid( const float X[1][768][1][1], float Y[1][768][1][1] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<1; i2++) {
	for (unsigned i3=0; i3<1; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block4_deep_2_se_out/mul
 */
LIB_HIDDEN void node_model_catnext_stack3_block4_deep_2_se_out_mul( const float A[1][768][24][24], const float B[1][768][1][1], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][0][0];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block4_deep_3_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block4_deep_3_conv_Conv2D( const float x[1][768][24][24], const float w[192][768][1][1], const float bias[192], float y[1][192][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<768; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name: catnext/stack3_block4_add/add
 */
LIB_HIDDEN void node_model_catnext_stack3_block4_add_add( const float A[1][192][24][24], const float B[1][192][24][24], float C[1][192][24][24] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block5_deep_1_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block5_deep_1_conv_Conv2D( const float x[1][192][24][24], const float w[768][192][1][1], const float bias[768], float y[1][768][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<768; m++) {
		for( int32_t o0=0, i0=0; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block5_deep_1_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block5_deep_1_swish_Sigmoid( const float X[1][768][24][24], float Y[1][768][24][24] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block5_deep_1_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack3_block5_deep_1_swish_mul_1( const float A[1][768][24][24], const float B[1][768][24][24], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block5_deep_2_GC_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block5_deep_2_GC_conv_Conv2D( const float x[1][768][24][24], const float w[768][16][3][3], const float bias[768], float y[1][768][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 48
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 16; // output group size, i.e. maps/group
	uint32_t gi = 16; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<48; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c-(gi*g)][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block5_deep_2_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block5_deep_2_swish_Sigmoid( const float X[1][768][24][24], float Y[1][768][24][24] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block5_deep_2_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack3_block5_deep_2_swish_mul_1( const float A[1][768][24][24], const float B[1][768][24][24], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           GlobalAveragePool
 * Name: catnext/tf.math.reduce_mean_12/Mean
 */
LIB_HIDDEN void node_model_catnext_tf_math_reduce_mean_12_Mean( const float input[1][768][24][24], float output[1][768][1][1] )
{
	/* GlobalAveragePool */
	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<768; c++ ) {
		float dimsum=0.0f;
		for( int32_t d0 = 0; d0<24; d0++ ) {
		for( int32_t d1 = 0; d1<24; d1++ ) {
			dimsum +=  input[b][c][d0][d1];
		}
		}
		output[b][c][0][0] = dimsum / 576;
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block5_deep_2_se_1_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack3_block5_deep_2_se_1_conv_BiasAdd( const float x[1][768][1][1], const float w[192][768][1][1], const float bias[192], float y[1][192][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<768; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Relu
 * Name: catnext/stack3_block5_deep_2_se_relu/Relu
 */
LIB_HIDDEN void node_model_catnext_stack3_block5_deep_2_se_relu_Relu( const float X[1][192][1][1], float Y[1][192][1][1] )
{
	/*Relu*/
	float *X_ptr = (float*)X;
	float *Y_ptr = (float*)Y;
	for( uint32_t i=0; i<192; i++ )
		Y_ptr[i] = X_ptr[i] > 0 ? X_ptr[i] : 0;

}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block5_deep_2_se_2_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack3_block5_deep_2_se_2_conv_BiasAdd( const float x[1][192][1][1], const float w[768][192][1][1], const float bias[768], float y[1][768][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<768; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block5_deep_2_se_sigmoid/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block5_deep_2_se_sigmoid_Sigmoid( const float X[1][768][1][1], float Y[1][768][1][1] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<1; i2++) {
	for (unsigned i3=0; i3<1; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block5_deep_2_se_out/mul
 */
LIB_HIDDEN void node_model_catnext_stack3_block5_deep_2_se_out_mul( const float A[1][768][24][24], const float B[1][768][1][1], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][0][0];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block5_deep_3_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block5_deep_3_conv_Conv2D( const float x[1][768][24][24], const float w[192][768][1][1], const float bias[192], float y[1][192][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<768; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name: catnext/stack3_block5_add/add
 */
LIB_HIDDEN void node_model_catnext_stack3_block5_add_add( const float A[1][192][24][24], const float B[1][192][24][24], float C[1][192][24][24] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block6_deep_1_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block6_deep_1_conv_Conv2D( const float x[1][192][24][24], const float w[768][192][1][1], const float bias[768], float y[1][768][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<768; m++) {
		for( int32_t o0=0, i0=0; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block6_deep_1_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block6_deep_1_swish_Sigmoid( const float X[1][768][24][24], float Y[1][768][24][24] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block6_deep_1_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack3_block6_deep_1_swish_mul_1( const float A[1][768][24][24], const float B[1][768][24][24], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block6_deep_2_GC_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block6_deep_2_GC_conv_Conv2D( const float x[1][768][24][24], const float w[768][16][3][3], const float bias[768], float y[1][768][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 48
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 16; // output group size, i.e. maps/group
	uint32_t gi = 16; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<48; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c-(gi*g)][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block6_deep_2_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block6_deep_2_swish_Sigmoid( const float X[1][768][24][24], float Y[1][768][24][24] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block6_deep_2_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack3_block6_deep_2_swish_mul_1( const float A[1][768][24][24], const float B[1][768][24][24], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           GlobalAveragePool
 * Name: catnext/tf.math.reduce_mean_13/Mean
 */
LIB_HIDDEN void node_model_catnext_tf_math_reduce_mean_13_Mean( const float input[1][768][24][24], float output[1][768][1][1] )
{
	/* GlobalAveragePool */
	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<768; c++ ) {
		float dimsum=0.0f;
		for( int32_t d0 = 0; d0<24; d0++ ) {
		for( int32_t d1 = 0; d1<24; d1++ ) {
			dimsum +=  input[b][c][d0][d1];
		}
		}
		output[b][c][0][0] = dimsum / 576;
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block6_deep_2_se_1_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack3_block6_deep_2_se_1_conv_BiasAdd( const float x[1][768][1][1], const float w[192][768][1][1], const float bias[192], float y[1][192][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<768; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Relu
 * Name: catnext/stack3_block6_deep_2_se_relu/Relu
 */
LIB_HIDDEN void node_model_catnext_stack3_block6_deep_2_se_relu_Relu( const float X[1][192][1][1], float Y[1][192][1][1] )
{
	/*Relu*/
	float *X_ptr = (float*)X;
	float *Y_ptr = (float*)Y;
	for( uint32_t i=0; i<192; i++ )
		Y_ptr[i] = X_ptr[i] > 0 ? X_ptr[i] : 0;

}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block6_deep_2_se_2_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack3_block6_deep_2_se_2_conv_BiasAdd( const float x[1][192][1][1], const float w[768][192][1][1], const float bias[768], float y[1][768][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<768; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block6_deep_2_se_sigmoid/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block6_deep_2_se_sigmoid_Sigmoid( const float X[1][768][1][1], float Y[1][768][1][1] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<1; i2++) {
	for (unsigned i3=0; i3<1; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block6_deep_2_se_out/mul
 */
LIB_HIDDEN void node_model_catnext_stack3_block6_deep_2_se_out_mul( const float A[1][768][24][24], const float B[1][768][1][1], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][0][0];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block6_deep_3_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block6_deep_3_conv_Conv2D( const float x[1][768][24][24], const float w[192][768][1][1], const float bias[192], float y[1][192][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<768; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name: catnext/stack3_block6_add/add
 */
LIB_HIDDEN void node_model_catnext_stack3_block6_add_add( const float A[1][192][24][24], const float B[1][192][24][24], float C[1][192][24][24] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block7_deep_1_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block7_deep_1_conv_Conv2D( const float x[1][192][24][24], const float w[768][192][1][1], const float bias[768], float y[1][768][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<768; m++) {
		for( int32_t o0=0, i0=0; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block7_deep_1_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block7_deep_1_swish_Sigmoid( const float X[1][768][24][24], float Y[1][768][24][24] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block7_deep_1_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack3_block7_deep_1_swish_mul_1( const float A[1][768][24][24], const float B[1][768][24][24], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block7_deep_2_GC_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block7_deep_2_GC_conv_Conv2D( const float x[1][768][24][24], const float w[768][16][3][3], const float bias[768], float y[1][768][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 48
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 16; // output group size, i.e. maps/group
	uint32_t gi = 16; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<48; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c-(gi*g)][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block7_deep_2_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block7_deep_2_swish_Sigmoid( const float X[1][768][24][24], float Y[1][768][24][24] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block7_deep_2_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack3_block7_deep_2_swish_mul_1( const float A[1][768][24][24], const float B[1][768][24][24], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           GlobalAveragePool
 * Name: catnext/tf.math.reduce_mean_14/Mean
 */
LIB_HIDDEN void node_model_catnext_tf_math_reduce_mean_14_Mean( const float input[1][768][24][24], float output[1][768][1][1] )
{
	/* GlobalAveragePool */
	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<768; c++ ) {
		float dimsum=0.0f;
		for( int32_t d0 = 0; d0<24; d0++ ) {
		for( int32_t d1 = 0; d1<24; d1++ ) {
			dimsum +=  input[b][c][d0][d1];
		}
		}
		output[b][c][0][0] = dimsum / 576;
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block7_deep_2_se_1_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack3_block7_deep_2_se_1_conv_BiasAdd( const float x[1][768][1][1], const float w[192][768][1][1], const float bias[192], float y[1][192][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<768; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Relu
 * Name: catnext/stack3_block7_deep_2_se_relu/Relu
 */
LIB_HIDDEN void node_model_catnext_stack3_block7_deep_2_se_relu_Relu( const float X[1][192][1][1], float Y[1][192][1][1] )
{
	/*Relu*/
	float *X_ptr = (float*)X;
	float *Y_ptr = (float*)Y;
	for( uint32_t i=0; i<192; i++ )
		Y_ptr[i] = X_ptr[i] > 0 ? X_ptr[i] : 0;

}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block7_deep_2_se_2_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack3_block7_deep_2_se_2_conv_BiasAdd( const float x[1][192][1][1], const float w[768][192][1][1], const float bias[768], float y[1][768][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<768; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block7_deep_2_se_sigmoid/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block7_deep_2_se_sigmoid_Sigmoid( const float X[1][768][1][1], float Y[1][768][1][1] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<1; i2++) {
	for (unsigned i3=0; i3<1; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block7_deep_2_se_out/mul
 */
LIB_HIDDEN void node_model_catnext_stack3_block7_deep_2_se_out_mul( const float A[1][768][24][24], const float B[1][768][1][1], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][0][0];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block7_deep_3_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block7_deep_3_conv_Conv2D( const float x[1][768][24][24], const float w[192][768][1][1], const float bias[192], float y[1][192][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<768; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name: catnext/stack3_block7_add/add
 */
LIB_HIDDEN void node_model_catnext_stack3_block7_add_add( const float A[1][192][24][24], const float B[1][192][24][24], float C[1][192][24][24] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block8_deep_1_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block8_deep_1_conv_Conv2D( const float x[1][192][24][24], const float w[768][192][1][1], const float bias[768], float y[1][768][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<768; m++) {
		for( int32_t o0=0, i0=0; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block8_deep_1_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block8_deep_1_swish_Sigmoid( const float X[1][768][24][24], float Y[1][768][24][24] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block8_deep_1_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack3_block8_deep_1_swish_mul_1( const float A[1][768][24][24], const float B[1][768][24][24], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block8_deep_2_GC_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block8_deep_2_GC_conv_Conv2D( const float x[1][768][24][24], const float w[768][16][3][3], const float bias[768], float y[1][768][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 48
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 16; // output group size, i.e. maps/group
	uint32_t gi = 16; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<48; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c-(gi*g)][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block8_deep_2_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block8_deep_2_swish_Sigmoid( const float X[1][768][24][24], float Y[1][768][24][24] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block8_deep_2_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack3_block8_deep_2_swish_mul_1( const float A[1][768][24][24], const float B[1][768][24][24], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           GlobalAveragePool
 * Name: catnext/tf.math.reduce_mean_15/Mean
 */
LIB_HIDDEN void node_model_catnext_tf_math_reduce_mean_15_Mean( const float input[1][768][24][24], float output[1][768][1][1] )
{
	/* GlobalAveragePool */
	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<768; c++ ) {
		float dimsum=0.0f;
		for( int32_t d0 = 0; d0<24; d0++ ) {
		for( int32_t d1 = 0; d1<24; d1++ ) {
			dimsum +=  input[b][c][d0][d1];
		}
		}
		output[b][c][0][0] = dimsum / 576;
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block8_deep_2_se_1_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack3_block8_deep_2_se_1_conv_BiasAdd( const float x[1][768][1][1], const float w[192][768][1][1], const float bias[192], float y[1][192][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<768; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Relu
 * Name: catnext/stack3_block8_deep_2_se_relu/Relu
 */
LIB_HIDDEN void node_model_catnext_stack3_block8_deep_2_se_relu_Relu( const float X[1][192][1][1], float Y[1][192][1][1] )
{
	/*Relu*/
	float *X_ptr = (float*)X;
	float *Y_ptr = (float*)Y;
	for( uint32_t i=0; i<192; i++ )
		Y_ptr[i] = X_ptr[i] > 0 ? X_ptr[i] : 0;

}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block8_deep_2_se_2_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack3_block8_deep_2_se_2_conv_BiasAdd( const float x[1][192][1][1], const float w[768][192][1][1], const float bias[768], float y[1][768][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<768; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block8_deep_2_se_sigmoid/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block8_deep_2_se_sigmoid_Sigmoid( const float X[1][768][1][1], float Y[1][768][1][1] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<1; i2++) {
	for (unsigned i3=0; i3<1; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block8_deep_2_se_out/mul
 */
LIB_HIDDEN void node_model_catnext_stack3_block8_deep_2_se_out_mul( const float A[1][768][24][24], const float B[1][768][1][1], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][0][0];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block8_deep_3_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block8_deep_3_conv_Conv2D( const float x[1][768][24][24], const float w[192][768][1][1], const float bias[192], float y[1][192][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<768; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name: catnext/stack3_block8_add/add
 */
LIB_HIDDEN void node_model_catnext_stack3_block8_add_add( const float A[1][192][24][24], const float B[1][192][24][24], float C[1][192][24][24] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block9_deep_1_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block9_deep_1_conv_Conv2D( const float x[1][192][24][24], const float w[768][192][1][1], const float bias[768], float y[1][768][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<768; m++) {
		for( int32_t o0=0, i0=0; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block9_deep_1_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block9_deep_1_swish_Sigmoid( const float X[1][768][24][24], float Y[1][768][24][24] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block9_deep_1_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack3_block9_deep_1_swish_mul_1( const float A[1][768][24][24], const float B[1][768][24][24], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block9_deep_2_GC_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block9_deep_2_GC_conv_Conv2D( const float x[1][768][24][24], const float w[768][16][3][3], const float bias[768], float y[1][768][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 48
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 16; // output group size, i.e. maps/group
	uint32_t gi = 16; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<48; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c-(gi*g)][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block9_deep_2_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block9_deep_2_swish_Sigmoid( const float X[1][768][24][24], float Y[1][768][24][24] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block9_deep_2_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack3_block9_deep_2_swish_mul_1( const float A[1][768][24][24], const float B[1][768][24][24], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           GlobalAveragePool
 * Name: catnext/tf.math.reduce_mean_16/Mean
 */
LIB_HIDDEN void node_model_catnext_tf_math_reduce_mean_16_Mean( const float input[1][768][24][24], float output[1][768][1][1] )
{
	/* GlobalAveragePool */
	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<768; c++ ) {
		float dimsum=0.0f;
		for( int32_t d0 = 0; d0<24; d0++ ) {
		for( int32_t d1 = 0; d1<24; d1++ ) {
			dimsum +=  input[b][c][d0][d1];
		}
		}
		output[b][c][0][0] = dimsum / 576;
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block9_deep_2_se_1_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack3_block9_deep_2_se_1_conv_BiasAdd( const float x[1][768][1][1], const float w[192][768][1][1], const float bias[192], float y[1][192][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<768; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Relu
 * Name: catnext/stack3_block9_deep_2_se_relu/Relu
 */
LIB_HIDDEN void node_model_catnext_stack3_block9_deep_2_se_relu_Relu( const float X[1][192][1][1], float Y[1][192][1][1] )
{
	/*Relu*/
	float *X_ptr = (float*)X;
	float *Y_ptr = (float*)Y;
	for( uint32_t i=0; i<192; i++ )
		Y_ptr[i] = X_ptr[i] > 0 ? X_ptr[i] : 0;

}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block9_deep_2_se_2_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack3_block9_deep_2_se_2_conv_BiasAdd( const float x[1][192][1][1], const float w[768][192][1][1], const float bias[768], float y[1][768][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<768; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block9_deep_2_se_sigmoid/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block9_deep_2_se_sigmoid_Sigmoid( const float X[1][768][1][1], float Y[1][768][1][1] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<1; i2++) {
	for (unsigned i3=0; i3<1; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block9_deep_2_se_out/mul
 */
LIB_HIDDEN void node_model_catnext_stack3_block9_deep_2_se_out_mul( const float A[1][768][24][24], const float B[1][768][1][1], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][0][0];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block9_deep_3_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block9_deep_3_conv_Conv2D( const float x[1][768][24][24], const float w[192][768][1][1], const float bias[192], float y[1][192][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<768; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name: catnext/stack3_block9_add/add
 */
LIB_HIDDEN void node_model_catnext_stack3_block9_add_add( const float A[1][192][24][24], const float B[1][192][24][24], float C[1][192][24][24] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block10_deep_1_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block10_deep_1_conv_Conv2D( const float x[1][192][24][24], const float w[768][192][1][1], const float bias[768], float y[1][768][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<768; m++) {
		for( int32_t o0=0, i0=0; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block10_deep_1_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block10_deep_1_swish_Sigmoid( const float X[1][768][24][24], float Y[1][768][24][24] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block10_deep_1_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack3_block10_deep_1_swish_mul_1( const float A[1][768][24][24], const float B[1][768][24][24], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block10_deep_2_GC_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block10_deep_2_GC_conv_Conv2D( const float x[1][768][24][24], const float w[768][16][3][3], const float bias[768], float y[1][768][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 48
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 16; // output group size, i.e. maps/group
	uint32_t gi = 16; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<48; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c-(gi*g)][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block10_deep_2_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block10_deep_2_swish_Sigmoid( const float X[1][768][24][24], float Y[1][768][24][24] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block10_deep_2_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack3_block10_deep_2_swish_mul_1( const float A[1][768][24][24], const float B[1][768][24][24], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           GlobalAveragePool
 * Name: catnext/tf.math.reduce_mean_17/Mean
 */
LIB_HIDDEN void node_model_catnext_tf_math_reduce_mean_17_Mean( const float input[1][768][24][24], float output[1][768][1][1] )
{
	/* GlobalAveragePool */
	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<768; c++ ) {
		float dimsum=0.0f;
		for( int32_t d0 = 0; d0<24; d0++ ) {
		for( int32_t d1 = 0; d1<24; d1++ ) {
			dimsum +=  input[b][c][d0][d1];
		}
		}
		output[b][c][0][0] = dimsum / 576;
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block10_deep_2_se_1_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack3_block10_deep_2_se_1_conv_BiasAdd( const float x[1][768][1][1], const float w[192][768][1][1], const float bias[192], float y[1][192][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<768; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Relu
 * Name: catnext/stack3_block10_deep_2_se_relu/Relu
 */
LIB_HIDDEN void node_model_catnext_stack3_block10_deep_2_se_relu_Relu( const float X[1][192][1][1], float Y[1][192][1][1] )
{
	/*Relu*/
	float *X_ptr = (float*)X;
	float *Y_ptr = (float*)Y;
	for( uint32_t i=0; i<192; i++ )
		Y_ptr[i] = X_ptr[i] > 0 ? X_ptr[i] : 0;

}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block10_deep_2_se_2_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack3_block10_deep_2_se_2_conv_BiasAdd( const float x[1][192][1][1], const float w[768][192][1][1], const float bias[768], float y[1][768][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<768; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block10_deep_2_se_sigmoid/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block10_deep_2_se_sigmoid_Sigmoid( const float X[1][768][1][1], float Y[1][768][1][1] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<1; i2++) {
	for (unsigned i3=0; i3<1; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block10_deep_2_se_out/mul
 */
LIB_HIDDEN void node_model_catnext_stack3_block10_deep_2_se_out_mul( const float A[1][768][24][24], const float B[1][768][1][1], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][0][0];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block10_deep_3_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block10_deep_3_conv_Conv2D( const float x[1][768][24][24], const float w[192][768][1][1], const float bias[192], float y[1][192][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<768; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name: catnext/stack3_block10_add/add
 */
LIB_HIDDEN void node_model_catnext_stack3_block10_add_add( const float A[1][192][24][24], const float B[1][192][24][24], float C[1][192][24][24] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block11_deep_1_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block11_deep_1_conv_Conv2D( const float x[1][192][24][24], const float w[768][192][1][1], const float bias[768], float y[1][768][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<768; m++) {
		for( int32_t o0=0, i0=0; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block11_deep_1_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block11_deep_1_swish_Sigmoid( const float X[1][768][24][24], float Y[1][768][24][24] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block11_deep_1_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack3_block11_deep_1_swish_mul_1( const float A[1][768][24][24], const float B[1][768][24][24], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block11_deep_2_GC_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block11_deep_2_GC_conv_Conv2D( const float x[1][768][24][24], const float w[768][16][3][3], const float bias[768], float y[1][768][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 48
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 16; // output group size, i.e. maps/group
	uint32_t gi = 16; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<48; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c-(gi*g)][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block11_deep_2_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block11_deep_2_swish_Sigmoid( const float X[1][768][24][24], float Y[1][768][24][24] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block11_deep_2_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack3_block11_deep_2_swish_mul_1( const float A[1][768][24][24], const float B[1][768][24][24], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           GlobalAveragePool
 * Name: catnext/tf.math.reduce_mean_18/Mean
 */
LIB_HIDDEN void node_model_catnext_tf_math_reduce_mean_18_Mean( const float input[1][768][24][24], float output[1][768][1][1] )
{
	/* GlobalAveragePool */
	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<768; c++ ) {
		float dimsum=0.0f;
		for( int32_t d0 = 0; d0<24; d0++ ) {
		for( int32_t d1 = 0; d1<24; d1++ ) {
			dimsum +=  input[b][c][d0][d1];
		}
		}
		output[b][c][0][0] = dimsum / 576;
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block11_deep_2_se_1_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack3_block11_deep_2_se_1_conv_BiasAdd( const float x[1][768][1][1], const float w[192][768][1][1], const float bias[192], float y[1][192][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<768; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Relu
 * Name: catnext/stack3_block11_deep_2_se_relu/Relu
 */
LIB_HIDDEN void node_model_catnext_stack3_block11_deep_2_se_relu_Relu( const float X[1][192][1][1], float Y[1][192][1][1] )
{
	/*Relu*/
	float *X_ptr = (float*)X;
	float *Y_ptr = (float*)Y;
	for( uint32_t i=0; i<192; i++ )
		Y_ptr[i] = X_ptr[i] > 0 ? X_ptr[i] : 0;

}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block11_deep_2_se_2_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack3_block11_deep_2_se_2_conv_BiasAdd( const float x[1][192][1][1], const float w[768][192][1][1], const float bias[768], float y[1][768][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<768; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block11_deep_2_se_sigmoid/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block11_deep_2_se_sigmoid_Sigmoid( const float X[1][768][1][1], float Y[1][768][1][1] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<1; i2++) {
	for (unsigned i3=0; i3<1; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block11_deep_2_se_out/mul
 */
LIB_HIDDEN void node_model_catnext_stack3_block11_deep_2_se_out_mul( const float A[1][768][24][24], const float B[1][768][1][1], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][0][0];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block11_deep_3_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block11_deep_3_conv_Conv2D( const float x[1][768][24][24], const float w[192][768][1][1], const float bias[192], float y[1][192][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<768; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name: catnext/stack3_block11_add/add
 */
LIB_HIDDEN void node_model_catnext_stack3_block11_add_add( const float A[1][192][24][24], const float B[1][192][24][24], float C[1][192][24][24] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block12_deep_1_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block12_deep_1_conv_Conv2D( const float x[1][192][24][24], const float w[768][192][1][1], const float bias[768], float y[1][768][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<768; m++) {
		for( int32_t o0=0, i0=0; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block12_deep_1_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block12_deep_1_swish_Sigmoid( const float X[1][768][24][24], float Y[1][768][24][24] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block12_deep_1_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack3_block12_deep_1_swish_mul_1( const float A[1][768][24][24], const float B[1][768][24][24], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block12_deep_2_GC_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block12_deep_2_GC_conv_Conv2D( const float x[1][768][24][24], const float w[768][16][3][3], const float bias[768], float y[1][768][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 48
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 16; // output group size, i.e. maps/group
	uint32_t gi = 16; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<48; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c-(gi*g)][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block12_deep_2_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block12_deep_2_swish_Sigmoid( const float X[1][768][24][24], float Y[1][768][24][24] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block12_deep_2_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack3_block12_deep_2_swish_mul_1( const float A[1][768][24][24], const float B[1][768][24][24], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           GlobalAveragePool
 * Name: catnext/tf.math.reduce_mean_19/Mean
 */
LIB_HIDDEN void node_model_catnext_tf_math_reduce_mean_19_Mean( const float input[1][768][24][24], float output[1][768][1][1] )
{
	/* GlobalAveragePool */
	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<768; c++ ) {
		float dimsum=0.0f;
		for( int32_t d0 = 0; d0<24; d0++ ) {
		for( int32_t d1 = 0; d1<24; d1++ ) {
			dimsum +=  input[b][c][d0][d1];
		}
		}
		output[b][c][0][0] = dimsum / 576;
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block12_deep_2_se_1_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack3_block12_deep_2_se_1_conv_BiasAdd( const float x[1][768][1][1], const float w[192][768][1][1], const float bias[192], float y[1][192][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<768; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Relu
 * Name: catnext/stack3_block12_deep_2_se_relu/Relu
 */
LIB_HIDDEN void node_model_catnext_stack3_block12_deep_2_se_relu_Relu( const float X[1][192][1][1], float Y[1][192][1][1] )
{
	/*Relu*/
	float *X_ptr = (float*)X;
	float *Y_ptr = (float*)Y;
	for( uint32_t i=0; i<192; i++ )
		Y_ptr[i] = X_ptr[i] > 0 ? X_ptr[i] : 0;

}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block12_deep_2_se_2_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack3_block12_deep_2_se_2_conv_BiasAdd( const float x[1][192][1][1], const float w[768][192][1][1], const float bias[768], float y[1][768][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<768; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack3_block12_deep_2_se_sigmoid/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack3_block12_deep_2_se_sigmoid_Sigmoid( const float X[1][768][1][1], float Y[1][768][1][1] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<1; i2++) {
	for (unsigned i3=0; i3<1; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack3_block12_deep_2_se_out/mul
 */
LIB_HIDDEN void node_model_catnext_stack3_block12_deep_2_se_out_mul( const float A[1][768][24][24], const float B[1][768][1][1], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][0][0];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack3_block12_deep_3_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack3_block12_deep_3_conv_Conv2D( const float x[1][768][24][24], const float w[192][768][1][1], const float bias[192], float y[1][192][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<768; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name: catnext/stack3_block12_add/add
 */
LIB_HIDDEN void node_model_catnext_stack3_block12_add_add( const float A[1][192][24][24], const float B[1][192][24][24], float C[1][192][24][24] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<192; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack4_block1_deep_1_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack4_block1_deep_1_conv_Conv2D( const float x[1][192][24][24], const float w[768][192][1][1], const float bias[768], float y[1][768][24][24] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<768; m++) {
		for( int32_t o0=0, i0=0; o0<24; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<24; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack4_block1_deep_1_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack4_block1_deep_1_swish_Sigmoid( const float X[1][768][24][24], float Y[1][768][24][24] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack4_block1_deep_1_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack4_block1_deep_1_swish_mul_1( const float A[1][768][24][24], const float B[1][768][24][24], float C[1][768][24][24] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<24; i2++) {
	for (unsigned i3=0; i3<24; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack4_block1_deep_2_GC_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack4_block1_deep_2_GC_conv_Conv2D( const float x[1][768][24][24], const float w[768][16][3][3], const float bias[768], float y[1][768][12][12] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 48
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 2 2 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 16; // output group size, i.e. maps/group
	uint32_t gi = 16; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<48; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<12; o0++, i0+=2) {
		for( int32_t o1=0, i1=-1; o1<12; o1++, i1+=2) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=24) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=24) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c-(gi*g)][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack4_block1_deep_2_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack4_block1_deep_2_swish_Sigmoid( const float X[1][768][12][12], float Y[1][768][12][12] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<12; i2++) {
	for (unsigned i3=0; i3<12; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack4_block1_deep_2_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack4_block1_deep_2_swish_mul_1( const float A[1][768][12][12], const float B[1][768][12][12], float C[1][768][12][12] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<12; i2++) {
	for (unsigned i3=0; i3<12; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           GlobalAveragePool
 * Name: catnext/tf.math.reduce_mean_20/Mean
 */
LIB_HIDDEN void node_model_catnext_tf_math_reduce_mean_20_Mean( const float input[1][768][12][12], float output[1][768][1][1] )
{
	/* GlobalAveragePool */
	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<768; c++ ) {
		float dimsum=0.0f;
		for( int32_t d0 = 0; d0<12; d0++ ) {
		for( int32_t d1 = 0; d1<12; d1++ ) {
			dimsum +=  input[b][c][d0][d1];
		}
		}
		output[b][c][0][0] = dimsum / 144;
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack4_block1_deep_2_se_1_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack4_block1_deep_2_se_1_conv_BiasAdd( const float x[1][768][1][1], const float w[192][768][1][1], const float bias[192], float y[1][192][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<768; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Relu
 * Name: catnext/stack4_block1_deep_2_se_relu/Relu
 */
LIB_HIDDEN void node_model_catnext_stack4_block1_deep_2_se_relu_Relu( const float X[1][192][1][1], float Y[1][192][1][1] )
{
	/*Relu*/
	float *X_ptr = (float*)X;
	float *Y_ptr = (float*)Y;
	for( uint32_t i=0; i<192; i++ )
		Y_ptr[i] = X_ptr[i] > 0 ? X_ptr[i] : 0;

}

/*
 * Operand:           Conv
 * Name: catnext/stack4_block1_deep_2_se_2_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack4_block1_deep_2_se_2_conv_BiasAdd( const float x[1][192][1][1], const float w[768][192][1][1], const float bias[768], float y[1][768][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<768; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack4_block1_deep_2_se_sigmoid/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack4_block1_deep_2_se_sigmoid_Sigmoid( const float X[1][768][1][1], float Y[1][768][1][1] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<1; i2++) {
	for (unsigned i3=0; i3<1; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack4_block1_deep_2_se_out/mul
 */
LIB_HIDDEN void node_model_catnext_stack4_block1_deep_2_se_out_mul( const float A[1][768][12][12], const float B[1][768][1][1], float C[1][768][12][12] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<768; i1++) {
	for (unsigned i2=0; i2<12; i2++) {
	for (unsigned i3=0; i3<12; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][0][0];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack4_block1_deep_3_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack4_block1_deep_3_conv_Conv2D( const float x[1][768][12][12], const float w[288][768][1][1], const float bias[288], float y[1][288][12][12] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<288; m++) {
		for( int32_t o0=0, i0=0; o0<12; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<12; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<768; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=12) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=12) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Conv
 * Name: catnext/stack4_block2_deep_1_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack4_block2_deep_1_conv_Conv2D( const float x[1][288][12][12], const float w[1152][288][1][1], float y[1][1152][12][12] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<1152; m++) {
		for( int32_t o0=0, i0=0; o0<12; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<12; o1++, i1+=1) {
			y[b][m][o0][o1] = 0;
			for( int32_t c=0; c<288; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=12) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=12) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           BatchNormalization
 * Name: catnext/stack4_block2_deep_1_bn/FusedBatchNormV3
 */
LIB_HIDDEN void node_model_catnext_stack4_block2_deep_1_bn_FusedBatchNormV3( const float X[1][1152][12][12], const float scale[1152], const float bias[1152], const float mean[1152], const float var[1152], float output[1][1152][12][12] )
{
	/* BatchNormalization
	 * epsilon = 1.0009999641624744982e-05
	 * momentum = 0.89999997615814208984
	 */

	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<1152; c++ ) {
	for( uint32_t i2=0; i2<12; i2++ ) {
	for( uint32_t i3=0; i3<12; i3++ ) {
		float tmp_X = ( X[b][c][i2][i3] - mean[c] ) / ( var[c] );
		output[b][c][i2][i3] = tmp_X * scale[c] + bias[c];
	}
	}
	}
	}
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack4_block2_deep_1_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack4_block2_deep_1_swish_Sigmoid( const float X[1][1152][12][12], float Y[1][1152][12][12] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<1152; i1++) {
	for (unsigned i2=0; i2<12; i2++) {
	for (unsigned i3=0; i3<12; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack4_block2_deep_1_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack4_block2_deep_1_swish_mul_1( const float A[1][1152][12][12], const float B[1][1152][12][12], float C[1][1152][12][12] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<1152; i1++) {
	for (unsigned i2=0; i2<12; i2++) {
	for (unsigned i3=0; i3<12; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack4_block2_deep_2_GC_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack4_block2_deep_2_GC_conv_Conv2D( const float x[1][1152][12][12], const float w[1152][16][3][3], const float bias[1152], float y[1][1152][12][12] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 72
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 16; // output group size, i.e. maps/group
	uint32_t gi = 16; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<72; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<12; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<12; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=12) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=12) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c-(gi*g)][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack4_block2_deep_2_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack4_block2_deep_2_swish_Sigmoid( const float X[1][1152][12][12], float Y[1][1152][12][12] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<1152; i1++) {
	for (unsigned i2=0; i2<12; i2++) {
	for (unsigned i3=0; i3<12; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack4_block2_deep_2_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_stack4_block2_deep_2_swish_mul_1( const float A[1][1152][12][12], const float B[1][1152][12][12], float C[1][1152][12][12] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<1152; i1++) {
	for (unsigned i2=0; i2<12; i2++) {
	for (unsigned i3=0; i3<12; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           GlobalAveragePool
 * Name: catnext/tf.math.reduce_mean_21/Mean
 */
LIB_HIDDEN void node_model_catnext_tf_math_reduce_mean_21_Mean( const float input[1][1152][12][12], float output[1][1152][1][1] )
{
	/* GlobalAveragePool */
	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<1152; c++ ) {
		float dimsum=0.0f;
		for( int32_t d0 = 0; d0<12; d0++ ) {
		for( int32_t d1 = 0; d1<12; d1++ ) {
			dimsum +=  input[b][c][d0][d1];
		}
		}
		output[b][c][0][0] = dimsum / 144;
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack4_block2_deep_2_se_1_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack4_block2_deep_2_se_1_conv_BiasAdd( const float x[1][1152][1][1], const float w[288][1152][1][1], const float bias[288], float y[1][288][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<288; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<1152; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Relu
 * Name: catnext/stack4_block2_deep_2_se_relu/Relu
 */
LIB_HIDDEN void node_model_catnext_stack4_block2_deep_2_se_relu_Relu( const float X[1][288][1][1], float Y[1][288][1][1] )
{
	/*Relu*/
	float *X_ptr = (float*)X;
	float *Y_ptr = (float*)Y;
	for( uint32_t i=0; i<288; i++ )
		Y_ptr[i] = X_ptr[i] > 0 ? X_ptr[i] : 0;

}

/*
 * Operand:           Conv
 * Name: catnext/stack4_block2_deep_2_se_2_conv/BiasAdd
 */
LIB_HIDDEN void node_model_catnext_stack4_block2_deep_2_se_2_conv_BiasAdd( const float x[1][288][1][1], const float w[1152][288][1][1], const float bias[1152], float y[1][1152][1][1] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<1152; m++) {
		for( int32_t o0=0, i0=0; o0<1; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<1; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<288; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=1) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=1) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/stack4_block2_deep_2_se_sigmoid/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_stack4_block2_deep_2_se_sigmoid_Sigmoid( const float X[1][1152][1][1], float Y[1][1152][1][1] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<1152; i1++) {
	for (unsigned i2=0; i2<1; i2++) {
	for (unsigned i3=0; i3<1; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/stack4_block2_deep_2_se_out/mul
 */
LIB_HIDDEN void node_model_catnext_stack4_block2_deep_2_se_out_mul( const float A[1][1152][12][12], const float B[1][1152][1][1], float C[1][1152][12][12] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<1152; i1++) {
	for (unsigned i2=0; i2<12; i2++) {
	for (unsigned i3=0; i3<12; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][0][0];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/stack4_block2_deep_3_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_stack4_block2_deep_3_conv_Conv2D( const float x[1][1152][12][12], const float w[288][1152][1][1], const float bias[288], float y[1][288][12][12] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<288; m++) {
		for( int32_t o0=0, i0=0; o0<12; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<12; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<1152; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=12) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=12) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Add
 * Name: catnext/stack4_block2_add/add
 */
LIB_HIDDEN void node_model_catnext_stack4_block2_add_add( const float A[1][288][12][12], const float B[1][288][12][12], float C[1][288][12][12] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<288; i1++) {
	for (unsigned i2=0; i2<12; i2++) {
	for (unsigned i3=0; i3<12; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: catnext/features_conv/Conv2D
 */
LIB_HIDDEN void node_model_catnext_features_conv_Conv2D( const float x[1][288][12][12], const float w[1536][288][1][1], const float bias[1536], float y[1][1536][12][12] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<1536; m++) {
		for( int32_t o0=0, i0=0; o0<12; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<12; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<288; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=12) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=12) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: catnext/features_swish/Sigmoid
 */
LIB_HIDDEN void node_model_catnext_features_swish_Sigmoid( const float X[1][1536][12][12], float Y[1][1536][12][12] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<1536; i1++) {
	for (unsigned i2=0; i2<12; i2++) {
	for (unsigned i3=0; i3<12; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: catnext/features_swish/mul_1
 */
LIB_HIDDEN void node_model_catnext_features_swish_mul_1( const float A[1][1536][12][12], const float B[1][1536][12][12], float C[1][1536][12][12] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<1536; i1++) {
	for (unsigned i2=0; i2<12; i2++) {
	for (unsigned i3=0; i3<12; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           GlobalAveragePool
 * Name: model/global_average_pooling2d/Mean
 */
LIB_HIDDEN void node_model_global_average_pooling2d_Mean( const float input[1][1536][12][12], float output[1][1536][1][1] )
{
	/* GlobalAveragePool */
	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<1536; c++ ) {
		float dimsum=0.0f;
		for( int32_t d0 = 0; d0<12; d0++ ) {
		for( int32_t d1 = 0; d1<12; d1++ ) {
			dimsum +=  input[b][c][d0][d1];
		}
		}
		output[b][c][0][0] = dimsum / 144;
	}
	}
}

/*
 * Operand:           Transpose
 * Name: model/conv2d/BiasAdd__742
 */
LIB_HIDDEN void node_model_conv2d_BiasAdd__742( const float input[1][39][384][384], float output[1][384][39][384] )
{
	/* Transpose
	 * perm = 0 3 1 2 
	 */
	for( uint32_t i0=0; i0<1; i0++ ) {
	for( uint32_t i1=0; i1<39; i1++ ) {
	for( uint32_t i2=0; i2<384; i2++ ) {
	for( uint32_t i3=0; i3<384; i3++ ) {
		output[i0][i3][i1][i2] = input[i0][i1][i2][i3];
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: model/conv2d/BiasAdd
 */
LIB_HIDDEN void node_model_conv2d_BiasAdd( const float x[1][384][39][384], const float w[8][384][3][3], const float bias[8], float y[1][8][19][191] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 3 3 
	 * pads: 0 0 0 0 
	 * strides: 2 2 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<8; m++) {
		for( int32_t o0=0, i0=0; o0<19; o0++, i0+=2) {
		for( int32_t o1=0, i1=0; o1<191; o1++, i1+=2) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<384; c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=39) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=384) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: model/conv2d/Sigmoid
 */
LIB_HIDDEN void node_model_conv2d_Sigmoid( const float X[1][8][19][191], float Y[1][8][19][191] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<8; i1++) {
	for (unsigned i2=0; i2<19; i2++) {
	for (unsigned i3=0; i3<191; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: model/conv2d/mul_1
 */
LIB_HIDDEN void node_model_conv2d_mul_1( const float A[1][8][19][191], const float B[1][8][19][191], float C[1][8][19][191] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<8; i1++) {
	for (unsigned i2=0; i2<19; i2++) {
	for (unsigned i3=0; i3<191; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Squeeze
 * Name: model/global_average_pooling2d/Mean_Squeeze__1173
 */
LIB_HIDDEN void node_model_global_average_pooling2d_Mean_Squeeze__1173( const float input[1][1536][1][1], const int64_t axes_tensor[2], float output[1][1536] )
{
	/*Squeeze*/
	float *data = (float*)input;
	float *squeezed= (float*)output;
	for( uint32_t i=0; i<1536; i++ )
		squeezed[i] = data[i];

}

/*
 * Operand:           Transpose
 * Name: Transpose__1178
 */
LIB_HIDDEN void node_Transpose__1178( const float input[1][8][19][191], float output[1][19][191][8] )
{
	/* Transpose
	 * perm = 0 2 3 1 
	 */
	for( uint32_t i0=0; i0<1; i0++ ) {
	for( uint32_t i1=0; i1<8; i1++ ) {
	for( uint32_t i2=0; i2<19; i2++ ) {
	for( uint32_t i3=0; i3<191; i3++ ) {
		output[i0][i2][i3][i1] = input[i0][i1][i2][i3];
	}
	}
	}
	}
}

/*
 * Operand:           Reshape
 * Name: model/flatten/Reshape
 */
LIB_HIDDEN void node_model_flatten_Reshape( const float data[1][19][191][8], const int64_t shape[2], float reshaped[1][29032] )
{
	/*Reshape*/
	float *data_ptr = (float*)data;
	float *reshaped_ptr = (float*)reshaped;
	for( uint32_t i=0; i<29032; i++ )
		reshaped_ptr[i] = data_ptr[i];

}

/*
 * Operand:           MatMul
 * Name: model/dense/MatMul
 */
LIB_HIDDEN void node_model_dense_MatMul( const float A[1][29032], const float B[29032][1], float Y[1][1] )
{
	/* MatMul */
	for( uint32_t r=0; r<1; r++ )
		for( uint32_t c=0; c<1; c++ ) {
			Y[r][c] = 0;
			for( uint32_t i=0; i<29032; i++ )
				Y[r][c] += A[r][i] * B[i][c];
		}
}

/*
 * Operand:           Add
 * Name: model/dense/BiasAdd
 */
LIB_HIDDEN void node_model_dense_BiasAdd( const float A[1][1], const float B[1], float C[1][1] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<1; i1++) {
		C[i0][i1] = A[0][0]+B[0];;
	}
	}
}

/*
 * Operand:           Sigmoid
 * Name: model/dense/Sigmoid
 */
LIB_HIDDEN void node_model_dense_Sigmoid( const float X[1][1], float Y[1][1] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<1; i1++) {
		Y[i0][i1] = 1/(1+exp(-X[i0][i1]));
	}
	}
}

/*
 * Operand:           Mul
 * Name: model/dense/mul_1
 */
LIB_HIDDEN void node_model_dense_mul_1( const float A[1][1], const float B[1][1], float C[1][1] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<1; i1++) {
		C[i0][i1] = A[0][0]*B[0][0];;
	}
	}
}

/*
 * Operand:           MatMul
 * Name: model/dense_1/MatMul
 */
LIB_HIDDEN void node_model_dense_1_MatMul( const float A[1][1], const float B[1][8], float Y[1][8] )
{
	/* MatMul */
	for( uint32_t r=0; r<1; r++ )
		for( uint32_t c=0; c<8; c++ ) {
			Y[r][c] = 0;
			for( uint32_t i=0; i<1; i++ )
				Y[r][c] += A[r][i] * B[i][c];
		}
}

/*
 * Operand:           Add
 * Name: model/dense_1/BiasAdd
 */
LIB_HIDDEN void node_model_dense_1_BiasAdd( const float A[1][8], const float B[8], float C[1][8] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<8; i1++) {
		C[i0][i1] = A[0][i1]+B[i1];;
	}
	}
}

/*
 * Operand:           Sigmoid
 * Name: model/dense_1/Sigmoid
 */
LIB_HIDDEN void node_model_dense_1_Sigmoid( const float X[1][8], float Y[1][8] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<8; i1++) {
		Y[i0][i1] = 1/(1+exp(-X[i0][i1]));
	}
	}
}

/*
 * Operand:           Reshape
 * Name: model/reshape_1/Reshape
 */
LIB_HIDDEN void node_model_reshape_1_Reshape( const float data[1][8], const int64_t shape[4], float reshaped[1][1][1][8] )
{
	/*Reshape*/
	float *data_ptr = (float*)data;
	float *reshaped_ptr = (float*)reshaped;
	for( uint32_t i=0; i<8; i++ )
		reshaped_ptr[i] = data_ptr[i];

}

/*
 * Operand:           Mul
 * Name: model/tf.math.multiply/Mul
 */
LIB_HIDDEN void node_model_tf_math_multiply_Mul( const float A[1][19][191][8], const float B[1][1][1][8], float C[1][19][191][8] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<19; i1++) {
	for (unsigned i2=0; i2<191; i2++) {
	for (unsigned i3=0; i3<8; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][0][0][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Transpose
 * Name: model/batch_normalization/FusedBatchNormV3__763
 */
LIB_HIDDEN void node_model_batch_normalization_FusedBatchNormV3__763( const float input[1][19][191][8], float output[1][8][19][191] )
{
	/* Transpose
	 * perm = 0 3 1 2 
	 */
	for( uint32_t i0=0; i0<1; i0++ ) {
	for( uint32_t i1=0; i1<19; i1++ ) {
	for( uint32_t i2=0; i2<191; i2++ ) {
	for( uint32_t i3=0; i3<8; i3++ ) {
		output[i0][i3][i1][i2] = input[i0][i1][i2][i3];
	}
	}
	}
	}
}

/*
 * Operand:           BatchNormalization
 * Name: model/batch_normalization/FusedBatchNormV3
 */
LIB_HIDDEN void node_model_batch_normalization_FusedBatchNormV3( const float X[1][8][19][191], const float scale[8], const float bias[8], const float mean[8], const float var[8], float output[1][8][19][191] )
{
	/* BatchNormalization
	 * epsilon = 0.0010000000474974513054
	 * momentum = 0.89999997615814208984
	 */

	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<8; c++ ) {
	for( uint32_t i2=0; i2<19; i2++ ) {
	for( uint32_t i3=0; i3<191; i3++ ) {
		float tmp_X = ( X[b][c][i2][i3] - mean[c] ) / ( var[c] );
		output[b][c][i2][i3] = tmp_X * scale[c] + bias[c];
	}
	}
	}
	}
}

/*
 * Operand:           Conv
 * Name: model/separable_conv2d/separable_conv2d/depthwise
 */
LIB_HIDDEN void node_model_separable_conv2d_separable_conv2d_depthwise( const float x[1][8][19][191], const float w[8][1][2][2], float y[1][8][18][190] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 8
	 * kernel_shape: 2 2 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<8; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=0; o0<18; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<190; o1++, i1+=1) {
			y[b][m][o0][o1] = 0;
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<2; k0++ ) {
			for( uint32_t k1=0; k1<2; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=19) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=191) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c-(gi*g)][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           Conv
 * Name: model/separable_conv2d/BiasAdd
 */
LIB_HIDDEN void node_model_separable_conv2d_BiasAdd( const float x[1][8][18][190], const float w[16][8][1][1], const float bias[16], float y[1][16][18][190] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<16; m++) {
		for( int32_t o0=0, i0=0; o0<18; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<190; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<8; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				if( ii0<0) continue;
				if( ii0>=18) continue;
				int ii1 = i1+k1 * 1;
				if( ii1<0) continue;
				if( ii1>=190) continue;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] *w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           Sigmoid
 * Name: model/separable_conv2d/Sigmoid
 */
LIB_HIDDEN void node_model_separable_conv2d_Sigmoid( const float X[1][16][18][190], float Y[1][16][18][190] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<16; i1++) {
	for (unsigned i2=0; i2<18; i2++) {
	for (unsigned i3=0; i3<190; i3++) {
		Y[i0][i1][i2][i3] = 1/(1+exp(-X[i0][i1][i2][i3]));
	}
	}
	}
	}
}

/*
 * Operand:           Mul
 * Name: model/separable_conv2d/mul_1
 */
LIB_HIDDEN void node_model_separable_conv2d_mul_1( const float A[1][16][18][190], const float B[1][16][18][190], float C[1][16][18][190] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<16; i1++) {
	for (unsigned i2=0; i2<18; i2++) {
	for (unsigned i3=0; i3<190; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][i1][i2][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Transpose
 * Name: Transpose__1174
 */
LIB_HIDDEN void node_Transpose__1174( const float input[1][16][18][190], float output[1][18][190][16] )
{
	/* Transpose
	 * perm = 0 2 3 1 
	 */
	for( uint32_t i0=0; i0<1; i0++ ) {
	for( uint32_t i1=0; i1<16; i1++ ) {
	for( uint32_t i2=0; i2<18; i2++ ) {
	for( uint32_t i3=0; i3<190; i3++ ) {
		output[i0][i2][i3][i1] = input[i0][i1][i2][i3];
	}
	}
	}
	}
}

/*
 * Operand:           Reshape
 * Name: model/flatten_1/Reshape
 */
LIB_HIDDEN void node_model_flatten_1_Reshape( const float data[1][18][190][16], const int64_t shape[2], float reshaped[1][54720] )
{
	/*Reshape*/
	float *data_ptr = (float*)data;
	float *reshaped_ptr = (float*)reshaped;
	for( uint32_t i=0; i<54720; i++ )
		reshaped_ptr[i] = data_ptr[i];

}

/*
 * Operand:           MatMul
 * Name: model/dense_2/MatMul
 */
LIB_HIDDEN void node_model_dense_2_MatMul( const float A[1][54720], const float B[54720][1], float Y[1][1] )
{
	/* MatMul */
	for( uint32_t r=0; r<1; r++ )
		for( uint32_t c=0; c<1; c++ ) {
			Y[r][c] = 0;
			for( uint32_t i=0; i<54720; i++ )
				Y[r][c] += A[r][i] * B[i][c];
		}
}

/*
 * Operand:           Add
 * Name: model/dense_2/BiasAdd
 */
LIB_HIDDEN void node_model_dense_2_BiasAdd( const float A[1][1], const float B[1], float C[1][1] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<1; i1++) {
		C[i0][i1] = A[0][0]+B[0];;
	}
	}
}

/*
 * Operand:           Sigmoid
 * Name: model/dense_2/Sigmoid
 */
LIB_HIDDEN void node_model_dense_2_Sigmoid( const float X[1][1], float Y[1][1] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<1; i1++) {
		Y[i0][i1] = 1/(1+exp(-X[i0][i1]));
	}
	}
}

/*
 * Operand:           Mul
 * Name: model/dense_2/mul_1
 */
LIB_HIDDEN void node_model_dense_2_mul_1( const float A[1][1], const float B[1][1], float C[1][1] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<1; i1++) {
		C[i0][i1] = A[0][0]*B[0][0];;
	}
	}
}

/*
 * Operand:           MatMul
 * Name: model/dense_3/MatMul
 */
LIB_HIDDEN void node_model_dense_3_MatMul( const float A[1][1], const float B[1][16], float Y[1][16] )
{
	/* MatMul */
	for( uint32_t r=0; r<1; r++ )
		for( uint32_t c=0; c<16; c++ ) {
			Y[r][c] = 0;
			for( uint32_t i=0; i<1; i++ )
				Y[r][c] += A[r][i] * B[i][c];
		}
}

/*
 * Operand:           Add
 * Name: model/dense_3/BiasAdd
 */
LIB_HIDDEN void node_model_dense_3_BiasAdd( const float A[1][16], const float B[16], float C[1][16] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<16; i1++) {
		C[i0][i1] = A[0][i1]+B[i1];;
	}
	}
}

/*
 * Operand:           Sigmoid
 * Name: model/dense_3/Sigmoid
 */
LIB_HIDDEN void node_model_dense_3_Sigmoid( const float X[1][16], float Y[1][16] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<16; i1++) {
		Y[i0][i1] = 1/(1+exp(-X[i0][i1]));
	}
	}
}

/*
 * Operand:           Reshape
 * Name: model/reshape_2/Reshape
 */
LIB_HIDDEN void node_model_reshape_2_Reshape( const float data[1][16], const int64_t shape[4], float reshaped[1][1][1][16] )
{
	/*Reshape*/
	float *data_ptr = (float*)data;
	float *reshaped_ptr = (float*)reshaped;
	for( uint32_t i=0; i<16; i++ )
		reshaped_ptr[i] = data_ptr[i];

}

/*
 * Operand:           Mul
 * Name: model/tf.math.multiply_1/Mul
 */
LIB_HIDDEN void node_model_tf_math_multiply_1_Mul( const float A[1][18][190][16], const float B[1][1][1][16], float C[1][18][190][16] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<18; i1++) {
	for (unsigned i2=0; i2<190; i2++) {
	for (unsigned i3=0; i3<16; i3++) {
		C[i0][i1][i2][i3] = A[0][i1][i2][i3]*B[0][0][0][i3];;
	}
	}
	}
	}
}

/*
 * Operand:           Reshape
 * Name: model/flatten_2/Reshape
 */
LIB_HIDDEN void node_model_flatten_2_Reshape( const float data[1][18][190][16], const int64_t shape[2], float reshaped[1][54720] )
{
	/*Reshape*/
	float *data_ptr = (float*)data;
	float *reshaped_ptr = (float*)reshaped;
	for( uint32_t i=0; i<54720; i++ )
		reshaped_ptr[i] = data_ptr[i];

}

/*
 * Operand:           Concat
 * Name: model/concatenate/concat
 */
LIB_HIDDEN void node_model_concatenate_concat( const float input_0[1][1536], const float input_1[1][54720], const float input_2[1][300], float output[1][56556] )
{
	/* Concat */
	int64_t outputOffset;
	outputOffset = 0;
	for (int64_t i = 0, j = 0; i < 1536; i++) {
		*((float*)output + (outputOffset + i)) = *((float*)input_0 + i);
		if (++j == 1536) {
			outputOffset += (55020);
			j = 0;
		}
	}
	outputOffset = 1536;
	for (int64_t i = 0, j = 0; i < 54720; i++) {
		*((float*)output + (outputOffset + i)) = *((float*)input_1 + i);
		if (++j == 54720) {
			outputOffset += (1836);
			j = 0;
		}
	}
	outputOffset = 56256;
	for (int64_t i = 0, j = 0; i < 300; i++) {
		*((float*)output + (outputOffset + i)) = *((float*)input_2 + i);
		if (++j == 300) {
			outputOffset += (56256);
			j = 0;
		}
	}
}

/*
 * Operand:           MatMul
 * Name: model/dense_5/MatMul
 */
LIB_HIDDEN void node_model_dense_5_MatMul( const float A[1][56556], const float B[56556][1792], float Y[1][1792] )
{
	/* MatMul */
	for( uint32_t r=0; r<1; r++ )
		for( uint32_t c=0; c<1792; c++ ) {
			Y[r][c] = 0;
			for( uint32_t i=0; i<56556; i++ )
				Y[r][c] += A[r][i] * B[i][c];
		}
}

/*
 * Operand:           Add
 * Name: model/dense_5/BiasAdd
 */
LIB_HIDDEN void node_model_dense_5_BiasAdd( const float A[1][1792], const float B[1792], float C[1][1792] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<1792; i1++) {
		C[i0][i1] = A[0][i1]+B[i1];;
	}
	}
}

/*
 * Operand:           Sigmoid
 * Name: model/dense_5/Sigmoid
 */
LIB_HIDDEN void node_model_dense_5_Sigmoid( const float X[1][1792], float Y[1][1792] )
{
	/* Sigmoid
	   Implemented with Elementwise template.
	   alpha = 0.0000000000000000000
	   beta = 0.0000000000000000000
	*/
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<1792; i1++) {
		Y[i0][i1] = 1/(1+exp(-X[i0][i1]));
	}
	}
}

/*
 * Operand:           Mul
 * Name: model/dense_5/mul_1
 */
LIB_HIDDEN void node_model_dense_5_mul_1( const float A[1][1792], const float B[1][1792], float C[1][1792] )
{
	/* Mul
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<1792; i1++) {
		C[i0][i1] = A[0][i1]*B[0][i1];;
	}
	}
}

/*
 * Operand:           MatMul
 * Name: model/dense_6/MatMul
 */
LIB_HIDDEN void node_model_dense_6_MatMul( const float A[1][1792], const float B[1792][66], float Y[1][66] )
{
	/* MatMul */
	for( uint32_t r=0; r<1; r++ )
		for( uint32_t c=0; c<66; c++ ) {
			Y[r][c] = 0;
			for( uint32_t i=0; i<1792; i++ )
				Y[r][c] += A[r][i] * B[i][c];
		}
}

/*
 * Operand:           Add
 * Name: model/dense_6/BiasAdd
 */
LIB_HIDDEN void node_model_dense_6_BiasAdd( const float A[1][66], const float B[66], float C[1][66] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++) {
	for (unsigned i1=0; i1<66; i1++) {
		C[i0][i1] = A[0][i1]+B[i1];;
	}
	}
}

/*
 * Operand:           Softmax
 * Name: model/dense_6/Softmax
 */
LIB_HIDDEN void node_model_dense_6_Softmax( const float input[1][66], float output[1][66] )
{
	/* Softmax 13
	 * axis = -1
	 */
	for( uint32_t i0=0; i0<1; i0++ ) {
		float max = -INFINITY;
		for( uint32_t i1=0; i1<66; i1++ ) {
			max = max>input[i0][i1] ? max :input[i0][i1];
		};
		float sum = 0.0;
		for( uint32_t i1=0; i1<66; i1++ ) {
			sum += expf(input[i0][i1] - max);
		};
		for( uint32_t i1=0; i1<66; i1++ ) {
			output[i0][i1] = expf(input[i0][i1] - max)/sum;
		};
	}
}


