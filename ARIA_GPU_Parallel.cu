#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#define BLOCKLEN 16

//FUNCTION Prototype
__device__ void Add_SubstOdd_Layer(uint8_t* input, uint8_t* key);
__device__ void Add_SubstEven_Layer(uint8_t* input, uint8_t* key);
__device__ void DiffLayer(uint8_t* input);
__device__ void Rot_L(uint8_t* input, uint8_t* output, int num);
__device__ void ROT_XOR(uint8_t* input, uint8_t* op, uint8_t* output, int num);
__global__ void Enc_KEY_Expansion(uint8_t* mk, uint8_t* rk);
__device__ void Round_odd(uint8_t* input, uint8_t* rk);
__device__ void Round_even(uint8_t* input, uint8_t* rk);
__device__ void Final_Round(uint8_t* input, uint8_t* key12, uint8_t* key13);
__device__ void ARIA_EnCrypt(uint8_t* pt, uint8_t* ct, uint8_t* rk);
__device__ void CTR_INC(uint8_t* ICTR);
__global__ void GPU_ARIA_CTR(uint8_t* pt, uint8_t* ct, uint8_t* iv, uint32_t enc_block, uint32_t ptlen, uint8_t* rk);
void printstate(uint8_t* data, int dataLen);


//s-box type1
__constant__ static const uint8_t ARIA_S1[256] = {
	0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
	0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
	0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
	0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
	0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
	0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
	0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
	0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
	0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
	0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
	0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
	0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
	0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
	0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
	0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
	0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

//s-box type2
__constant__ static const uint8_t ARIA_S2[256] = {
	0xe2, 0x4e, 0x54, 0xfc, 0x94, 0xc2, 0x4a, 0xcc, 0x62, 0x0d, 0x6a, 0x46, 0x3c, 0x4d, 0x8b, 0xd1,
	0x5e, 0xfa, 0x64, 0xcb, 0xb4, 0x97, 0xbe, 0x2b, 0xbc, 0x77, 0x2e, 0x03, 0xd3, 0x19, 0x59, 0xc1,
	0x1d, 0x06, 0x41, 0x6b, 0x55, 0xf0, 0x99, 0x69, 0xea, 0x9c, 0x18, 0xae, 0x63, 0xdf, 0xe7, 0xbb,
	0x00, 0x73, 0x66, 0xfb, 0x96, 0x4c, 0x85, 0xe4, 0x3a, 0x09, 0x45, 0xaa, 0x0f, 0xee, 0x10, 0xeb,
	0x2d, 0x7f, 0xf4, 0x29, 0xac, 0xcf, 0xad, 0x91, 0x8d, 0x78, 0xc8, 0x95, 0xf9, 0x2f, 0xce, 0xcd,
	0x08, 0x7a, 0x88, 0x38, 0x5c, 0x83, 0x2a, 0x28, 0x47, 0xdb, 0xb8, 0xc7, 0x93, 0xa4, 0x12, 0x53,
	0xff, 0x87, 0x0e, 0x31, 0x36, 0x21, 0x58, 0x48, 0x01, 0x8e, 0x37, 0x74, 0x32, 0xca, 0xe9, 0xb1,
	0xb7, 0xab, 0x0c, 0xd7, 0xc4, 0x56, 0x42, 0x26, 0x07, 0x98, 0x60, 0xd9, 0xb6, 0xb9, 0x11, 0x40,
	0xec, 0x20, 0x8c, 0xbd, 0xa0, 0xc9, 0x84, 0x04, 0x49, 0x23, 0xf1, 0x4f, 0x50, 0x1f, 0x13, 0xdc,
	0xd8, 0xc0, 0x9e, 0x57, 0xe3, 0xc3, 0x7b, 0x65, 0x3b, 0x02, 0x8f, 0x3e, 0xe8, 0x25, 0x92, 0xe5,
	0x15, 0xdd, 0xfd, 0x17, 0xa9, 0xbf, 0xd4, 0x9a, 0x7e, 0xc5, 0x39, 0x67, 0xfe, 0x76, 0x9d, 0x43,
	0xa7, 0xe1, 0xd0, 0xf5, 0x68, 0xf2, 0x1b, 0x34, 0x70, 0x05, 0xa3, 0x8a, 0xd5, 0x79, 0x86, 0xa8,
	0x30, 0xc6, 0x51, 0x4b, 0x1e, 0xa6, 0x27, 0xf6, 0x35, 0xd2, 0x6e, 0x24, 0x16, 0x82, 0x5f, 0xda,
	0xe6, 0x75, 0xa2, 0xef, 0x2c, 0xb2, 0x1c, 0x9f, 0x5d, 0x6f, 0x80, 0x0a, 0x72, 0x44, 0x9b, 0x6c,
	0x90, 0x0b, 0x5b, 0x33, 0x7d, 0x5a, 0x52, 0xf3, 0x61, 0xa1, 0xf7, 0xb0, 0xd6, 0x3f, 0x7c, 0x6d,
	0xed, 0x14, 0xe0, 0xa5, 0x3d, 0x22, 0xb3, 0xf8, 0x89, 0xde, 0x71, 0x1a, 0xaf, 0xba, 0xb5, 0x81
};

//inverse of s-box type1
__constant__ static const uint8_t ARIA_RS1[256] = {
	0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
	0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
	0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
	0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
	0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
	0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
	0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
	0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
	0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
	0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
	0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
	0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
	0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
	0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
	0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
	0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
};

//inverse of s-box type2
__constant__ static const int8_t ARIA_RS2[256] = {
	0x30, 0x68, 0x99, 0x1b, 0x87, 0xb9, 0x21, 0x78, 0x50, 0x39, 0xdb, 0xe1, 0x72, 0x09, 0x62, 0x3c,
	0x3e, 0x7e, 0x5e, 0x8e, 0xf1, 0xa0, 0xcc, 0xa3, 0x2a, 0x1d, 0xfb, 0xb6, 0xd6, 0x20, 0xc4, 0x8d,
	0x81, 0x65, 0xf5, 0x89, 0xcb, 0x9d, 0x77, 0xc6, 0x57, 0x43, 0x56, 0x17, 0xd4, 0x40, 0x1a, 0x4d,
	0xc0, 0x63, 0x6c, 0xe3, 0xb7, 0xc8, 0x64, 0x6a, 0x53, 0xaa, 0x38, 0x98, 0x0c, 0xf4, 0x9b, 0xed,
	0x7f, 0x22, 0x76, 0xaf, 0xdd, 0x3a, 0x0b, 0x58, 0x67, 0x88, 0x06, 0xc3, 0x35, 0x0d, 0x01, 0x8b,
	0x8c, 0xc2, 0xe6, 0x5f, 0x02, 0x24, 0x75, 0x93, 0x66, 0x1e, 0xe5, 0xe2, 0x54, 0xd8, 0x10, 0xce,
	0x7a, 0xe8, 0x08, 0x2c, 0x12, 0x97, 0x32, 0xab, 0xb4, 0x27, 0x0a, 0x23, 0xdf, 0xef, 0xca, 0xd9,
	0xb8, 0xfa, 0xdc, 0x31, 0x6b, 0xd1, 0xad, 0x19, 0x49, 0xbd, 0x51, 0x96, 0xee, 0xe4, 0xa8, 0x41,
	0xda, 0xff, 0xcd, 0x55, 0x86, 0x36, 0xbe, 0x61, 0x52, 0xf8, 0xbb, 0x0e, 0x82, 0x48, 0x69, 0x9a,
	0xe0, 0x47, 0x9e, 0x5c, 0x04, 0x4b, 0x34, 0x15, 0x79, 0x26, 0xa7, 0xde, 0x29, 0xae, 0x92, 0xd7,
	0x84, 0xe9, 0xd2, 0xba, 0x5d, 0xf3, 0xc5, 0xb0, 0xbf, 0xa4, 0x3b, 0x71, 0x44, 0x46, 0x2b, 0xfc,
	0xeb, 0x6f, 0xd5, 0xf6, 0x14, 0xfe, 0x7c, 0x70, 0x5a, 0x7d, 0xfd, 0x2f, 0x18, 0x83, 0x16, 0xa5,
	0x91, 0x1f, 0x05, 0x95, 0x74, 0xa9, 0xc1, 0x5b, 0x4a, 0x85, 0x6d, 0x13, 0x07, 0x4f, 0x4e, 0x45,
	0xb2, 0x0f, 0xc9, 0x1c, 0xa6, 0xbc, 0xec, 0x73, 0x90, 0x7b, 0xcf, 0x59, 0x8f, 0xa1, 0xf9, 0x2d,
	0xf2, 0xb1, 0x00, 0x94, 0x37, 0x9f, 0xd0, 0x2e, 0x9c, 0x6e, 0x28, 0x3f, 0x80, 0xf0, 0x3d, 0xd3,
	0x25, 0x8a, 0xb5, 0xe7, 0x42, 0xb3, 0xc7, 0xea, 0xf7, 0x4c, 0x11, 0x33, 0x03, 0xa2, 0xac, 0x60
};

//AddRoundKey + SubstOdd_Layer(홀수 라운드 치환계층)
__device__ void Add_SubstOdd_Layer(uint8_t* input, uint8_t* key) {
	*(input + 0) = ARIA_S1[*(input + 0) ^ *(key + 0)];
	*(input + 1) = ARIA_S2[*(input + 1) ^ *(key + 1)];
	*(input + 2) = ARIA_RS1[*(input + 2) ^ *(key + 2)];
	*(input + 3) = ARIA_RS2[*(input + 3) ^ *(key + 3)];
	*(input + 4) = ARIA_S1[*(input + 4) ^ *(key + 4)];
	*(input + 5) = ARIA_S2[*(input + 5) ^ *(key + 5)];
	*(input + 6) = ARIA_RS1[*(input + 6) ^ *(key + 6)];
	*(input + 7) = ARIA_RS2[*(input + 7) ^ *(key + 7)];
	*(input + 8) = ARIA_S1[*(input + 8) ^ *(key + 8)];
	*(input + 9) = ARIA_S2[*(input + 9) ^ *(key + 9)];
	*(input + 10) = ARIA_RS1[*(input + 10) ^ *(key + 10)];
	*(input + 11) = ARIA_RS2[*(input + 11) ^ *(key + 11)];
	*(input + 12) = ARIA_S1[*(input + 12) ^ *(key + 12)];
	*(input + 13) = ARIA_S2[*(input + 13) ^ *(key + 13)];
	*(input + 14) = ARIA_RS1[*(input + 14) ^ *(key + 14)];
	*(input + 15) = ARIA_RS2[*(input + 15) ^ *(key + 15)];
}

//AddRoundKey + SubstOdd_Layer(짝수 라운드 치환계층)
__device__ void Add_SubstEven_Layer(uint8_t* input, uint8_t* key) {
	*(input + 0) = ARIA_RS1[*(input + 0) ^ *(key + 0)];
	*(input + 1) = ARIA_RS2[*(input + 1) ^ *(key + 1)];
	*(input + 2) = ARIA_S1[*(input + 2) ^ *(key + 2)];
	*(input + 3) = ARIA_S2[*(input + 3) ^ *(key + 3)];
	*(input + 4) = ARIA_RS1[*(input + 4) ^ *(key + 4)];
	*(input + 5) = ARIA_RS2[*(input + 5) ^ *(key + 5)];
	*(input + 6) = ARIA_S1[*(input + 6) ^ *(key + 6)];
	*(input + 7) = ARIA_S2[*(input + 7) ^ *(key + 7)];
	*(input + 8) = ARIA_RS1[*(input + 8) ^ *(key + 8)];
	*(input + 9) = ARIA_RS2[*(input + 9) ^ *(key + 9)];
	*(input + 10) = ARIA_S1[*(input + 10) ^ *(key + 10)];
	*(input + 11) = ARIA_S2[*(input + 11) ^ *(key + 11)];
	*(input + 12) = ARIA_RS1[*(input + 12) ^ *(key + 12)];
	*(input + 13) = ARIA_RS2[*(input + 13) ^ *(key + 13)];
	*(input + 14) = ARIA_S1[*(input + 14) ^ *(key + 14)];
	*(input + 15) = ARIA_S2[*(input + 15) ^ *(key + 15)];
}

//확산계층
__device__ void DiffLayer(uint8_t* input) {
	uint8_t temp[16] = { 0x00, };//행렬곱셈 결과 담아놓을 변수
	*(temp + 0) = *(input + 3) ^ *(input + 4) ^ *(input + 6) ^ *(input + 8) ^ *(input + 9) ^ *(input + 13) ^ *(input + 14);
	*(temp + 1) = *(input + 2) ^ *(input + 5) ^ *(input + 7) ^ *(input + 8) ^ *(input + 9) ^ *(input + 12) ^ *(input + 15);
	*(temp + 2) = *(input + 1) ^ *(input + 4) ^ *(input + 6) ^ *(input + 10) ^ *(input + 11) ^ *(input + 12) ^ *(input + 15);
	*(temp + 3) = *(input + 0) ^ *(input + 5) ^ *(input + 7) ^ *(input + 10) ^ *(input + 11) ^ *(input + 13) ^ *(input + 14);
	*(temp + 4) = *(input + 0) ^ *(input + 2) ^ *(input + 5) ^ *(input + 8) ^ *(input + 11) ^ *(input + 14) ^ *(input + 15);
	*(temp + 5) = *(input + 1) ^ *(input + 3) ^ *(input + 4) ^ *(input + 9) ^ *(input + 10) ^ *(input + 14) ^ *(input + 15);
	*(temp + 6) = *(input + 0) ^ *(input + 2) ^ *(input + 7) ^ *(input + 9) ^ *(input + 10) ^ *(input + 12) ^ *(input + 13);
	*(temp + 7) = *(input + 1) ^ *(input + 3) ^ *(input + 6) ^ *(input + 8) ^ *(input + 11) ^ *(input + 12) ^ *(input + 13);
	*(temp + 8) = *(input + 0) ^ *(input + 1) ^ *(input + 4) ^ *(input + 7) ^ *(input + 10) ^ *(input + 13) ^ *(input + 15);
	*(temp + 9) = *(input + 0) ^ *(input + 1) ^ *(input + 5) ^ *(input + 6) ^ *(input + 11) ^ *(input + 12) ^ *(input + 14);
	*(temp + 10) = *(input + 2) ^ *(input + 3) ^ *(input + 5) ^ *(input + 6) ^ *(input + 8) ^ *(input + 13) ^ *(input + 15);
	*(temp + 11) = *(input + 2) ^ *(input + 3) ^ *(input + 4) ^ *(input + 7) ^ *(input + 9) ^ *(input + 12) ^ *(input + 14);
	*(temp + 12) = *(input + 1) ^ *(input + 2) ^ *(input + 6) ^ *(input + 7) ^ *(input + 9) ^ *(input + 11) ^ *(input + 12);
	*(temp + 13) = *(input + 0) ^ *(input + 3) ^ *(input + 6) ^ *(input + 7) ^ *(input + 8) ^ *(input + 10) ^ *(input + 13);
	*(temp + 14) = *(input + 0) ^ *(input + 3) ^ *(input + 4) ^ *(input + 5) ^ *(input + 9) ^ *(input + 11) ^ *(input + 14);
	*(temp + 15) = *(input + 1) ^ *(input + 2) ^ *(input + 4) ^ *(input + 5) ^ *(input + 8) ^ *(input + 10) ^ *(input + 15);
	//결과처리
	memcpy(input, temp, 16 * sizeof(uint8_t));
}


//128-bit Left ROTATE 함수
//ROT_R은 ROT_L(N)=ROT_R(128-N)성질 이용할것!
__device__ void Rot_L(uint8_t* input, uint8_t* output, int num) {
	uint8_t copy_input[16] = { 0x00, };
	memcpy(copy_input, input, 16 * sizeof(uint8_t));
	int r = num & 0x07;//%8
	int q = num >> 3;
	uint8_t temp[16] = { 0x00, };
	memcpy(temp, input, 16 * sizeof(uint8_t));
	if (r == 0) {
		*(copy_input + 0) = *(temp + (0 + q) % BLOCKLEN);
		*(copy_input + 1) = *(temp + (1 + q) % BLOCKLEN);
		*(copy_input + 2) = *(temp + (2 + q) % BLOCKLEN);
		*(copy_input + 3) = *(temp + (3 + q) % BLOCKLEN);
		*(copy_input + 4) = *(temp + (4 + q) % BLOCKLEN);
		*(copy_input + 5) = *(temp + (5 + q) % BLOCKLEN);
		*(copy_input + 6) = *(temp + (6 + q) % BLOCKLEN);
		*(copy_input + 7) = *(temp + (7 + q) % BLOCKLEN);
		*(copy_input + 8) = *(temp + (8 + q) % BLOCKLEN);
		*(copy_input + 9) = *(temp + (9 + q) % BLOCKLEN);
		*(copy_input + 10) = *(temp + (10 + q) % BLOCKLEN);
		*(copy_input + 11) = *(temp + (11 + q) % BLOCKLEN);
		*(copy_input + 12) = *(temp + (12 + q) % BLOCKLEN);
		*(copy_input + 13) = *(temp + (13 + q) % BLOCKLEN);
		*(copy_input + 14) = *(temp + (14 + q) % BLOCKLEN);
		*(copy_input + 15) = *(temp + (15 + q) % BLOCKLEN);
	}
	else {
		*(copy_input + 0) = *(temp + (0 + q) % BLOCKLEN) << r | (*(temp + (0 + q + 1) % BLOCKLEN) >> (8 - r));
		*(copy_input + 1) = *(temp + (1 + q) % BLOCKLEN) << r | (*(temp + (1 + q + 1) % BLOCKLEN) >> (8 - r));
		*(copy_input + 2) = *(temp + (2 + q) % BLOCKLEN) << r | (*(temp + (2 + q + 1) % BLOCKLEN) >> (8 - r));
		*(copy_input + 3) = *(temp + (3 + q) % BLOCKLEN) << r | (*(temp + (3 + q + 1) % BLOCKLEN) >> (8 - r));
		*(copy_input + 4) = *(temp + (4 + q) % BLOCKLEN) << r | (*(temp + (4 + q + 1) % BLOCKLEN) >> (8 - r));
		*(copy_input + 5) = *(temp + (5 + q) % BLOCKLEN) << r | (*(temp + (5 + q + 1) % BLOCKLEN) >> (8 - r));
		*(copy_input + 6) = *(temp + (6 + q) % BLOCKLEN) << r | (*(temp + (6 + q + 1) % BLOCKLEN) >> (8 - r));
		*(copy_input + 7) = *(temp + (7 + q) % BLOCKLEN) << r | (*(temp + (7 + q + 1) % BLOCKLEN) >> (8 - r));
		*(copy_input + 8) = *(temp + (8 + q) % BLOCKLEN) << r | (*(temp + (8 + q + 1) % BLOCKLEN) >> (8 - r));
		*(copy_input + 9) = *(temp + (9 + q) % BLOCKLEN) << r | (*(temp + (9 + q + 1) % BLOCKLEN) >> (8 - r));
		*(copy_input + 10) = *(temp + (10 + q) % BLOCKLEN) << r | (*(temp + (10 + q + 1) % BLOCKLEN) >> (8 - r));
		*(copy_input + 11) = *(temp + (11 + q) % BLOCKLEN) << r | (*(temp + (11 + q + 1) % BLOCKLEN) >> (8 - r));
		*(copy_input + 12) = *(temp + (12 + q) % BLOCKLEN) << r | (*(temp + (12 + q + 1) % BLOCKLEN) >> (8 - r));
		*(copy_input + 13) = *(temp + (13 + q) % BLOCKLEN) << r | (*(temp + (13 + q + 1) % BLOCKLEN) >> (8 - r));
		*(copy_input + 14) = *(temp + (14 + q) % BLOCKLEN) << r | (*(temp + (14 + q + 1) % BLOCKLEN) >> (8 - r));
		*(copy_input + 15) = *(temp + (15 + q) % BLOCKLEN) << r | (*(temp + (15 + q + 1) % BLOCKLEN) >> (8 - r));
	}
	memcpy(output, copy_input, BLOCKLEN * sizeof(uint8_t));
}

//rotate시키고 xor까지 실행
//parameter 정보(parameter 순서대로)
//rotate시킬값, 앞의값에 xor해줄값, 결과값, rotate횟수(R_ROTATE라면 128-num)
__device__ void ROT_XOR(uint8_t* input, uint8_t* op, uint8_t* output, int num) {
	uint8_t temp[16] = { 0x00, };
	Rot_L(input, temp, num);
	*(output + 0) = *(temp + 0) ^ *(op + 0);
	*(output + 1) = *(temp + 1) ^ *(op + 1);
	*(output + 2) = *(temp + 2) ^ *(op + 2);
	*(output + 3) = *(temp + 3) ^ *(op + 3);
	*(output + 4) = *(temp + 4) ^ *(op + 4);
	*(output + 5) = *(temp + 5) ^ *(op + 5);
	*(output + 6) = *(temp + 6) ^ *(op + 6);
	*(output + 7) = *(temp + 7) ^ *(op + 7);
	*(output + 8) = *(temp + 8) ^ *(op + 8);
	*(output + 9) = *(temp + 9) ^ *(op + 9);
	*(output + 10) = *(temp + 10) ^ *(op + 10);
	*(output + 11) = *(temp + 11) ^ *(op + 11);
	*(output + 12) = *(temp + 12) ^ *(op + 12);
	*(output + 13) = *(temp + 13) ^ *(op + 13);
	*(output + 14) = *(temp + 14) ^ *(op + 14);
	*(output + 15) = *(temp + 15) ^ *(op + 15);
}


//암호화 과정 키스케줄
//mk는 마스터키 -> 마스터키로부터 12R rk생성
__global__ void Enc_KEY_Expansion(uint8_t* mk, uint8_t* rk) { //rk[13][16] -> rk[208]
	//initial part 
	uint8_t KL[16] = { 0x00, };
	memcpy(KL, mk, 16 * sizeof(uint8_t));

	uint8_t CK1[16] = { 0x51,0x7c,0xc1,0xb7,0x27,0x22,0x0a,0x94,0xfe,0x13,0xab,0xe8,0xfa,0x9a,0x6e,0xe0 };
	uint8_t CK2[16] = { 0x6d,0xb1,0x4a,0xcc,0x9e,0x21,0xc8,0x20,0xff,0x28,0xb1,0xd5,0xef,0x5d,0xe2,0xb0 };
	uint8_t CK3[16] = { 0xdb,0x92,0x37,0x1d,0x21,0x26,0xe9,0x70,0x03,0x24,0x97,0x75,0x04,0xe8,0xc9,0x0e };

	//W0
	uint8_t W0[16] = { 0x00, };
	memcpy(W0, KL, 16 * sizeof(uint8_t));
	//W1
	uint8_t copy_W0[16] = { 0x00, };
	memcpy(copy_W0, W0, 16 * sizeof(uint8_t));
	uint8_t W1[16] = { 0x00, };
	Round_odd(copy_W0, CK1);
	memcpy(W1, copy_W0, 16 * sizeof(uint8_t));
	//W2
	uint8_t W2[16] = { 0x00, };
	uint8_t copy_W1[16] = { 0x00, };
	memcpy(copy_W1, W1, 16 * sizeof(uint8_t));
	Round_even(copy_W1, CK2);
	*(W2 + 0) = *(copy_W1 + 0) ^ *(W0 + 0);
	*(W2 + 1) = *(copy_W1 + 1) ^ *(W0 + 1);
	*(W2 + 2) = *(copy_W1 + 2) ^ *(W0 + 2);
	*(W2 + 3) = *(copy_W1 + 3) ^ *(W0 + 3);
	*(W2 + 4) = *(copy_W1 + 4) ^ *(W0 + 4);
	*(W2 + 5) = *(copy_W1 + 5) ^ *(W0 + 5);
	*(W2 + 6) = *(copy_W1 + 6) ^ *(W0 + 6);
	*(W2 + 7) = *(copy_W1 + 7) ^ *(W0 + 7);
	*(W2 + 8) = *(copy_W1 + 8) ^ *(W0 + 8);
	*(W2 + 9) = *(copy_W1 + 9) ^ *(W0 + 9);
	*(W2 + 10) = *(copy_W1 + 10) ^ *(W0 + 10);
	*(W2 + 11) = *(copy_W1 + 11) ^ *(W0 + 11);
	*(W2 + 12) = *(copy_W1 + 12) ^ *(W0 + 12);
	*(W2 + 13) = *(copy_W1 + 13) ^ *(W0 + 13);
	*(W2 + 14) = *(copy_W1 + 14) ^ *(W0 + 14);
	*(W2 + 15) = *(copy_W1 + 15) ^ *(W0 + 15);
	//W3
	uint8_t W3[16] = { 0x00, };
	uint8_t copy_W2[16] = { 0x00, };
	memcpy(copy_W2, W2, 16 * sizeof(uint8_t));
	Round_odd(copy_W2, CK3);
	*(W3 + 0) = *(copy_W2 + 0) ^ *(W1 + 0);
	*(W3 + 1) = *(copy_W2 + 1) ^ *(W1 + 1);
	*(W3 + 2) = *(copy_W2 + 2) ^ *(W1 + 2);
	*(W3 + 3) = *(copy_W2 + 3) ^ *(W1 + 3);
	*(W3 + 4) = *(copy_W2 + 4) ^ *(W1 + 4);
	*(W3 + 5) = *(copy_W2 + 5) ^ *(W1 + 5);
	*(W3 + 6) = *(copy_W2 + 6) ^ *(W1 + 6);
	*(W3 + 7) = *(copy_W2 + 7) ^ *(W1 + 7);
	*(W3 + 8) = *(copy_W2 + 8) ^ *(W1 + 8);
	*(W3 + 9) = *(copy_W2 + 9) ^ *(W1 + 9);
	*(W3 + 10) = *(copy_W2 + 10) ^ *(W1 + 10);
	*(W3 + 11) = *(copy_W2 + 11) ^ *(W1 + 11);
	*(W3 + 12) = *(copy_W2 + 12) ^ *(W1 + 12);
	*(W3 + 13) = *(copy_W2 + 13) ^ *(W1 + 13);
	*(W3 + 14) = *(copy_W2 + 14) ^ *(W1 + 14);
	*(W3 + 15) = *(copy_W2 + 15) ^ *(W1 + 15);
	//result part
	ROT_XOR(W1, W0, rk, 128 - 19);
	ROT_XOR(W2, W1, rk + 16, 128 - 19);
	ROT_XOR(W3, W2, rk + 32, 128 - 19);
	ROT_XOR(W0, W3, rk + 48, 128 - 19);
	ROT_XOR(W1, W0, rk + 64, 128 - 31);
	ROT_XOR(W2, W1, rk + 80, 128 - 31);
	ROT_XOR(W3, W2, rk + 96, 128 - 31);
	ROT_XOR(W0, W3, rk + 112, 128 - 31);
	ROT_XOR(W1, W0, rk + 128, 61);
	ROT_XOR(W2, W1, rk + 144, 61);
	ROT_XOR(W3, W2, rk + 160, 61);
	ROT_XOR(W0, W3, rk + 176, 61);
	ROT_XOR(W1, W0, rk + 192, 31);
}

//홀수 라운드함수
__device__ void Round_odd(uint8_t* input, uint8_t* rk) {
	Add_SubstOdd_Layer(input, rk);
	DiffLayer(input);
}

//짝수 라운드함수
__device__ void Round_even(uint8_t* input, uint8_t* rk) {
	Add_SubstEven_Layer(input, rk);
	DiffLayer(input);
}

//마지막 라운드 함수
__device__ void Final_Round(uint8_t* input, uint8_t* key12, uint8_t* key13) {
	Add_SubstEven_Layer(input, key12);
	//마지막 라운드키와 XOR
	*(input + 0) ^= *(key13 + 0);
	*(input + 1) ^= *(key13 + 1);
	*(input + 2) ^= *(key13 + 2);
	*(input + 3) ^= *(key13 + 3);
	*(input + 4) ^= *(key13 + 4);
	*(input + 5) ^= *(key13 + 5);
	*(input + 6) ^= *(key13 + 6);
	*(input + 7) ^= *(key13 + 7);
	*(input + 8) ^= *(key13 + 8);
	*(input + 9) ^= *(key13 + 9);
	*(input + 10) ^= *(key13 + 10);
	*(input + 11) ^= *(key13 + 11);
	*(input + 12) ^= *(key13 + 12);
	*(input + 13) ^= *(key13 + 13);
	*(input + 14) ^= *(key13 + 14);
	*(input + 15) ^= *(key13 + 15);
}


//ARIA128 암호화 함수
__device__ void ARIA_EnCrypt(uint8_t* pt, uint8_t* ct, uint8_t* rk) {
	uint8_t copy_pt[16] = { 0x00, };
	memcpy(copy_pt, pt, 16 * sizeof(uint8_t));

	//1R~11R
	Round_odd(copy_pt, rk);
	Round_even(copy_pt, rk + 16);
	Round_odd(copy_pt, rk + 32);
	Round_even(copy_pt, rk + 48);
	Round_odd(copy_pt, rk + 64);
	Round_even(copy_pt, rk + 80);
	Round_odd(copy_pt, rk + 96);
	Round_even(copy_pt, rk + 112);
	Round_odd(copy_pt, rk + 128);
	Round_even(copy_pt, rk + 144);
	Round_odd(copy_pt, rk + 160);

	//12R
	Final_Round(copy_pt, rk + 176, rk + 192);

	//결과처리
	memcpy(ct, copy_pt, 16 * sizeof(uint8_t));
}

//CTR값 +1씩 해주는 함수
__device__ void CTR_INC(uint8_t* ICTR) {
	int cnt_i;
	for (cnt_i = 15; cnt_i > 7; cnt_i--) {
		if (*(ICTR + cnt_i) != 0xff) {
			*(ICTR + cnt_i) += 1;
			break;
		}
		else {
			*(ICTR + cnt_i) = 0x00;
		}
	}
}

//GPU에서 멀티스레드로 암호연산을 병렬적으로 처리해주는 코어 함수
__global__ void GPU_ARIA_CTR(uint8_t* pt, uint8_t* ct, uint8_t* iv, uint32_t enc_block, uint32_t ptlen, uint8_t* rk) {
	int i;
	//CTR IV
	uint8_t ICTR[16] = { 0x00, };
	memcpy(ICTR, iv, 16 * sizeof(uint8_t));

	//temp_pt -> 각 스레드가 pt영역에서 자기가 암호화 해줘야 하는 일부분을 떼와서 저장할 변수
	//temp_ctr -> 각 스레드가 자기가 암호화 해줘야 하는 CTR값을 저장할 변수
	uint8_t temp_pt[16] = { 0x00, };
	uint8_t temp_ctr[16] = { 0x00, };

	//Round Key를 global memory -> shared memory 
	//메모리 접근 속도는 빨라짐
	//bank conflict는 해결X
	__shared__ uint8_t roundkey[208];
	memcpy(roundkey, rk, 208);

	//암호연산 결과를 각 스레드에 맞는 ct에 넣어주기 위한 index
	int index = (threadIdx.x * 16) + (blockDim.x * blockIdx.x * 16);

	//모든 block내 스레드들을 순차적으로 번호를 매겨서 부여
	int thread_num = threadIdx.x + (blockDim.x * blockIdx.x);

	
	//각 스레드 자기 전체적인 스레드번호에 맞게 CTR_INC 수행
	for (i = 0; i < thread_num; i++) {
		CTR_INC(ICTR);
	}

	//각 스레드가 자기한테 알맞은 CounTer값을 암호화
	ARIA_EnCrypt(ICTR, temp_ctr, roundkey);

	//각 스레드 pt에서 자기가 암호화 해야하는 영역 떼오는 작업 
	for (i = 0; i < 16; i++) {
		*(temp_pt + i) = *(pt + (enc_block * i + thread_num));
	}

	//각 스레드 자기가 암호화 해야하는 영역을 xor해주는 작업 
	for (i = 0; i < 16; i++) {
		*(ct + (index + i)) = *(temp_pt + i) ^ *(temp_ctr + i);
	}
}

//GPU내에서 멀티스레드로 병렬적으로 암호를 연산하고 나서의 결과값을 
//츨력해주는 함수
void printstate(uint8_t* data, int dataLen) {
	int i;
	for (i = 0; i < dataLen; i++) {
		if (i != 0 && i % 16 == 0) {
			printf("\n");
		}
		printf("%02x ", data[i]);
	}
}

int main() {
	//암호화할 data & datalen
	char* MSG = "Hello every body rbrbrb hahaha hahaha hanghanghang hihihi nice to meet you hahaha i am fine thank you ha window linux gpu cpu mac os network database";
	uint32_t msg_size = (uint32_t)strlen(MSG);//'\0'문자 제외한 길이 return

	//MasterKey in CPU
	uint8_t CPU_MasterKey[16] = { 0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff };
	//Counter IV
	uint8_t IV[16] = { 0x00, };

	//block -> 메세지 사이즈에 맞게 암호화할 블록 수 구해 줌
	//blocksize -> 암호화 해야할 블록수에 맞게 cuda block 사이즈 정해줌
	//threadsize 32로 고정(warp단위 작동 고려해서 설정)
	int block = ((msg_size - 1) >> 4) + 1;//ceiling 기법으로 구해줌
	int blocksize = ((block - 1) >> 5) + 1;//ceiling 
	int threadsize = 32;

	//control divergence 없애기 위해서 16*32의 배수로 메세지크기 재조정
	//new_MSG에는 dummy data가 들어있음
	//실제 data 값을 제외한 dummy data는 연산하고 나서
	//쓰지 않을 거임!
	uint32_t new_size = blocksize << 9;//blocksize * 16 * 32
	char* new_MSG = (char*)calloc(new_size, sizeof(uint8_t));
	assert(new_MSG != NULL);
	memcpy(new_MSG, MSG, msg_size * sizeof(uint8_t));

	//기존 메세지를 memory coalescing기법을 적용할 수 있는 메모리 배열로 바꿔주기 위한
	//새로운 메모리 할당
	char* COL_MSG = (char*)calloc(new_size, sizeof(uint8_t));
	assert(COL_MSG != NULL);

	int i;

	//기존 메세지 -> memory coalescing기법을 적용할 수 있는 메모리 배열
	//block(암호화 할 block) 0번 index ~ (n-1)번 index 까지 차례대로 재배열
	for (i = 0; i < block; i++) {
		*(COL_MSG + (0 * block + i)) = *(new_MSG + (16 * i + 0));
		*(COL_MSG + (1 * block + i)) = *(new_MSG + (16 * i + 1));
		*(COL_MSG + (2 * block + i)) = *(new_MSG + (16 * i + 2));
		*(COL_MSG + (3 * block + i)) = *(new_MSG + (16 * i + 3));
		*(COL_MSG + (4 * block + i)) = *(new_MSG + (16 * i + 4));
		*(COL_MSG + (5 * block + i)) = *(new_MSG + (16 * i + 5));
		*(COL_MSG + (6 * block + i)) = *(new_MSG + (16 * i + 6));
		*(COL_MSG + (7 * block + i)) = *(new_MSG + (16 * i + 7));
		*(COL_MSG + (8 * block + i)) = *(new_MSG + (16 * i + 8));
		*(COL_MSG + (9 * block + i)) = *(new_MSG + (16 * i + 9));
		*(COL_MSG + (10 * block + i)) = *(new_MSG + (16 * i + 10));
		*(COL_MSG + (11 * block + i)) = *(new_MSG + (16 * i + 11));
		*(COL_MSG + (12 * block + i)) = *(new_MSG + (16 * i + 12));
		*(COL_MSG + (13 * block + i)) = *(new_MSG + (16 * i + 13));
		*(COL_MSG + (14 * block + i)) = *(new_MSG + (16 * i + 14));
		*(COL_MSG + (15 * block + i)) = *(new_MSG + (16 * i + 15));
	}

	//암호화 연산 이후 결과값을 담아올 변수
	//개선점: 동적 할당은 속도적인 측면에서 좋지 않음
	//미리 stack에 충분한 크기의 메모리를 할당 후 필요한 만큼 쓰는 방식이
	//속도적인 측면에서는 더 효율적!!!
	uint8_t* enc_msg = (uint8_t*)calloc(msg_size, sizeof(uint8_t));
	assert(enc_msg != NULL);

	//GPU에서 사용할 메모리 선언 및 할당
	uint8_t* GPU_MasterKey = NULL;
	uint8_t* GPU_PT = NULL;
	uint8_t* GPU_ROUNDKEY = NULL;
	uint8_t* GPU_CT = NULL;
	uint8_t* GPU_IV = NULL;
	cudaMalloc((void**)&GPU_PT, new_size * sizeof(uint8_t));
	cudaMalloc((void**)&GPU_ROUNDKEY, 208 * sizeof(uint8_t));
	cudaMalloc((void**)&GPU_CT, new_size * sizeof(uint8_t));
	cudaMalloc((void**)&GPU_IV, 16 * sizeof(uint8_t));
	cudaMalloc((void**)&GPU_MasterKey, 16 * sizeof(uint8_t));

	//CPU영역에 있는 데이타 -> GPU영역의 변수들에 할당
	cudaMemcpy(GPU_PT, COL_MSG, new_size * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(GPU_IV, IV, 16 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(GPU_MasterKey, CPU_MasterKey, 16 * sizeof(uint8_t), cudaMemcpyHostToDevice);

	//GPU상에서 ROUNDKEY 생성
	Enc_KEY_Expansion<<<1,1>>>(GPU_MasterKey, GPU_ROUNDKEY);

	//GPU에서 암호화 연산 진행할 수 있도록
	//KERNEL 함수 호출!
	GPU_ARIA_CTR<<<blocksize, threadsize>>>(GPU_PT, GPU_CT, GPU_IV, block, msg_size, GPU_ROUNDKEY);

	//GPU에서 연산하고 나온 결과값 -> CPU로 옮기자!
	//위에서 control divergence를 피하기 위해서 사용한 dummy data 부분은 잘라내고
	//실제 유효한 데이타 크기인 msg_size만큼만 잘라서 올바른 연산값만 복사!
	cudaMemcpy(enc_msg, GPU_CT, msg_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

	//결과값(enc_msg)출력 part
	printstate(enc_msg, msg_size);

	//메모리 해제 부분
	cudaFree(GPU_PT);
	cudaFree(GPU_CT);
	cudaFree(GPU_ROUNDKEY);
	cudaFree(GPU_IV);
	cudaFree(GPU_MasterKey);
	
	free(enc_msg);
	free(COL_MSG);
	free(new_MSG);

	return 0;
}