
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_DATA 128

//디바이스 코드... gpu에서 각각의 스레드들이 실행할 명령
__device__ void deviceVecAdd(int* a, int* b, int* c, int tID) {
    if (tID < NUM_DATA) {
        c[tID] = a[tID] + b[tID];
        /*printf("%d + %d = %d from thread %d\n", a[tID], b[tID], c[tID], tID);*/
    }
}

// GPU에서 실행되는 커널 함수
__global__ void vecAdd(int* a, int* b, int* c) {
    int tID = threadIdx.x;  // 스레드 ID를 글로벌에서 생성해서 디바이스코드로 전달하기 위해 threadIdx.x를 사용
    deviceVecAdd(a, b, c, tID);
}

int main(void) {
    int* a, * b, * c;         // 호스트 메모리 포인터
    int* d_a, * d_b, * d_c;   // 디바이스 메모리 포인터
    int memSize = sizeof(int) * NUM_DATA; //메모리사이즈 할당 위해 계산... int => 4byte => 32bit => -2^31 ~ -2^31 + 1 범위가짐

    printf("%d elements, memSize = %d bytes\n", NUM_DATA, memSize);

    // 호스트 메모리 할당 및 초기화
    a = new int[NUM_DATA];
    b = new int[NUM_DATA];
    c = new int[NUM_DATA];
    for (int i = 0; i < NUM_DATA; i++) {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }

    // 디바이스 메모리 할당
    cudaMalloc(&d_a, memSize);
    cudaMalloc(&d_b, memSize);
    cudaMalloc(&d_c, memSize);

    // 호스트 메모리 -> 디바이스 메모리 복사
    // cudaMemcpy(목적지, 출발지, 메모리사이즈,..,복사유형..HostToDevice같은..)
    cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);

    // GPU 커널 호출
    vecAdd <<<1, NUM_DATA >>> (d_a, d_b, d_c);
    cudaDeviceSynchronize();  // GPU 연산 완료 대기

    // 디바이스 메모리 -> 호스트 메모리 복사
    cudaMemcpy(c, d_c, memSize, cudaMemcpyDeviceToHost);

    // 결과 출력
    /*printf("Vector Addition Result\n");
    for (int i = 0; i < NUM_DATA; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }*/

    // 디바이스 메모리 해제
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // 호스트 메모리 해제
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}