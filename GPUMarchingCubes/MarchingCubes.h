#pragma once
#include "CUDACheck.h"

using uint = unsigned int;
const int m_uint = sizeof(uint);

void initVoxels(int argc, char **argv);
void preDetermineKernel(const uint &nVoxels);