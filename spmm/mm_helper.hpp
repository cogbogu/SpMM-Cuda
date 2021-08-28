/*
 * mm_helper.hpp
 * Copyright (C) 2020 
 * 	Aravind SUKUMARAN RAJAM (asr) <aravind_sr@outlook.com>
 *
 * Distributed under terms of the GNU LGPL3 license.
 */

#ifndef MM_HELPER_HPP
#define MM_HELPER_HPP

#include "sparse_representation.hpp"

COO read_matrix_market_to_COO(const char* fname);
CSR read_matrix_market_to_CSR(const char* fname);
CSC read_matrix_market_to_CSC(const char* fname);


#endif /* !MM_HELPER_HPP */
