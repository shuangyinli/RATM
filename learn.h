/*
 * rtm-learn.h
 *
 *  Created on: Mar 21, 2016
 *      Author: shuangyinli
 */

#ifndef LEARN_H_
#define LEARN_H_

#include "ratm.h"
#include "utils.h"

void normalize_log_matrix_rows(double* log_mat, int rows, int cols) ;
void learnBeta(senDocument** corpus, Model* model, int num_docs);
void learnPi(senDocument** corpus, Model* model, Configuration* configuration, int num_docs);

#endif /* LEARN_H_ */
