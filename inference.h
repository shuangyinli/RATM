/*
 * ratm-inference.h
 */

#ifndef INFERENCE_H_
#define INFERENCE_H_

#include "utils.h"
#include "pthread.h"
#include "unistd.h"
#include "stdlib.h"

#include "ratm.h"
#include "learn.h"


struct ThreadData {
    senDocument** corpus;
    int start;
    int end;
    Configuration* configuration;
    Model* model;
    ThreadData(senDocument** corpus_, int start_, int end_, Configuration* configuration_, Model* model_) : corpus(corpus_), start(start_), end(end_), configuration(configuration_), model(model_) {
    }
};

void inferenceGamma(Sentence* sentence, Model* model);
void inferenceXi(Sentence* sentence, Model* model,Configuration* configuration);
void runThreadInference(senDocument ** corpus, Model* model, Configuration* config, int num_docs);
double LDAInference(senDocument ** corpus, Model* model, int num_docs);
double getXiFunction(Sentence* sentence, Model* model);
void getDescentXi(Sentence* sentence, Model* model,double* descent_xi);
inline void initXi(double* xi,int win);
void inferenceRou(senDocument* document, Model* model);
double verifyTestSet(senDocument** test_corpus, Model* model, Configuration* configuration, int test_num_docs);

#endif /* INFERENCE_H_ */
