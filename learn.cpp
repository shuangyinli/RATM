/*
 * ratm-learn.cpp
 */

#include "learn.h"

void initPi(double* pi, int win) {
    for (int i = 0; i < win; i++) {
        pi[i] = util::random() * 2;
    }
}

double getPiFunction(senDocument** corpus, Model* model) {
    double pi_function_value = 0.0;
    int num_docs = model->num_docs;
    double* pi = model->pi;
    for (int d = 0; d < num_docs; d++) {
    	 senDocument* doc = corpus[d];
    	for(int s=0; s< doc->num_sentences; s++){

    		Sentence* sentence = doc->sentences[s];
    		double sigma_pi = 0.0;
    		double sigma_xi = 0.0;
    		int win = sentence->win;
    		for (int i = 0; i < win; i++) {
    		            sigma_pi += pi[i];
    		            sigma_xi += sentence->xi[i];
    		}

    		pi_function_value += util::log_gamma(sigma_pi);
    		for (int i = 0; i < win; i++) {
    		            pi_function_value -= util::log_gamma(pi[i]);
    		            pi_function_value += (pi[i] - 1) * (util::digamma(sentence->xi[i]) - util::digamma(sigma_xi));
    		}
    	}
    }
    return pi_function_value;
}

void getDescentPi(senDocument** corpus, Model* model, double* descent_pi) {
    int win = model->win;
    int num_docs = model->num_docs;
    memset(descent_pi,0,sizeof(double)* win);
    double* pi = model->pi;
    for (int d = 0; d < num_docs; d++) {
    	 senDocument* doc = corpus[d];
    	for(int s = 0; s<doc->num_sentences; s++){
    		Sentence* sentence = doc->sentences[s];

    		double sigma_pi = 0.0;
    		 double sigma_xi = 0.0;
    		 for (int i = 0; i < win; i++) {
    		             sigma_pi += pi[i];
    		             sigma_xi += sentence->xi[i];
    		 }

    		 for (int i = 0; i < win; i++) {
    		            double pis = pi[i];
    		            descent_pi[i] += util::digamma(sigma_pi) - util::digamma(pis) + util::digamma(sentence->xi[i]) - util::digamma(sigma_xi);
    		  }

    	}
    }
}

void learnPi(senDocument** corpus, Model* model, Configuration* configuration) {
    int num_round = 0;
    int win = model->win;
    double* last_pi = new double [model->win];
    double* descent_pi = new double[win];
    double z;
    int num_wait_for_z = 0;
    do {
    	initPi(model->pi,win);
        z = getPiFunction(corpus,model);
        fprintf(stderr, "wait for z >=0\n");
        num_wait_for_z ++;
    }
    while ( z < 0 && num_wait_for_z <= 20);
    double last_z;
    double learn_rate = configuration->pi_learn_rate;
    double eps = 1000;
    int max_pi_iter = configuration->max_pi_iter;
    double pi_min_eps = configuration->pi_min_eps;
    bool has_neg_value_flag = false;
    do {
        last_z = z;
        memcpy(last_pi,model->pi,sizeof(double) * win);
        getDescentPi(corpus,model,descent_pi);
        for (int i = 0; !has_neg_value_flag && i < win; i++) {
            model->pi[i] += learn_rate * descent_pi[i];
            if (model->pi[i] < 0) has_neg_value_flag = true;
        }
        if (has_neg_value_flag || last_z > (z=getPiFunction(corpus,model))) {
            learn_rate *= 0.1;
            z = last_z;
            //for ( int i = 0; i < num_labels; i++) pi[i] = last_pi[i];
            memcpy(model->pi,last_pi,sizeof(double) * win);
            eps = 1000.0;
        }
        else eps = util::norm2(last_pi, model->pi, win);
        num_round += 1;
    }
    while (num_round < max_pi_iter && eps > pi_min_eps);
    delete[] last_pi;
    delete[] descent_pi;
}

void learnBeta(senDocument** corpus, Model* model) {
	int num_docs = model->num_docs;
	int num_topics = model->num_topics;
	int num_words = model->num_words;
	bool* reset_beta_flag = new bool[num_topics * model->num_words];
	memset(reset_beta_flag, 0, sizeof(bool) * num_topics * model->num_words);
	for (int d = 0; d < num_docs; d++) {

		for(int s = 0; s<corpus[d]->num_sentences;s++){
			Sentence* sentence = corpus[d]->sentences[s];
			int sen_num_words = sentence->num_words;
			for (int k = 0; k < num_topics; k++) {
						for (int i = 0; i < sen_num_words; i++) {
							int wordid = sentence->words_ptr[i];
							if (!reset_beta_flag[k * num_words + wordid]) {
								reset_beta_flag[k * num_words + wordid] = true;
								model->log_beta[k * num_words + wordid] = log(sentence->words_cnt_ptr[i])+ sentence->log_gamma[i * num_topics + k];
							} else {
								model->log_beta[k * num_words + wordid] = util::log_sum(
										model->log_beta[k * num_words + wordid],
										sentence->log_gamma[i * num_topics + k]
												+ log(sentence->words_cnt_ptr[i]));
							}
							/*if (isnan(doc->log_gamma[i*num_topics +k]) || isnan(model->log_phi[k * num_words + wordid])) {
							 printf("%lf %lf\n",doc->log_gamma[i*num_topics +k], model->log_phi[k * num_words + wordid]);
							 }*/
						}
					}
		}

	}
	normalize_log_matrix_rows(model->log_beta, num_topics, model->num_words);
	delete[] reset_beta_flag;
}

//////////////////////////////////////


void init_pi(double* pi, int num_labels) {
    for (int i = 0; i < num_labels; i++) {
        pi[i] = util::random() * 2;
    }
}



void normalize_matrix_rows(double* mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        double temp = 0;
        for (int j = 0; j < cols; j++) temp += mat[ i * cols + j];
        for (int j = 0; j < cols; j++) {
            mat[i*cols +j] /= temp;
            if (mat[i*cols + j] == 0)mat[i*cols + j] = 1e-300;
        }
    }
}

void normalize_log_matrix_rows(double* log_mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        double temp = log_mat[ i * cols];
        /*if (isnan(temp) || isnan(-temp)) {
            printf("normalize nan\n");
        }*/
        for (int j = 1; j < cols; j++) temp = util::log_sum(temp, log_mat[i * cols + j]);
        /*if (isnan(-temp) || isnan(temp)) {
            printf("normalize nan\n");
        }*/
        for (int j = 0; j < cols; j++) log_mat[i*cols + j] -= temp;
    }
}



