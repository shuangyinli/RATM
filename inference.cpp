/*
 * ratm-inference.cpp
 */
#include "inference.h"

void doInference(senDocument* doc, Model* model, Configuration* configuration) {
    double doc_lik_old = 0.0;
    double doc_lik;
    double doc_var_converence = 1.0;
    int doc_max_var_iter = 0;
    int num_sentence = doc->num_sentences;
    int win = model->win;
    int num_topics = model->num_topics;
    int G0 = model->G0;
    //update the rou of document

    while((doc_var_converence > configuration->doc_var_converence) && (doc_max_var_iter < configuration->doc_max_var_iter)){ //judge the whole doc converged
    	doc_max_var_iter ++;

    	for(int s =0; s<num_sentence; s++){
    		//for each sentence
    		int var_iter = 0;
    		double sen_lik_old = 0.0;
    		double sen_converged = 1.0;
    		double sen_lik;
    		Sentence* sentence = doc->sentences[s];

    		if(G0 == 1){
				for (int i = 0; i < win - 1; i++) {
					for (int k = 0; k < num_topics; k++) {
						sentence->wintopics[i * num_topics + k] = doc->docTopicMatrix[(i + s + 1) * num_topics + k];
					}
				}
				for (int k = 0; k < num_topics; k++) {
					sentence->wintopics[(win-1) * num_topics + k] =doc->doctopic[k];
				}
			}else if (G0 == 0) {
				for (int i = 0; i < win; i++) {
					for (int k = 0; k < num_topics; k++) {
						sentence->wintopics[i * num_topics + k] = doc->docTopicMatrix[(i + s) * num_topics + k];
					}
				}
			}

			double * old_sen_loggamma = new double[sentence->num_words * num_topics];
    		double * old_sen_topics = new double[num_topics];
    		double * old_sen_xi = new double[win];
    		memcpy(old_sen_loggamma,sentence->log_gamma,sizeof(double)*sentence->num_words*num_topics);
    		memcpy(old_sen_topics,sentence->topic,sizeof(double)*num_topics);
    		memcpy(old_sen_xi,sentence->xi,sizeof(double)*win);

    		while ((sen_converged > configuration->sen_var_converence) && ((var_iter < configuration->sen_max_var_iter || configuration->sen_max_var_iter == -1))) {
    		        var_iter ++;
    		        inferenceXi(sentence,model, configuration);
    		        inferenceGamma(sentence, model);
    		        sen_lik = inference_sen_likelihood(sentence,model);
    		        sentence->senlik = sen_lik;
    		        sen_converged = (sen_lik_old -sen_lik) / sen_lik_old;
    		        if(sen_converged <0){
    		        	memcpy(sentence->log_gamma, old_sen_loggamma,sizeof(double)*sentence->num_words*num_topics);
    					memcpy(sentence->topic, old_sen_topics,sizeof(double)*num_topics);
    					memcpy(sentence->xi, old_sen_xi,sizeof(double)*win);
    					sentence->senlik = sen_lik;
    					break;
    		        }

					memcpy(old_sen_loggamma,sentence->log_gamma,sizeof(double)*sentence->num_words*num_topics);
    				memcpy(old_sen_topics,sentence->topic,sizeof(double)*num_topics);
    				memcpy(old_sen_xi,sentence->xi,sizeof(double)*win);

    		        sen_lik_old = sen_lik;
    		    }
    		for(int k=0; k<num_topics; k++){
    			//update the log topic distribution of the current sentence
    			doc->docTopicMatrix[(s+win)*num_topics + k] = sentence->topic[k];
    		}
    		delete[] old_sen_loggamma;
    		delete[] old_sen_topics;
    		delete[] old_sen_xi;

    	}
    	//here do not update the topics of each doc.
    	doc_lik = compute_doc_likelihood(doc, model);
    	doc_var_converence = fabs(doc_lik_old - doc_lik);
    	doc_lik_old=doc_lik;
    }

    // update the Rou and update the topic distritbution of the document
     inferenceRou(doc, model);

    return;
}

void inferenceRou(senDocument* document, Model* model) {
		int num_sentence = document->num_sentences;
	    int win = model->win;
	    int num_topics = model->num_topics;
	    double * doc_rou = document->rou;
	    double * alpha = model->alpha;
	    double sigma_rou = 0.0;
        memset(doc_rou, 0.0, sizeof(double) * num_topics);

	    for(int s =0; s<num_sentence; s++){
	    	Sentence* sentence = document->sentences[s];
	    	int sen_num_words = sentence->num_words;
	    	double * log_gamma = sentence->log_gamma;
	    	double sigma_xi = 0;
	    	for (int i = 0; i < win; i++) sigma_xi += sentence->xi[i];
	    	for (int k = 0; k < num_topics; k++) {
	    			double sigma_gamma = 0.0;
	    			for (int j = 0; j < sen_num_words; j++) {
	    				sigma_gamma += exp(log_gamma[j * num_topics + k]);
	    			}
	    			doc_rou[k] += sigma_gamma * (sentence->xi[win - 1] / sigma_xi);
	    			if (isnan(doc_rou[k]) || isinf(doc_rou[k])) {
	    				printf("rou is nan ");
	    			}
	    		}
	    }

	for (int k = 0; k < num_topics; k++) {
		doc_rou[k] += alpha[k];
		sigma_rou += doc_rou[k];
	}
    // update the document's topic distributions here:
	//compute the topics of the document
	for (int k = 0; k < num_topics; k++) {
		document->doctopic[k] = log(doc_rou[k] / sigma_rou);
	}

}

void inferenceGamma(Sentence* sentence, Model* model) {
		double* log_theta = sentence->wintopics;
	    double* log_phi = model->log_beta;
	    int num_topics = model->num_topics;
	    int num_words = model->num_words;
	    int sen_num_words = sentence->num_words;
	    double* log_gamma = sentence->log_gamma;
	    double* theta_xi = new double[num_topics];
	    double sigma_xi = 0;
	    int win = sentence->win;
	    for (int i = 0; i < win; i++){
	        sigma_xi += sentence->xi[i];
	    }
	    for (int k = 0; k < num_topics; k++) {
	        double temp = 0;
	        for (int i = 0; i < win; i++) {
	            temp += sentence->xi[i]/sigma_xi * log_theta[i*num_topics + k];
	        }
	        theta_xi[k] = temp;
	    }
	    for (int i = 0; i < sen_num_words; i++) {
	        int wordid = sentence->words_ptr[i];
	        double sum_log_gamma = 0;
	        for (int k = 0; k < num_topics; k++) {
	            double temp = log_phi[k * num_words + wordid] + theta_xi[k];
	            log_gamma[ i * num_topics + k] = temp;
	            if (k == 0) sum_log_gamma = temp;
	            else sum_log_gamma = util::log_sum(sum_log_gamma, temp);
	        }
	        for (int k = 0; k < num_topics; k++)log_gamma[i*num_topics + k] -= sum_log_gamma;
	    }
    delete[] theta_xi;
}

void inferenceXi(Sentence* sentence, Model* model,Configuration* configuration) {
    int win = model->win;
    double* descent_xi = new double[win];
    initXi(sentence->xi,win);
    double z = getXiFunction(sentence,model);
    double learn_rate = configuration->xi_learn_rate;
    double eps = 10000;
    int num_round = 0;
    int max_xi_iter = configuration->max_xi_iter;
    double xi_min_eps = configuration->xi_min_eps;
    double last_z;
    double* last_xi = new double[win];
    double * init_xi = new double[win];
    memcpy(init_xi,sentence->xi,sizeof(double)*win);
    do {
        last_z = z;
        memcpy(last_xi,sentence->xi,sizeof(double)*win);
        getDescentXi(sentence,model,descent_xi);

        bool has_neg_value_flag = false;
        for (int i = 0; !has_neg_value_flag && i < win; i++) {
            sentence->xi[i] += learn_rate * descent_xi[i];
            if (sentence->xi[i] < 0)has_neg_value_flag = true;
            if (isnan(-sentence->xi[i]) || isnan(sentence->xi[i]) || isinf(sentence->xi[i]) || isinf(-sentence->xi[i]) ){

            	if (isnan(last_xi[i]) || isnan(last_xi[i])){
            		printf("last xi nan 1 \n");
            		memcpy(sentence->xi,init_xi,sizeof(double)*win);
            	} if (isinf(last_xi[i]) || isinf(last_xi[i])){
            		printf("last xi is inf 1 \n");
            		memcpy(sentence->xi,init_xi,sizeof(double)*win);
            	}else{
            		//printf("last_xi is not  nan or inf 1 \n");
            	    memcpy(sentence->xi,last_xi,sizeof(double)*win);
            	}
            }
        }
        if ( has_neg_value_flag || last_z > (z = getXiFunction(sentence,model))) {
            learn_rate *= 0.1;
            z = last_z;
            eps = 10000;
            memcpy(sentence->xi,last_xi,sizeof(double)*win);
        }
        else eps = util::norm2(last_xi,sentence->xi,win);
        num_round ++;
    }
    while (num_round < max_xi_iter && eps > xi_min_eps);

	for (int i = 0; i < win; i++) {
		 if (isnan(-sentence->xi[i]) || isnan(sentence->xi[i]) || isinf(sentence->xi[i]) || isinf(-sentence->xi[i]) ){
			printf("doc->xi[i] nan here, so back \n");
			memcpy(sentence->xi, init_xi, sizeof(double) * win);
			break;
		}
	}
	delete [] init_xi;
    delete[] last_xi;
    delete[] descent_xi;

}

void getDescentXi(Sentence* sentence, Model* model,double* descent_xi) {
		double sigma_xi = 0.0;
	    double sigma_pi = 0.0;
	    int win = sentence->win;
	    for (int i = 0; i < win; i++) {
	        sigma_xi += sentence->xi[i];
	        sigma_pi += model->pi[i];
	    }
	    for (int i = 0; i < win; i++) {
	        descent_xi[i] = util::trigamma(sentence->xi[i]) * ( model->pi[i] - sentence->xi[i]);
	        descent_xi[i] -= util::trigamma(sigma_xi) * (sigma_pi - sigma_xi);
	    }
	    int sen_num_words = sentence->num_words;
	    int num_topics = model->num_topics;
	    double* log_theta = sentence->wintopics;
	    double* sum_log_theta = new double[num_topics];
	    memset(sum_log_theta, 0, sizeof(double) * num_topics);
	    for (int k = 0; k < num_topics; k++) {
	        sum_log_theta[k] = 0;
	        for (int i = 0; i < win; i++) {
	            sum_log_theta[k] +=log_theta[i * num_topics + k] * sentence->xi[i];
	        }
	    }
	    double* sum_gamma_array = new double[num_topics];
	    for (int k = 0; k < num_topics; k++) {
	        sum_gamma_array[k] = 0;
	        for (int i = 0; i < sen_num_words; i++) {
	            sum_gamma_array[k] += exp(sentence->log_gamma[i * num_topics + k]) * sentence->words_cnt_ptr[i];
	        }
	    }
	    for (int j = 0; j < win; j++) {
	        for (int k = 0; k < num_topics; k++) {
	            double temp = 0;
	            double sum_gamma = 0.0;
	            temp += log_theta[j* num_topics + k] * sigma_xi;
	            sum_gamma = sum_gamma_array[k];
	            temp -= sum_log_theta[k];
	            temp = sum_gamma * (temp/(sigma_xi * sigma_xi));
	            /*if (isnan(temp)) {
	                printf("sum_gamma:%lf temp:%lf descent_xi:%lf\n",sum_gamma,temp,descent_xi[j]);
	            }*/
	            descent_xi[j] += temp;
	        }
	        /*if (isnan(descent_xi[j])) {
	            printf("descent_xi nan\n");
	        }*/
	    }
	    delete[] sum_log_theta;
	    delete[] sum_gamma_array;
}


double getXiFunction(Sentence* sentence, Model* model) {
		double xi_function_value = 0.0;
	    int win = sentence->win;
	    double sigma_xi = 0.0;
	    double* pi = model->pi;
	    double* log_theta = sentence->wintopics;

	    for (int i = 0; i < win; i++) sigma_xi += sentence->xi[i];

	    for (int i = 0; i < win; i++) {
	        xi_function_value += (pi[i] - sentence->xi[i] )* (util::digamma(sentence->xi[i]) - util::digamma(sigma_xi)) + util::log_gamma(sentence->xi[i]);
	    }

	    xi_function_value -= util::log_gamma(sigma_xi);

	    int sen_num_words = sentence->num_words;
	    int num_topics = model->num_topics;

	    double* sum_log_theta = new double[num_topics];
	    for (int k = 0; k < num_topics; k++) {
	        double temp = 0;
	        for(int j = 0; j < win; j++){
	        	temp += log_theta[j * num_topics + k] * sentence->xi[j]/sigma_xi;
	        }
	        sum_log_theta[k] = temp;
	    }

	    for (int i = 0; i < sen_num_words; i++) {
	        for (int k = 0; k < num_topics; k++) {
	            double temp = sum_log_theta[k];
	            xi_function_value += temp * exp(sentence->log_gamma[i * num_topics + k]) * sentence->words_cnt_ptr[i];
	        }
	    }
	    delete[] sum_log_theta;
	    return xi_function_value;
}
inline void initXi(double* xi,int win) {
    for (int i = 0; i < win; i++) xi[i] = util::random();//init 100?!
}

double verifyTestSet(senDocument** test_corpus, Model* model, Configuration* configuration, int test_num_docs) {
    int win = model->win;
    int num_topics = model->num_topics;
    int G0 = model->G0;
    int num_words = model->num_words;
    bool* reset_beta_flag = new bool[num_topics * model->num_words];
    memset(reset_beta_flag, 0, sizeof(bool) * num_topics * model->num_words);
    for(int d=0; d< test_num_docs; d++){
        senDocument* doc = test_corpus[d];
        for(int s =0; s<doc->num_sentences; s++){
            Sentence* sentence = doc->sentences[s];
            if(G0 == 1){
                for (int i = 0; i < win - 1; i++)
                    for (int k = 0; k < num_topics; k++) sentence->wintopics[i * num_topics + k] = doc->docTopicMatrix[(i + s + 1) * num_topics + k];
                for (int k = 0; k < num_topics; k++) sentence->wintopics[(win-1) * num_topics + k] =doc->doctopic[k];
            }else if (G0 == 0) {
                for (int i = 0; i < win; i++)
                    for (int k = 0; k < num_topics; k++) sentence->wintopics[i * num_topics + k] = doc->docTopicMatrix[(i + s) * num_topics + k];  
            }
            inferenceXi(sentence,model, configuration);
            inferenceGamma(sentence, model);
            sentence->senlik = inference_sen_likelihood(sentence,model);
        }
    }
        for (int d = 0; d < test_num_docs; d++) {
            for(int s = 0; s<test_corpus[d]->num_sentences;s++){
                Sentence* sentence = test_corpus[d]->sentences[s];
                for (int k = 0; k < num_topics; k++) {
                    for (int i = 0; i < sentence->num_words; i++) {
                        int wordid = sentence->words_ptr[i];
                        if (!reset_beta_flag[k * num_words + wordid]) {
                        reset_beta_flag[k * num_words + wordid] = true;
                        model->log_beta[k * num_words + wordid] = log(sentence->words_cnt_ptr[i])+ sentence->log_gamma[i * num_topics + k];
                        } 
                        else model->log_beta[k * num_words + wordid] = util::log_sum(model->log_beta[k * num_words + wordid], sentence->log_gamma[i * num_topics + k]+ log(sentence->words_cnt_ptr[i]));
                    }
                }
            }
        }
    normalize_log_matrix_rows(model->log_beta, num_topics, model->num_words);
    delete[] reset_beta_flag;
    return corpuslikelihood(test_corpus, model, test_num_docs);
}

double LDAInference(senDocument** corpus, Model* model, int num_docs) {
    //int num_docs = model->num_docs;
    int num_words = model->num_words;
    int num_topics = model->num_topics;
    double* sum_phi_w = new double[num_words];
    double lik = 0.0;
    //printf("num_docs: %d\nnum_words: %d\nnum_topics: %d\n", num_docs, num_words, num_topics);
    for (int w = 0; w < num_words; w++) {
        sum_phi_w[w] = 0;
        for (int k =0; k < num_topics; k++) sum_phi_w[w] += exp(model->log_beta[k * num_words + w]);
    }

    for (int d = 0; d < num_docs; d++) {
       senDocument* doc = corpus[d];
       int num_sentences = doc->num_sentences;
       
       for(int s = 0; s< num_sentences; s++){
            
    	   Sentence * sentence = doc->sentences[s];
    	   double* topic = sentence->topic;
    	   int sen_num_words = sentence->num_words;
    	   double sum_topic = 0;
    	   for(int k = 0; k < num_topics; k++){
    		   topic[k] = 0;
    		   for (int w = 0; w < sen_num_words; w++) {
    		         int wordid = sentence->words_ptr[w];
    		         topic[k] += exp(model->log_beta[k * num_words + wordid])/sum_phi_w[wordid];
    		   }
    		   sum_topic += topic[k];
    	   }
          
    	   for (int k = 0; k < num_topics; k++) topic[k] /= sum_topic;
           
    	   sentence->senlik = 0;
    	   for (int w = 0; w < sen_num_words; w++) {
    		   int wordid = sentence->words_ptr[w];
    	       double sum_pr = 0;
    	       for (int k = 0; k < num_topics; k++) {
    	    	   sum_pr += topic[k] * exp(model->log_beta[k * num_words + wordid]);
    	       }
    	       sentence->senlik += log(sum_pr);
               lik +=sentence->senlik;
    	   }
           
       }
       

    }
    delete[] sum_phi_w;
    return lik;
}

void* ThreadInference(void* thread_data) {
    ThreadData* thread_data_ptr = (ThreadData*) thread_data;
    senDocument** corpus = thread_data_ptr->corpus;
    int start = thread_data_ptr->start;
    int end = thread_data_ptr->end;
    Configuration* configuration = thread_data_ptr->configuration;
    Model* model = thread_data_ptr->model;
    for (int i = start; i < end; i++) {
        doInference(corpus[i], model, configuration);
    }
    return NULL;
}

void runThreadInference(senDocument** corpus, Model* model, Configuration* configuration, int num_docs) {
    int num_threads = configuration->num_threads;
    pthread_t* pthread_ts = new pthread_t[num_threads];
    int num_per_threads = num_docs/num_threads;
    int i;
    ThreadData** thread_datas = new ThreadData* [num_threads];
    for (i = 0; i < num_threads - 1; i++) {
        thread_datas[i] = new ThreadData(corpus, i * num_per_threads, (i+1)*num_per_threads, configuration, model);;
        pthread_create(&pthread_ts[i], NULL, ThreadInference, (void*) thread_datas[i]);
    }
    thread_datas[i] = new ThreadData(corpus, i * num_per_threads, num_docs, configuration, model);
    pthread_create(&pthread_ts[i], NULL, ThreadInference, (void*) thread_datas[i]);
    for (i = 0; i < num_threads; i++) pthread_join(pthread_ts[i],NULL);
    for (i = 0; i < num_threads; i++) delete thread_datas[i];
    delete[] thread_datas;
}

inline void init_xi(double* xi,int num_labels) {
    for (int i = 0; i < num_labels; i++) xi[i] = util::random();//init 100?!
}

inline bool has_neg_value(double* vec,int dim) {
    for (int i =0; i < dim; i++) {
        if (vec[dim] < 0)return true;
    }
    return false;
}
