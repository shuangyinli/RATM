//============================================================================
// Name        : ratm.cpp
//============================================================================

#include "stdio.h"
#include "string.h"
#include "utils.h"
#include "map"
#include "vector"
#include "set"

#include "inference.h"
#include "learn.h"
#include "ratm.h"
#include "cstdlib"
#include "algorithm"
using namespace std;
/*
	    *
	    *the corpus format is
	    *for each line is a doc with several sentences.
	    *102 @ 4 1:2 23:3 55:11 1345:43 @ 5 11:2 23:3 55:34 1345:1 10:1 @ .... @
	    *[the number of sentences] @ [number of words] [wordid:wordcount] [wordid:wordcount] [wordid:wordcount] @ ... @
	    * */

senDocument ** readData(char* filename, int num_topics,int& num_words, int& num_docs, int& num_all_words, int& win){
	num_words = 0; //keep the total number words in dictionary
	num_docs = 0; //keep the total number documents in the corpus
	num_all_words = 0; // keep the total number words
	FILE* fp = fopen(filename,"r"); //calcaulte the file line num
	    char c;
	    while((c=getc(fp))!=EOF) {
	        if (c=='\n') num_docs++;
	    }
	   fclose(fp);
	   fp = fopen(filename,"r");
       //int doc_num_words;

	   char str[10];
	   senDocument ** corpus = new senDocument * [num_docs];

	   num_docs = 0;
	   int doc_num_sentences;

	    while(fscanf(fp,"%d",&doc_num_sentences) != EOF) {
	    	int doc_num_all_words = 1;
	        Sentence ** sentences = new Sentence * [ doc_num_sentences];
	        for(int i = 0; i < doc_num_sentences; i++){
	        	int word_num_inSen;
	        	fscanf(fp,"%s",str); //read @
	        	fscanf(fp, "%d", &word_num_inSen);
	        	int* words_ptr = new int[word_num_inSen];
	        	int* words_cnt_ptr = new int [word_num_inSen];
	        	for(int w=0; w<word_num_inSen; w++){
	        		fscanf(fp,"%d:%d", &words_ptr[w],&words_cnt_ptr[w]);
	        		num_words = num_words < words_ptr[w]?words_ptr[w]:num_words;
	        		doc_num_all_words += words_cnt_ptr[w];
	        	}
	        	sentences[i] = new Sentence(words_ptr, words_cnt_ptr, word_num_inSen, num_topics, win);
	        }

	        corpus[num_docs++]  = new senDocument(doc_num_all_words, num_topics, win, doc_num_sentences, sentences);
	        num_all_words +=doc_num_all_words;
	    }
	    fclose(fp);
	    num_words += 1;
	    //printf("num_docs: %d\nnum_words:%d\n",num_docs,num_words);
	    return corpus;
}

void readinitParameters(Model* model, char* beta_file){

	FILE* fp_beta = fopen(beta_file, "r");
	double * log_beta_ = model->log_beta;
	int num_topics = model->num_topics;
	int num_words = model->num_words;
	for (int i = 0; i < num_topics; i++) {
		for (int j = 0; j < num_words; j++) {
			fscanf(fp_beta, "%lf", &log_beta_[i * num_words + j]);
		}
	}
	fclose(fp_beta);

}

void Configuration::read_settingfile(char* settingfile){
	FILE* fp = fopen(settingfile,"r");
	    char key[100];
	    char test_action[100];
	    while (fscanf(fp,"%s",key)!=EOF){

                if (strcmp(key,"pi_learn_rate")==0) {
	            fscanf(fp,"%lf",&pi_learn_rate);
	            continue;
	        }
	        if (strcmp(key,"max_pi_iter") == 0) {
	            fscanf(fp,"%d",&max_pi_iter);
	            continue;
	        }
	        if (strcmp(key,"pi_min_eps") == 0) {
	            fscanf(fp,"%lf",&pi_min_eps);
	            continue;
	        }
	        if (strcmp(key,"xi_learn_rate") == 0) {
	            fscanf(fp,"%lf",&xi_learn_rate);
	            continue;
	        }
	        if (strcmp(key,"max_xi_iter") == 0) {
	            fscanf(fp,"%d",&max_xi_iter);
	            continue;
	        }
	        if (strcmp(key,"xi_min_eps") == 0) {
	            fscanf(fp,"%lf",&xi_min_eps);
	            continue;
	        }
	        if (strcmp(key,"max_em_iter") == 0) {
	            fscanf(fp,"%d",&max_em_iter);
	            continue;
	        }
	        if (strcmp(key,"num_threads") == 0) {
	            fscanf(fp, "%d", &num_threads);
	            continue;
	        }
	        if (strcmp(key, "sen_var_converence") == 0) {
	            fscanf(fp, "%lf", &sen_var_converence);
	            continue;
	        }
	        if (strcmp(key, "sen_max_var_iter") == 0) {
	            fscanf(fp, "%d", &sen_max_var_iter);
	            continue;
	        }
	        if (strcmp(key, "doc_var_converence") == 0) {
	            fscanf(fp, "%lf", &doc_var_converence);
	            continue;
	        }
	        if (strcmp(key, "doc_max_var_iter") == 0) {
	            fscanf(fp, "%d", &doc_max_var_iter);
	            continue;
	        }
	        if (strcmp(key, "em_converence") == 0) {
	            fscanf(fp, "%lf", &em_converence);
	            continue;
	        }
 			if (strcmp(key, "num_topics") == 0) {
	            fscanf(fp, "%d", &num_topics);
	            continue;
	        }
			if (strcmp(key, "window") == 0) {
	            fscanf(fp, "%d", &win);
	            continue;
	        }
	        if (strcmp(key, "G0") == 0) {
	            fscanf(fp, "%d", &G0);
	            continue;
	        }

	        if (strcmp(key, "test") == 0) {
	            fscanf(fp, "%s", test_action);
	        }

	        if (strcmp(test_action, "no")==0) test = 0;
	        else test =1;
	    }
}
void Model::init(Model* init_model) {
    if (init_model) {
    	for(int i=0; i<win; i++) pi[i] = init_model->pi[i];
        for (int k = 0; k < num_topics; k++) {
            for (int i = 0; i < num_words; i++) log_beta[k*num_words + i] = init_model->log_beta[k*num_words + i];
        }
        return;
    }
    for(int i=0; i<win; i++) pi[i] = util::random()*0.5 + 1;
    for(int i=0; i<num_topics; i++) alpha[i] = 0.001;
    for (int k = 0; k < num_topics; k++) {
    	double total = 0;
        for (int i = 0; i < num_words; i++){
        	double temrandom = rand()/(RAND_MAX+1.0);
        	total += temrandom;
        	log_beta[k*num_words + i]= temrandom;
        }
        for (int i = 0; i < num_words; i++){
        	log_beta[k*num_words + i] = log(log_beta[k*num_words + i] / total);
        }
    }
}
void senDocument::init() {
	double total = 0;
	for (int i = 0; i < num_topics; i++) {
		double temrandom = rand()/(RAND_MAX+1.0);
		doctopic[i] =temrandom;
		total += temrandom;
	}
	 for (int i = 0; i < num_topics; i++) {
		 doctopic[i] = log(doctopic[i]/total);
		 rou[i] = util::random();
	 }

	 for(int i=0; i<docwin+num_sentences; i++){
		    total = 0;
	    	for(int j=0; j<num_topics; j++){
	    		double temrandom = rand()/(RAND_MAX+1.0);
	    		docTopicMatrix[i*num_topics + j] = temrandom;
	    		total += temrandom;
	    	}
	    	for(int j=0; j<num_topics; j++){
	    		docTopicMatrix[i*num_topics + j] = log(docTopicMatrix[i*num_topics + j]/ total);
	    	}
	    }
}

void Sentence::init() {
	for (int i = 0; i < win; i++) {
		xi[i] = util::random();
	}
	double total = 0;
	for (int i = 0; i < num_words; i++) {
		 total = 0;
		for (int k = 0; k < num_topics; k++){
			double temrandom = rand()/(RAND_MAX+1.0);
			log_gamma[i * num_topics + k] = temrandom;
			total += temrandom;
		}
		for (int k = 0; k < num_topics; k++){
			log_gamma[i * num_topics + k] = log(log_gamma[i * num_topics + k] / total);
		}
	}
	total = 0;
	for (int i = 0; i < num_topics; i++) {
		double temrandom = rand()/(RAND_MAX+1.0);
		topic[i] = temrandom;
		total += temrandom;
	}
	for (int i = 0; i < num_topics; i++) {
			topic[i] = log(topic[i]/ total);
	}
	for (int i = 0; i < win; i++) {
		total = 0;
		for (int j = 0; j < num_topics; j++) {
			double temrandom = rand()/(RAND_MAX+1.0);
			wintopics[i * num_topics + j] = temrandom;
			total += temrandom;
		}
		for (int j = 0; j < num_topics; j++) {
			wintopics[i * num_topics + j] = log(wintopics[i * num_topics + j] / total);
		}
	}
}

void Model::read_model_info(char* model_root) {
    char filename[1000];
    sprintf(filename, "%s/model.info",model_root);
    printf("%s\n",filename);
    FILE* fp = fopen(filename,"r");
    char str[100];
    int value;
    while (fscanf(fp,"%s%d",str,&value)!=EOF) {
        if (strcmp(str, "num_words:") == 0)num_words = value;
        if (strcmp(str, "num_topics:") == 0)num_topics = value;

		if (strcmp(str, "num_docs:") == 0) num_docs = value;
        if (strcmp(str, "test_num_docs:") == 0)test_num_docs = value;
        if (strcmp(str, "train_num_docs:") == 0)train_num_docs = value;

        if (strcmp(str, "num_all_words_in_test:") == 0)num_all_words_in_test = value;
        if (strcmp(str, "num_all_words_in_train:") == 0)num_all_words_in_train = value;

        if (strcmp(str, "win:") == 0)win = value;
        if (strcmp(str, "G0:") == 0)G0 = value;

    }
    printf("num_words: %d\nnum_topics: %d\n",num_words, num_topics);
    fclose(fp);
}

double* Model::load_mat(char* filename, int row, int col) {
    FILE* fp = fopen(filename,"r");
    double* mat = new double[row * col];
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            fscanf(fp, "%lf", &mat[i*col+j]);
        }
    }
    fclose(fp);
    return mat;
}

void print_model_info(char* model_root, Model* model, senDocument** corpus) {
    char filename[1000];
    sprintf(filename, "%s/model.info",model_root);
    FILE* fp = fopen(filename,"w");
    fprintf(fp, "num_words: %d\n", model->num_words);
    fprintf(fp, "num_topics: %d\n", model->num_topics);

    fprintf(fp, "num_docs: %d\n", model->num_docs);
    fprintf(fp, "test_num_docs: %d\n", model->test_num_docs);
    fprintf(fp, "train_num_docs: %d\n", model->train_num_docs);

    fprintf(fp, "num_all_words_in_test: %d\n", model->num_all_words_in_test);
    fprintf(fp, "num_all_words_in_train: %d\n", model->num_all_words_in_train);
    fprintf(fp, "win: %d\n", model->win);
    fprintf(fp, "G0: %d\n", model->G0);
    fclose(fp);
}
/*
double inference_corpuslikelihood(senDocument** corpus, Model* model) {
    int num_docs = model->num_docs;
    double lik = 0.0;
    for (int d = 0; d < num_docs; d++) {
        double temp_lik = compute_doc_likelihood(corpus[d],model);
        lik += temp_lik;
        corpus[d]->doclik = temp_lik;
    }

    return lik;
}

double inference_doc_likelihood(senDocument* doc, Model* model) {
    double lik = 0.0;
    for(int s=0; s<doc->num_sentences; s++){
    	Sentence* sentence = doc->sentences[s];
    	double temp_lik = inference_sen_likelihood(sentence,model);
    	lik+=temp_lik;
    	doc->sentences[s]->senlik =temp_lik;
    }
    return lik;
}
*/
double inference_sen_likelihood(Sentence* sentence, Model* model){
		double lik = 0.0;

		double* log_topic = sentence->topic;
		double* log_beta = model->log_beta;
		int num_topics = model->num_topics;
		int num_words = model->num_words;
		memset(log_topic, 0, sizeof(double) * num_topics);
		bool* reset_log_topic = new bool[num_topics];
		memset(reset_log_topic, false, sizeof(bool) * num_topics);

		double sigma_xi = 0;
		double* xi = sentence->xi;
		for (int i = 0; i < model->win; i++) {
		        sigma_xi += xi[i];
		}

		for (int i = 0; i < model->win; i++) {
			for (int k = 0; k < num_topics; k++) {
				if (!reset_log_topic[k]) {
		        	log_topic[k] = sentence->wintopics[i * num_topics + k] + log(xi[i]) - log(sigma_xi);
		        	reset_log_topic[k] = true;
		        	}
		        else {
		        	log_topic[k] = util::log_sum(log_topic[k], sentence->wintopics[i * num_topics + k] + log(xi[i]) - log(sigma_xi));
		        }
		    }
		}
		int sen_num_words = sentence->num_words;
		for (int i = 0; i < sen_num_words; i++) {
			double temp = 0;
		    int wordid = sentence->words_ptr[i];
		    temp = log_topic[0] + log_beta[wordid];
		    for (int k = 1; k < num_topics; k++) temp = util::log_sum(temp, log_topic[k] + log_beta[k * num_words + wordid]);
		    lik += temp; //* sentence->words_cnt_ptr[i];
		}
		delete[] reset_log_topic;
		return lik;
}


double corpuslikelihood(senDocument** corpus, Model* model, int num_docs) {
    double lik = 0.0;
    for (int d = 0; d < num_docs; d++) {
        double temp_lik = compute_doc_likelihood(corpus[d],model);
        lik += temp_lik;
    }
    return lik;
}

double compute_doc_likelihood(senDocument* doc, Model* model) {
    double lik = 0.0; 
    for(int s=0; s<doc->num_sentences; s++){
    	Sentence* sentence = doc->sentences[s];
    	double temp_lik = compute_sen_likelihood(doc, sentence, model);
    	lik+=temp_lik;
    }
    return lik;
}

double compute_sen_likelihood(senDocument* doc, Sentence* sentence, Model* model){
		double lik = 0.0;
		double* log_beta = model->log_beta;
		int num_topics = model->num_topics;
		int num_words = model->num_words;
		double* log_topic = new double[num_topics];
		//memcpy(log_topic, doc->doctopic, sizeof(double) * num_topics);
		memcpy(log_topic, sentence->topic, sizeof(double) * num_topics);
		bool* reset_log_topic = new bool[num_topics];
		memset(reset_log_topic, false, sizeof(bool) * num_topics);

		int sen_num_words = sentence->num_words;
		for (int i = 0; i < sen_num_words; i++) {
			double temp = 0;
		    int wordid = sentence->words_ptr[i];

		    temp = log_topic[0] + log_beta[wordid];
		    for (int k = 1; k < num_topics; k++) temp = util::log_sum(temp, log_topic[k] + log_beta[k * num_words + wordid]);
		    lik += temp; //* sentence->words_cnt_ptr[i];
		}
		delete[] log_topic;
		delete[] reset_log_topic;
		return lik;
}

Sentence ** convert_to_lda_corpus(senDocument ** batch_corpus, Model* model, int num_docs){
	int num_topics = model->num_topics;
	int win = model->win;

	Sentence ** lda_corpus = new Sentence * [ num_docs];
	map<int, int> words_ids_cnts;
	for(int d = 0; d<num_docs; d++){
		senDocument* doc = batch_corpus[d];
		//int num_words_in_doc=doc->num_words_in_doc;
		int num_sentences = doc->num_sentences;

		for(int s=0; s<num_sentences; s++){
			Sentence* sen = doc->sentences[s];
			int num_words_in_sentence =sen->num_words;
			for(int w=0; w<num_words_in_sentence; w++){
				int wordid = sen->words_ptr[w];
				int wordcnt = sen->words_cnt_ptr[w];
				map< int, int >::iterator iter;
				iter = words_ids_cnts.find(wordid);
				if(iter == words_ids_cnts.end()){
					words_ids_cnts.insert(map<int,int>::value_type(wordid,wordcnt));
				}else{
					words_ids_cnts[wordid] = wordcnt + words_ids_cnts[wordid];
				}
			}
		}
		int word_num_inSen = words_ids_cnts.size();
		int* words_ptr = new int[word_num_inSen];
		int* words_cnt_ptr = new int [word_num_inSen];

		map<int,int>::iterator it;
		int m =0;
		for(it=words_ids_cnts.begin();it!=words_ids_cnts.end();++it){
			words_ptr[m] = it->first;
			words_cnt_ptr[m] = it->second;
			m++;
		}
		lda_corpus[d] = new Sentence(words_ptr, words_cnt_ptr, word_num_inSen, num_topics, win);
		words_ids_cnts.clear();
	}
	return lda_corpus;
}

double lda_inference(Sentence* doc, Model* model, double* var_gamma, double** phi){
	int num_words_a = model->num_words;
    double converged = 1;
    double phisum = 0;
    double* oldphi = new double[model->num_topics];
    int k, n, var_iter;
    double* digamma_gam = new double[model->num_topics];
    double * log_beta = model->log_beta;

    for (k = 0; k < model->num_topics; k++){
        var_gamma[k] = model->alpha[k] + (doc->num_words/((double) model->num_topics));
        digamma_gam[k] = util::digamma(var_gamma[k]);
        for (n = 0; n < doc->num_words; n++)
            phi[n][k] = 1.0/model->num_topics;
    }
    var_iter = 0;
    int VAR_MAX_ITER = 20;
    while ((converged > 1e-6) && ((var_iter < VAR_MAX_ITER) || (VAR_MAX_ITER == -1))){
    		var_iter++;
    		for (n = 0; n < doc->num_words; n++){
            phisum = 0;
            for (k = 0; k < model->num_topics; k++){
            		int wordid = doc->words_ptr[n];
                oldphi[k] = phi[n][k];
                phi[n][k] = digamma_gam[k] + log_beta[k*num_words_a + wordid];
                if (k > 0) phisum = util::log_sum(phisum, phi[n][k]);
                else phisum = phi[n][k];
            }
            for (k = 0; k < model->num_topics; k++){
                phi[n][k] = exp(phi[n][k] - phisum);
                double p_o = phi[n][k] - oldphi[k];
                if(p_o < 0) p_o = 0;
                var_gamma[k] = var_gamma[k] + doc->words_cnt_ptr[n]*p_o;
                digamma_gam[k] = util::digamma(var_gamma[k]);
            }
    		}
    }

    delete [] digamma_gam;
    delete [] oldphi;
    return 0.0;
}

void initDocTopics2(senDocument** corpus, Model* model){
	int num_docs = model->num_docs;
    int num_words = model->num_words;
    int num_topics = model->num_topics;
	double* sum_phi_w = new double[num_words];
    printf("num_docs: %d\nnum_words: %d\nnum_topics: %d\n", num_docs, num_words, num_topics);
    for (int w = 0; w < num_words; w++) {
        sum_phi_w[w] = 0.0;
        for (int k =0; k < num_topics; k++) sum_phi_w[w] += exp(model->log_beta[k * num_words + w]);
    }

	Sentence ** lda_corpus = convert_to_lda_corpus(corpus, model, num_docs);

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
    	       sentence->senlik += log(sum_pr) * sentence->words_cnt_ptr[w];
    	   }
       }

       Sentence* lda_doc = lda_corpus[d];
       double* lda_topic = lda_doc->topic;
       int lda_num_words = lda_doc->num_words;
       double sum_topic = 0.0;
       for(int k = 0; k < num_topics; k++){
    		lda_topic[k] = 0;
    		for (int w = 0; w < lda_num_words; w++) {
    		         int wordid = lda_doc->words_ptr[w];
    		         lda_topic[k] += exp(model->log_beta[k * num_words + wordid])/sum_phi_w[wordid];
    		}
    		sum_topic += lda_topic[k];
    	}
    	for (int k = 0; k < num_topics; k++) {
    		doc->doctopic[k] = log(lda_topic[k]/sum_topic);
    		doc->rou[k] = lda_topic[k]/sum_topic;
    	}

    	int winsentenceno = doc->docwin + doc->num_sentences;
		for(int s=0; s<winsentenceno; s++)
			for(int k=0; k<num_topics; k++)
				doc->docTopicMatrix[s*num_topics + k] = doc->doctopic[k];

    }
    delete[] sum_phi_w;
}

void initDocTopics(senDocument** corpus, Model* model){
	int num_docs =  model->num_docs;
	int num_topics = model->num_topics;

	printf("begin to convert lda corpus \n");
	Sentence ** lda_corpus = convert_to_lda_corpus(corpus, model, num_docs);

	for(int i =0; i<num_docs; i++){
		if(i % 5000 == 0){
			printf(" lda inference doc: %d \n ", i);
		}
		
		Sentence* doc = lda_corpus[i];
		int num_words_in_doc = doc->num_words;
		double * docgamma = doc->log_gamma;
		double ** docphi = new double*[num_words_in_doc];
		for(int w =0; w<num_words_in_doc; w++) docphi[w] = new double[num_topics];

		lda_inference(doc, model, docgamma, docphi);
		double tem = 0.0;

		for(int j=0; j<num_topics; j++) tem += docgamma[j];
		//normalized the ditribution before log it.
		for(int j=0; j<num_topics; j++){
					corpus[i]->doctopic[j] = log(docgamma[j] / tem);
					corpus[i]->rou[j] = docgamma[j] / tem;
				}
		int winsentenceno = corpus[i]->docwin+corpus[i]->num_sentences;
		for(int s=0; s<winsentenceno; s++)
			for(int j=0; j<num_topics; j++)
				corpus[i]->docTopicMatrix[s*num_topics + j] = corpus[i]->doctopic[j];

		delete [] docgamma;
		for(int w =0; w<num_words_in_doc; w++) delete[] docphi[w];
		delete [] docphi;
	}

	printf("end to inference by lda  \n");
	//init all the sentence's topic distribution with the document topic distribution
	for(int d=0; d<num_docs; d++){
		int senno = corpus[d]->num_sentences;
		double* topic = corpus[d]->doctopic;
		for(int s= 0; s<senno; s++){
			for(int k = 0; k<num_topics; k++){
				corpus[d]->sentences[s]->topic[k] = topic[k];
			}
			for(int w =0; w< model->win; w++){
				for(int k = 0; k<num_topics; k++){
					//init all the fore topic distributions for each sentence. by document topics
					corpus[d]->sentences[s]->wintopics[w*num_topics +k] = topic[k];
				}
			}
		}
	}
	

}

void split_corpus(senDocument ** corpus, senDocument ** test_corpus, senDocument ** train_corpus,
 				int num_docs, int test_num_docs, int train_num_docs, 
 				int& num_all_words_in_test, int& num_all_words_in_train){

	srand( (unsigned)time(NULL) ); 
	vector<int> poker_index;
	for (int num = 0; num < num_docs; ++num) poker_index.push_back(num);
	//random_shuffle(poker_index.begin(), poker_index.end());

	set<int> test_index_set, train_index_set;
	for (int i = 0; i < train_num_docs; i++) train_index_set.insert(poker_index[i]);
	for (int i = train_num_docs; i < train_num_docs+test_num_docs; i++) test_index_set.insert(poker_index[i]);
	
	set<int>::iterator test_it;
	int num_test = 0;
	for(test_it=test_index_set.begin();test_it!=test_index_set.end();test_it++){
		int id = *test_it;
		test_corpus[num_test] = corpus[id];
		senDocument * doc = corpus[id];
		num_all_words_in_test += doc->num_total_words;

		num_test++;
	}
	
	set<int>::iterator train_it;
	int num_train = 0;
	for(train_it=train_index_set.begin();train_it!=train_index_set.end();train_it++){
		int id = *train_it;
		train_corpus[num_train] = corpus[id];
		num_all_words_in_train += train_corpus[num_train]->num_total_words;
		num_train++;
	}

}

void begin_ratm(char* settingfile, char* inputfile, char* model_root, char* beta_file = NULL) {
	setbuf(stdout, NULL);
	Configuration config = Configuration(settingfile);
	int win = config.win;
	int num_topics = config.num_topics;

    int num_docs;
    int num_words;
    int num_all_words;
    
   	int G0 = config.G0;
   	if(G0 == 1) {
   		puts("Use the topic distribution of doc as the G0.");
   	 	win = win+1;
   	} else {
   		puts("Ignore the G0.");
   		win = win;
   	}
   	printf("The window size is %d, and the number of topics is %d. \n", win, num_topics);
    srand(unsigned(time(0)));
    senDocument** corpus = readData(inputfile,num_topics,num_words,num_docs, num_all_words, win);
	
	
	int num_all_words_in_test = 0;
    int num_all_words_in_train = 0;
    int test_num_docs =0;

	if(config.test == 1){
		test_num_docs = int(floor(num_docs*0.2));
	}
    
    int train_num_docs = num_docs-test_num_docs;

    senDocument ** test_corpus = new senDocument * [test_num_docs];
    senDocument ** train_corpus = new senDocument * [train_num_docs];
    split_corpus(corpus, test_corpus, train_corpus, 
    	num_docs, test_num_docs, train_num_docs, num_all_words_in_test, num_all_words_in_train);

    if(config.test == 1){
    	printf("The test set contains %d docs, and total %d words. \n", test_num_docs, num_all_words_in_test);
		printf("The train set contains %d docs, and total %d words. \n", train_num_docs, num_all_words_in_train);
	} else printf("The train set contains %d docs, and total %d words. \n", train_num_docs, num_all_words_in_train);

    //model for train set
    Model* model = new Model(num_docs, test_num_docs, train_num_docs,
		num_all_words_in_test, num_all_words_in_train, num_words, num_topics, win, G0);
    

    //save the model
    print_model_info(model_root, model, corpus);
    
    time_t learn_begin_time = time(0);
    int num_round = 0;
    
    // init the beta and topics
    if(beta_file){
    	 readinitParameters(model, beta_file);
    }
    printf("Initialize the topics by lda... \n\n");
    initDocTopics(corpus, model);

    double lik_train = 0.0;
    double lik_test = 0.0;
    double plik;
    double converged = 1;
    puts("Now begin to train: ");
    do {
        time_t cur_round_begin_time = time(0);
        plik = lik_train;
        printf("Round %d -> ", num_round);
        //printf("run inference...");
        runThreadInference(train_corpus, model, &config, train_num_docs);
        //note that, we update the pi and beta only USING train_corpus.
        learnPi(train_corpus, model, &config, train_num_docs);
        learnBeta(train_corpus, model, train_num_docs);

		//printf("compute loglikelihood on train set.\n");
        lik_train = corpuslikelihood(train_corpus, model, train_num_docs);
        double perplexity_train = exp(-lik_train/num_all_words_in_train);

        converged = (plik - lik_train) / plik;
        if (converged < 0) config.doc_max_var_iter *= 2;

        unsigned int cur_round_cost_time = time(0) - cur_round_begin_time;

		printf("loglikelihood =%lf, perplexity=%lf, converged=%lf, time=%u secs.\n", 
			lik_train, perplexity_train, converged, cur_round_cost_time);
		num_round += 1;

        if (num_round % 20 == 0){
        	printParameters(corpus,num_round, model_root, model);
        	// run test set every 5 rounds.
        	if(config.test == 1){
	        	printf("Verify on Test Set -> ");
	        	double lik_test = verifyTestSet(test_corpus, model, &config, test_num_docs);
				double perplexity_test = exp(-lik_test/num_all_words_in_test);
				printf("loglikelihood on test=%lf, test perplexity=%lf.\n", lik_test, perplexity_test);
			}
        }
    }
    while (num_round < config.max_em_iter && (converged < 0 || converged > config.em_converence || num_round < 10));
    unsigned int learn_cost_time = time(0) - learn_begin_time;
 	printf("all learn runs %d rounds and cost %u secs.\n", num_round, learn_cost_time);
    printParameters(corpus,-1,model_root, model);

	if(config.test == 1){
	        	printf("Verify on Test Set -> ");
	        	double lik_test = verifyTestSet(test_corpus, model, &config, test_num_docs);
				double perplexity_test = exp(-lik_test/num_all_words_in_test);
				printf("loglikelihood on test=%lf, test perplexity=%lf.\n", lik_test, perplexity_test);
	}

    delete model;
    for (int i = 0; i < num_docs; i++) delete corpus[i];
    delete[] corpus;
	
}


void infer_ratm(char* settingfile, char* test_file, char* model_root, char* prefix, char* out_dir=NULL) {
    setbuf(stdout,NULL);
    Configuration configuration = Configuration(settingfile);

    Model* model = new Model(model_root, prefix);
    int num_topics = model->num_topics;
    int win = model->win;
   
    int num_test_words;
    int num_test_docs;
    int num_test_all_words;
    srand(unsigned(time(0)));
    //readData(char* filename, int num_topics,int& num_words, int& num_docs, int& num_all_words, int& win){
    senDocument** corpus = readData(test_file,num_topics,num_test_words,num_test_docs, num_test_all_words, win);
    model->num_all_words_in_test = num_test_all_words;
    model->test_num_docs = num_test_docs;
    model->train_num_docs = 0;
    
	initDocTopics(corpus, model);
    
    runThreadInference(corpus, model, &configuration, num_test_docs);

    double lik = corpuslikelihood(corpus, model, num_test_docs);
    double perplex = exp(-lik/num_test_all_words);
    printf("likehood: %lf perplexity:%lf num all words: %d\n", lik, perplex, num_test_all_words);

    if (out_dir) {
    	saveDocumentsTopicsSentencesAttentions(corpus, model, out_dir);
    }

    for (int i = 0; i < num_test_docs; i++) {
        delete corpus[i];
    }
    delete[] corpus;
}

///////////////////////////////////

void printParameters(senDocument** corpus, int num_round, char* model_root, Model* model) {
    char pi_file[1000];
    char beta_file[1000];
    char topic_dis_file[1000];
    char liks_file[1000];
    if (num_round != -1) {
        sprintf(pi_file, "%s/%03d.pi", model_root, num_round);
        sprintf(beta_file, "%s/%03d.topicdis_overwords_beta", model_root, num_round);
        sprintf(topic_dis_file, "%s/%03d.doc_dis_overtopic", model_root, num_round);
        sprintf(liks_file, "%s/%03d.likehoods", model_root, num_round);
    }
    else {
        sprintf(pi_file, "%s/final.pi", model_root);
        sprintf(beta_file, "%s/final.topicdis_overwords_beta", model_root);
        sprintf(topic_dis_file,"%s/final.doc_dis_overtopics", model_root);
        sprintf(liks_file, "%s/final.likehoods", model_root);
    }
    print_mat(model->log_beta, model->num_topics, model->num_words, beta_file);
    print_mat(model->pi, model->win, 1, pi_file);

    //save the topic distribution of all the training documents.
    int num_docs = model->num_docs;
    FILE* topic_dis_fp = fopen(topic_dis_file,"w");
    FILE* liks_fp = fopen(liks_file, "w");
    for (int d = 0; d < num_docs; d++) {
        fprintf(liks_fp, "%lf\n", corpus[d]->doclik);
        senDocument* doc = corpus[d];
        fprintf(topic_dis_fp, "%lf", doc->doctopic[0]);
        for (int k = 1; k < doc->num_topics; k++)fprintf(topic_dis_fp, " %lf", doc->doctopic[k]);
        fprintf(topic_dis_fp, "\n");
    }
    fclose(topic_dis_fp);
    fclose(liks_fp);
}

void saveDocumentsTopicsSentencesAttentions(senDocument** corpus, Model * model, char* output_dir) {
    char filename[1000];
    sprintf(filename, "%s/test_doc_dis_overtopics.txt", output_dir);
    char liks_file[1000];
    sprintf(liks_file, "%s/test_likehoods.txt", output_dir);
    char attentionfilename[1000];
    sprintf(attentionfilename, "%s/test_attentions.txt", output_dir);
    FILE* liks_fp = fopen(liks_file, "w");
    FILE* topic_dis_fp = fopen(filename,"w");
    FILE* attentions_fp = fopen(attentionfilename,"w");
    int win = model->win;
    for (int d = 0; d < model->test_num_docs; d++) {
           fprintf(liks_fp, "%lf\n", corpus[d]->doclik);
           senDocument* doc = corpus[d];
           fprintf(topic_dis_fp, "%lf", doc->doctopic[0]);
           for (int k = 1; k < doc->num_topics; k++)fprintf(topic_dis_fp, " %lf", doc->doctopic[k]);
           fprintf(topic_dis_fp, "\n");
           int sentensNum = doc->num_sentences;
           for (int s=0; s< sentensNum; s++){
        	   Sentence * sentence = doc->sentences[s];
        	   for(int w=0; w<win; w++){
        		   fprintf(attentions_fp, "%lf ", sentence->xi[w]);
        	   }
        	   fprintf(attentions_fp, "##");
           }
           fprintf(attentions_fp, "\n");

       }
    fclose(topic_dis_fp);
    fclose(liks_fp);
}

////////////////////////////////

void print_mat(double* mat, int row, int col, char* filename) {
    FILE* fp = fopen(filename,"w");
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            fprintf(fp,"%lf ",mat[i*col + j]);
        }
        fprintf(fp,"\n");
    }
    fclose(fp);
}

int main(int argc, char* argv[]) {
	//begin_rbm(argv[1],argv[2],atoi(argv[3]),atoi(argv[4]),argv[5], argv[6],argv[7]);
/*	if(argc <= 1 || !(strcmp(argv[1],"est") == 0 && (argc == 6  || argc == 8))  || !(strcmp(argv[1],"inf") == 0 && argc == 6)){

	}*/
	if (argc > 1 && argc == 5 && strcmp(argv[1],"est") == 0) {
		printf("Now begin training...\n");
		begin_ratm(argv[2],argv[3], argv[4], NULL);
	}else if(argc > 1 && argc == 6 && strcmp(argv[1],"est") == 0){
		printf("Now begin training with initial parameters...\n");
		begin_ratm(argv[2],argv[3], argv[4], argv[5]);
	}else if(argc > 1 && argc == 7 && strcmp(argv[1],"inf") == 0){
		printf("Now begin inference...\n");
		infer_ratm(argv[2], argv[3], argv[4],argv[5], argv[6]);
	}
	else {
		printf("Please use the following setting.\n");
		printf("\n");
		printf("*************Trainning***********************\n");

		printf("Pleae set G0 = [0 1] in setting.txt \n 0: ignore the G0 \n 1: use the topic distribution of doc as the G0\n");
		printf("\n");
		printf(
				"./ratm est <setting.txt>  <input data file><model save dir>\n\n");
		printf(
				"If you want to initialize the model with default parameters, please use: \n");
		printf(
				"./ratm est <setting.txt> <input data file> <model save dir> <topic_dis_overwords_beta file >\n\n");
		printf("**************Inference**********************\n");
		printf(
				"./ratm inf <setting.txt> <input test data file> <model dir> <perfix> <output dir>\n");
		printf("\n");
	}
	return 0;
}