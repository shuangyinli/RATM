/* */

#ifndef RATM_H_
#define RATM_H_

#include "stdio.h"
#include "utils.h"

struct Sentence{
	double* xi;
	double* log_gamma;
	int* words_ptr;
	int* words_cnt_ptr;
	int num_words;
	int num_topics;
	double* topic;
	double senlik;
	int win;
	double* wintopics;
	Sentence(int* words_ptr_, int* words_cnt_ptr_, int num_words_, int num_topics_, int win_){
		words_ptr = words_ptr_;
		words_cnt_ptr = words_cnt_ptr_;
		num_words = num_words_;
		num_topics = num_topics_;
		win = win_;
		xi = new double[win];
		log_gamma= new double[num_words * num_topics];
		topic = new double[num_topics];
		senlik = 100;
		wintopics = new double[win * num_topics];
		init();
	}
	void init();
	~Sentence(){
		if (xi)delete[] xi;
		if (log_gamma) delete[] log_gamma;
		if (words_ptr) delete[] words_ptr;
		if (words_cnt_ptr) delete[] words_cnt_ptr;
		if (topic) delete[] topic;
		if (wintopics) delete[] wintopics;
	}
};

struct senDocument{
	double * doctopic;
	double * docTopicMatrix;
	int num_total_words;
	int num_topics;
	double doclik;
	int docwin;
	int num_sentences;
	struct Sentence ** sentences;
	double* rou;
	senDocument(int num_total_words_, int num_topics_, int docwin_, int num_sentences_){
		num_topics = num_topics_;
		num_total_words = num_total_words_;
		doclik = 100;
		docwin = docwin_;
		num_sentences = num_sentences_;
		rou = new double[num_topics];
		doctopic = new double[num_topics];
		docTopicMatrix = new double[(docwin+num_sentences)*num_topics];
		sentences = new Sentence* [num_sentences];
		init();
	}
	senDocument(int num_total_words_, int num_topics_, int docwin_, int num_sentences_, struct Sentence** sentence_){
			num_topics = num_topics_;
			num_total_words = num_total_words_;
			doclik = 100;
			docwin = docwin_;
			num_sentences = num_sentences_;
			doctopic = new double[num_topics];
			rou = new double[num_topics];
			docTopicMatrix = new double[(docwin+num_sentences)*num_topics];
			sentences = sentence_;
			init();
		}
	~senDocument(){
        if(doctopic) delete [] doctopic;
        if(sentences) delete [] sentences;
        if (rou) delete[] rou;
	}
	void init();
};

struct Model {
	int num_docs;
	int num_words;
	int num_topics;
	int win;
	int num_all_words;
	double* pi;
	double* log_beta;
	char * G0;
	double* alpha;
	Model(int num_docs_, int num_words_, int num_topics_, int win_, int num_all_words_, char* G0_ = NULL, Model* init_model = NULL){
		num_docs = num_docs_;
		num_words = num_words_;
		win = win_;
		num_all_words = num_all_words_;
		num_topics = num_topics_;
		G0 = G0_;
		pi = new double[win];
		alpha = new double[num_topics];
		log_beta = new double[num_topics * num_words];
		init(init_model);
	}
	void init(Model* init_model=NULL);
	Model(char* model_root, char* prefix) {
	        read_model_info(model_root);
	        char pi_file[1000];
	        sprintf(pi_file, "%s/%s.pi", model_root, prefix);
	        char beta_file[1000];
	        sprintf(beta_file,"%s/%s.topicdis_overwords_beta",model_root,prefix);
	        //char alpha_file[1000];
	        //sprintf(alpha_file,"%s/%s.alpha",model_root,prefix);
	        pi = load_mat(pi_file, win, 1);
	        alpha = new double[num_topics];
	        for(int i=0; i<num_topics; i++) alpha[i] = 0.001;
	        log_beta = load_mat(beta_file, num_topics, num_words);
	    }
	    ~Model() {
	        if (pi)delete[] pi;
	        if (alpha)delete[] alpha;
	        if (log_beta) delete[] log_beta;
	    }
	    void read_model_info(char* model_root);
	    double* load_mat(char* filename,int row,int col);
};

struct Configuration {
    double pi_learn_rate;
    int max_pi_iter;
    double pi_min_eps;
    double xi_learn_rate;
    int max_xi_iter;
    double xi_min_eps;
    int max_em_iter;
    static bool print_debuginfo;
    int num_threads;
    int sen_max_var_iter;
    int doc_max_var_iter;
    double sen_var_converence;
    double doc_var_converence;
    double em_converence;
    Configuration(char* settingfile) {
        pi_learn_rate = 0.00001;
        max_pi_iter = 100;
        pi_min_eps = 1e-5;
        max_xi_iter = 100;
        xi_learn_rate = 10;
        xi_min_eps = 1e-5;
        max_em_iter = 30;
        num_threads = 1;
        sen_var_converence = 1e-6;
        sen_max_var_iter = 30;
        doc_var_converence = 0.001;
        doc_max_var_iter = 3;
        em_converence = 1e-4;
        if(settingfile) read_settingfile(settingfile);
    }
    void read_settingfile(char* settingfile);
};

double compute_sen_likelihood(Sentence* sentence, Model* model);
double compute_doc_likelihood(senDocument* doc, Model* model);
double corpuslikelihood(senDocument** corpus, Model* model);
double compute_sen_likelihood2(senDocument* doc, Sentence* sentence, Model* model);
double compute_doc_likelihood2(senDocument* doc, Model* model);
double corpuslikelihood2(senDocument** corpus, Model* model);
double quick_compute_sen_likelihood(senDocument* doc, Sentence* sentence, Model* model);
double quick_compute_doc_likelihood(senDocument* doc, Model* model);
double quick_corpuslikelihood(senDocument** corpus, Model* model);
void print_mat(double* mat, int row, int col, char* filename);
void printParameters(senDocument** corpus, int num_round, char* model_root, Model* model);
void print_lik(double* likehood_record, int num_round, char* model_root);
void readinitParameters(senDocument** corpus, Model* model, char* beta_file);
void saveDocumentsTopicsSentencesAttentions(senDocument** corpus, Model* model, char* output_dir);

#endif
