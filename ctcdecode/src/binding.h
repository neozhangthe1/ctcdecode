int stream_decoder_init(string key,
                        const char* labels,
                        int vocab_size,
                        size_t beam_size,
                        size_t num_processes,
                        double cutoff_prob,
                        size_t cutoff_top_n,
                        size_t blank_id,
                        bool log_input,
                        void *scorer);


int stream_decoder_feed(string key,
                        THFloatTensor *th_probs,
                        int th_seq_len,
                        THIntTensor *th_output,
                        THIntTensor *th_timesteps,
                        THFloatTensor *th_scores,
                        THIntTensor *th_out_length)


int stream_decoder_finalize(string key);


int paddle_beam_decode(THFloatTensor *th_probs,
                       THIntTensor *th_seq_lens,
                       const char* labels,
                       int vocab_size,
                       size_t beam_size,
                       size_t num_processes,
                       double cutoff_prob,
                       size_t cutoff_top_n,
                       size_t blank_id,
                       int log_input,
                       THIntTensor *th_output,
                       THIntTensor *th_timesteps,
                       THFloatTensor *th_scores,
                       THIntTensor *th_out_length);

int paddle_beam_decode_lm(THFloatTensor *th_probs,
                          THIntTensor *th_seq_lens,
                          const char* labels,
                          int vocab_size,
                          size_t beam_size,
                          size_t num_processes,
                          double cutoff_prob,
                          size_t cutoff_top_n,
                          size_t blank_id,
                          bool log_input,
                          int *scorer,
                          THIntTensor *th_output,
                          THIntTensor *th_timesteps,
                          THFloatTensor *th_scores,
                          THIntTensor *th_out_length);

void* paddle_get_scorer(double alpha,
                        double beta,
                        const char* lm_path,
                        const char* labels,
                        int vocab_size);

void paddle_release_scorer(void* scorer);

int is_character_based(void *scorer);
size_t get_max_order(void *scorer);
size_t get_dict_size(void *scorer);
void reset_params(void *scorer, double alpha, double beta);
