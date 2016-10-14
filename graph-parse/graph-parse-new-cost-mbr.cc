#include "cnn/nodes.h"

#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

#include "Eisner.cc"


#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <deque>
#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <vector>
#include <limits>
#include <cmath>
#include <chrono>
#include <ctime>

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>

#include <unordered_map>
#include <unordered_set>

#include <Eigen/Dense>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

using namespace std;
using namespace cnn;

volatile bool requested_stop = false;

unsigned INPUT_DIM = 0;
unsigned PRET_DIM = 0;
unsigned LSTM_INPUT_DIM = 0;
unsigned HIDDEN_DIM = 0;
unsigned LAYERS = 0;
unsigned POS_DIM = 0;
unsigned DIR_DIM = 0;
unsigned MLP_DIM = 0;

unsigned VOCAB_SIZE = 0;
unsigned POS_SIZE = 0;
unsigned LABEL_SIZE = 0;
unsigned N_WORDS = 0;

unsigned NUM_ENSEMBLE = 0;

unsigned LEFT_DIR = 0;
unsigned RIGHT_DIR = 1;

float DROPOUT = 0.0f;
bool USE_POS = true;
bool USE_ENSEMBLE = false;

float BETA = 0.0f;
float GAMMA = 0.0f;
float ETA_DECAY = 0.0f;

cnn::Dict words_dict, pret_dict, pos_dict, labels_dict;
unordered_map<unsigned, vector<float>> pretrained;
set<unsigned> singletons;

//given a particular row and column in a matrix, returns its index in a linear vector
inline int rc2ind(int r, int c, int n) {
  return c * n + r;
}

namespace po = boost::program_options;
void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("train,t", po::value<string>(), "Training data")
        ("dev,d", po::value<string>(), "Dev data")
        ("test,p", po::value<string>(), "Test data")
        ("model,m", po::value<string>(), "Trained model file")
        ("hidden_dim", po::value<unsigned>()->default_value(100), "Dimension of the LSTM hidden layers")
        ("lstm_input_dim", po::value<unsigned>()->default_value(100), "Dimension of the LSTM input")
        ("mlp_dim", po::value<unsigned>()->default_value(100), "Dimension of the MLP")
        ("input_dim", po::value<unsigned>()->default_value(32), "Dimension of the learnable words embedding")
        ("pos_dim", po::value<unsigned>()->default_value(16), "Dimension of the learnable pos tag embedding")
        ("words,w", po::value<string>(), "Pretrained word embedding")
        ("beta", po::value<float>()->default_value(0.0), "Regulariser coefficient beta")
        ("gamma", po::value<float>()->default_value(0.0), "Regulariser coefficient gamma")
        ("num_ensemble", po::value<float>()->default_value(18.0), "Number of ensemble")
        ("cost_matrix", po::value<string>(), "File containing the cost from the ensemble")
        ("eta_decay", po::value<float>()->default_value(0.00), "Eta decay rate for the Adams")
        ("learn,x", "Should training be run?")
        ("dropout,D", po::value<float>(), "Dropout rate")
        ("use_pos_tags,P", "make POS tags visible to parser")
        ("pretrained_dim", po::value<unsigned>(), "Dimension of the pre-trained word embedding")
        ("layers", po::value<unsigned>()->default_value(2), "Number of the LSTM layers (default is 2-layered LSTM)")
        ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);

  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
  if (conf->count("train") == 0) {
    cerr << "Need to specify the training corpus, even during test time, for vocabulary mapping purposes" << endl;
  }

}

set<unsigned> getSingletons(const vector<vector<int>>& sentenceWords) {
  set<unsigned> singletons; 
  unordered_map<unsigned, unsigned> wordFreq; 
  assert (words_dict.size() > 0 && sentenceWords.size() > 0);
  cerr << "Mapping the singleton words from " << sentenceWords.size() << " training sentences" << endl;
  for (const auto& sent : sentenceWords) {
    for (int idx = 0; idx < sent.size(); ++idx) {
      if (!wordFreq.count(sent[idx])) wordFreq[sent[idx]] = 1;
      else wordFreq.at(sent[idx]) = wordFreq.at(sent[idx]) + 1; 
    }
  } 
  assert (wordFreq.size() == (words_dict.size() - 1));
  int ctr = 0;
  for (const auto& it : wordFreq) {
    if (it.second == 1) {
     ++ctr;
     singletons.insert(it.first);
     if (ctr < 10) cerr << "Sample singleton: " << words_dict.Convert(it.first) << endl;
    }
  } 
  if (singletons.size() == 0) {
     cerr << "Possible error! Are there really no singletons on the training set?" << endl;
     assert (singletons.size() > 0);
   }
  assert (singletons.size() == ctr);
  cerr << "Number of singletons: " << singletons.size() << endl;
  return singletons;
}


template <class Builder>
struct ParserBuilder {
  Builder l2rbuilder;
  Builder r2lbuilder;
  LookupParameters* p_w; //trainable word embedding
  LookupParameters* p_pt; //pos tag embedding
  LookupParameters* p_t; //pre-trained word embedding
  Parameters* p_w_input; //connection between learnable word embedding and lstm input
  Parameters* p_pt_input; //connection between learnable pos tag embedding and lstm input
  Parameters* p_t_input; //connection between fixed pre-trained word embedding and lstm input
  Parameters* p_input_bias; //bias for the lstm input
  Parameters* p_root_score; //compute how suitable a particular token is to become the head of the sentence
  Parameters* p_root_score_bias; //scalar bias to assess the root suitability of a particular token
  Parameters* p_root_MLP; //compute how suitable a particular token is to become the head of the sentence
  Parameters* p_root_MLP_bias; //compute how suitable a particular token is to become the head of the sentence
  Parameters* p_MLP_features;
  Parameters* p_MLP_bias;
  Parameters* p_MLP_output;
  Parameters* p_MLP_labels;
  Parameters* p_MLP_labels_bias;
  Parameters* p_MLP_labels_softmax;
  Parameters* p_MLP_labels_softmax_bias;
  Parameters* p_score_bias; //scalar bias to compute the score
  explicit ParserBuilder(Model& model, const unordered_map<unsigned, vector<float>>& pretrained):
    l2rbuilder(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, &model),
    r2lbuilder(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, &model) {
      p_w = model.add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM});
      p_MLP_features = model.add_parameters({MLP_DIM, 4 * HIDDEN_DIM});
      p_MLP_bias = model.add_parameters({MLP_DIM});
      p_MLP_output = model.add_parameters({MLP_DIM});
      p_score_bias = model.add_parameters({1});
      p_w_input = model.add_parameters({LSTM_INPUT_DIM, INPUT_DIM});
      p_input_bias = model.add_parameters({LSTM_INPUT_DIM});
      p_root_score = model.add_parameters({MLP_DIM});
      p_root_score_bias = model.add_parameters({1});
      p_root_MLP = model.add_parameters({MLP_DIM, 2 * HIDDEN_DIM});
      p_root_MLP_bias = model.add_parameters({MLP_DIM});
      p_MLP_labels = model.add_parameters({MLP_DIM, 4 * HIDDEN_DIM});
      p_MLP_labels_bias = model.add_parameters({MLP_DIM});
      p_MLP_labels_softmax = model.add_parameters({LABEL_SIZE, MLP_DIM});
      p_MLP_labels_softmax_bias = model.add_parameters({LABEL_SIZE});
      if (USE_POS) {
        p_pt = model.add_lookup_parameters(POS_SIZE, {POS_DIM});
        p_pt_input = model.add_parameters({LSTM_INPUT_DIM, POS_DIM});
      } else {
        p_pt = nullptr;
        p_pt_input = nullptr;
      }
      if (pretrained.size() > 0) {
        p_t = model.add_lookup_parameters(pretrained.size(), {PRET_DIM});
        p_t_input = model.add_parameters({LSTM_INPUT_DIM, PRET_DIM});
        // initialise the pre-trained word embedding
        for (auto it : pretrained)
          p_t->Initialize(it.first, it.second);
      } else {
        p_t = nullptr;
        p_t_input = nullptr;
      }
  }

  vector<Expression> EmbedBiLSTM(ComputationGraph& cg, const vector<int>& toks, const vector<int>& pret_idx, const vector<int>& tags, bool apply_dropout) {
    assert (toks.size() == tags.size() && tags.size() == pret_idx.size());
    const unsigned slen = toks.size();
    l2rbuilder.new_graph(cg);  // reset RNN builder for new graph
    l2rbuilder.start_new_sequence();
    r2lbuilder.new_graph(cg);  // reset RNN builder for new graph
    r2lbuilder.start_new_sequence();
    if (apply_dropout) {
      l2rbuilder.set_dropout(DROPOUT);
      r2lbuilder.set_dropout(DROPOUT);
    } else {
      l2rbuilder.disable_dropout();
      r2lbuilder.disable_dropout();
    }
    Expression input_bias = parameter(cg, p_input_bias);
    Expression w_input = parameter(cg, p_w_input);
    Expression pt_input;
    if (USE_POS) pt_input = parameter(cg, p_pt_input);
    Expression t_input;
    if (p_t_input) t_input = parameter(cg, p_t_input);
    // first compose the tokens and tags to form the LSTM input
    vector<Expression> composedInput(toks.size());
    assert (singletons.size() > 0);
    for (unsigned t = 0; t < toks.size(); ++t) {
      Expression i_i;
      Expression w;
      if (apply_dropout && singletons.count(toks[t])) {
        // with a 0.5 probability, a singleton word at training will be treated as the UNK symbol. 
        // using this approach we can train a good UNK vector while still training the embedding of rare words.
        int flip = rand() % 2 + 1;
        assert (flip == 1 || flip == 2);
        if (flip == 1)
          w = lookup(cg, p_w, words_dict.Convert("kUNK")); 
        else
          w = lookup(cg, p_w, toks[t]);
      } else {
       w = lookup(cg, p_w, toks[t]);
      }
      if (USE_POS) {
        Expression pt = lookup(cg, p_pt, tags[t]);
        i_i = affine_transform({input_bias, w_input, w, pt_input, pt});
      } else {
        i_i = affine_transform({input_bias, w_input, w});
      }
      if (p_t && pretrained.count(pret_idx[t])) {
        unsigned idx = pret_idx[t]; 
        Expression pret = const_lookup(cg, p_t, idx); 
        i_i = affine_transform({i_i, t_input, pret});
      } 

      //Lastly, apply some non-linearity
      composedInput[t] = rectify(i_i);
    }
    assert (composedInput.size() == slen); 
    // read sequence from left to right
    vector<Expression> left_Output;
    for (unsigned t = 0; t < slen; ++t) {
      l2rbuilder.add_input(composedInput[t]);
      left_Output.push_back(l2rbuilder.back()); // dropout not applied
    }
    // read sequence from right to left
    deque<Expression> right_Output;
    for (unsigned t = 0; t < slen; ++t) {
      r2lbuilder.add_input(composedInput[slen - t - 1]);
      right_Output.push_front(r2lbuilder.back());
    }
    assert (right_Output.size() == left_Output.size());
    assert (right_Output.size() == slen);
    vector<Expression> result;
    // concatenate the embedding from the left and right LSTMs
    for (unsigned i = 0; i < left_Output.size(); ++i)
      result.push_back(concatenate({left_Output[i], right_Output[i]}));
    assert (result.size() == right_Output.size());
    return result;
  }
  
  pair<Eigen::MatrixXf, Eigen::MatrixXf> BuildGraph(const vector<int>& toks, const vector<int>& pret_idx, const vector<int>& tags, ComputationGraph& cg, const vector<int>& heads, const vector<int>& labels, bool apply_dropout, vector<Expression>& embedded, const vector<float>& currCost) {
    assert (toks.size() == pret_idx.size() && toks.size() == tags.size() && toks.size() == heads.size() && toks.size() == labels.size());
    if (currCost.size() > 0) {
      assert (apply_dropout); //cost-sensitive training only applies if we're in training mode
    }
    //Embed the sentence with bidirectional LSTM
    const unsigned slen = toks.size();
    embedded = EmbedBiLSTM(cg, toks, pret_idx, tags, apply_dropout);
    assert (embedded.size() == slen);
    vector<Expression> result(slen * slen);

    Expression MLP_features = parameter(cg, p_MLP_features);
    Expression MLP_bias = parameter(cg, p_MLP_bias);
    Expression score_bias = parameter(cg, p_score_bias);
    Expression MLP_output = parameter(cg, p_MLP_output);
    Expression zero = input(cg, 0.0f);

    // compute the (local) score of an arc based on the bidirectional LSTM encoding
    vector<Expression> sum_MLP_squared_norm;
    sum_MLP_squared_norm.push_back(zero);
    for (unsigned h = 0; h < toks.size(); ++h) {
        for (unsigned m = 0; m < toks.size(); ++m) {
            Expression MLP_hidden = tanh(affine_transform({MLP_bias, MLP_features, concatenate({embedded[h], embedded[m]})})); 
            // add the regulariser cost
            if (h != m) //only add the regulariser cost if the head is not equal to the modifier
            sum_MLP_squared_norm.push_back(squared_norm(MLP_hidden));
            result[rc2ind(h, m, slen)] = dot_product(MLP_output, MLP_hidden) + score_bias;  //no non-linearity is applied to the final scoring function
        }
    }
    assert (sum_MLP_squared_norm.size() == (toks.size() * toks.size() - toks.size() + 1));
    assert (result.size() == (slen * slen));

    // reshape the score between all possible arcs into an n by n matrix, where n = number of tokens
    Expression result_expr = reshape(concatenate(result), {slen, slen});
    auto& result_matrix = *(cg.incremental_forward()); // n by n matrix representing the scores, excluding root selection scores

    // compute the scores of each token being the root of the sentence (using a separate 1-hidden layer neural net)
    Expression root_score = parameter(cg, p_root_score);
    Expression root_score_bias = parameter(cg, p_root_score_bias);
    Expression root_MLP = parameter(cg, p_root_MLP);
    Expression root_MLP_bias = parameter(cg, p_root_MLP_bias);
    vector<Expression> root_scores(slen);
    for (unsigned m = 0; m < slen; ++m) { 
      Expression hidden_root_score = tanh(affine_transform({root_MLP_bias, root_MLP, embedded[m]})); 
      root_scores[m] = dot_product(hidden_root_score, root_score) + root_score_bias;
    }
    assert (root_scores.size() == slen);  
    Expression root_scores_expr = reshape(concatenate(root_scores), {1, slen});
    auto& root_scores_vector = *(cg.incremental_forward());

    //if during training time, get the distillation cost matrix, obtain the CLE, and back-propagate
    if (apply_dropout) {
      vector<float> cost((slen + 1) * (slen + 1), 1.0);
      int count_root = 0;
      if (USE_ENSEMBLE) { // if using the distillation cost function
        assert (currCost.size() > 0 && currCost.size() == ((slen + 1) * (slen + 1))); 
        for (unsigned head = 0; head < (slen + 1); ++head) {
          for (unsigned mod = 0; mod < (slen + 1); ++mod) {
            if (mod == 0) assert(currCost[head * (slen + 1) + mod] == 0.0);
            if (head == mod) assert(currCost[head * (slen + 1) + mod] == 0.0);
            float curr_cost = currCost[head * (slen + 1) + mod];
            assert (curr_cost >= 0.0 && curr_cost <= NUM_ENSEMBLE);
            cost[rc2ind(head, mod, slen + 1)] -= (currCost[head * (slen + 1) + mod] / NUM_ENSEMBLE);
          }
        }
      } else {
        assert (currCost.size() == 0);
      }

      for (int mod = 0; mod < heads.size(); ++mod) {
        assert (heads[mod] >= 0);
        if (heads[mod] == 0) count_root += 1;
      }
      assert (count_root >= 1); // a sentence must have at least one root (more for other languages like German)

      // the cost matrix (whether distillation or regular Hamming cost) is put back into the computation graph
      Expression cost_matrix = input(cg, {slen + 1, slen + 1}, cost);
      auto eigen_cost_matrix = *(cg.incremental_forward());
      // fill in the cost of the root token (col. 0) and all diagonals with a very small number
      // this is because the root token cannot have any heads and a token cannot be a head of itself (diagonal entries on the matrix)
      eigen_cost_matrix.col(0).fill(-9e+99);
      eigen_cost_matrix.diagonal().fill(-9e+99);
      
      assert (root_scores_vector.cols() == slen && root_scores_vector.rows() == 1 && result_matrix.rows() == slen && result_matrix.cols() == slen);
      eigen_cost_matrix.block(1, 1, slen, slen) = eigen_cost_matrix.block(1, 1, slen, slen) + result_matrix; 
      eigen_cost_matrix.block(0, 1, 1, slen) = eigen_cost_matrix.block(0, 1, 1, slen) + root_scores_vector;

      // Find the highest-scoring tree according to the model, which may not correspond with the real gold tree
      vector<int> arg_max;
      arg_max.push_back(-999);
      Eigen::VectorXf maxVal(eigen_cost_matrix.cols());
      Eigen::MatrixXf::Index maxIndex[eigen_cost_matrix.cols()];
      for (int i = 1; i < eigen_cost_matrix.cols(); ++i) {
        maxVal(i) = eigen_cost_matrix.col(i).maxCoeff(&maxIndex[i]);
        arg_max.push_back(maxIndex[i]);
      }
      vector<int> max_neg = arg_max;
      assert (arg_max.size() == eigen_cost_matrix.cols());

      vector<Expression> loss_sum;
      //max_neg is the highest-scoring head for each modifier, according to CLE
      Expression zero = zeroes(cg, {1});
      for (int mod = 0; mod < max_neg.size(); ++mod) {
        if (mod != 0) {
          assert (max_neg[mod] >= 0);
          if (max_neg[mod] != 0) { 
            float goldCost = cost[rc2ind(heads[mod-1], mod, slen + 1)];
            float currCost = cost[rc2ind(max_neg[mod], mod, slen + 1)];
            Expression goldCostExp = input(cg, goldCost);
            Expression currCostExp = input(cg, currCost);
            // cost-augmented score = model score + cost
            Expression currScore = result[rc2ind(max_neg[mod] - 1, mod - 1, slen)] + currCostExp;
            Expression goldScore;
            if (heads[mod-1] > 0) { 
             goldScore = result[rc2ind(heads[mod-1] - 1, mod - 1, slen)] + goldCostExp;
            } else {
	     goldScore = root_scores[mod - 1] + goldCostExp;
            }
            // the loss for picking a particular arc over the gold arc
            Expression currLoss = max(currScore - goldScore, zero);
            if (heads[mod-1] == max_neg[mod]) { // if the predicted head is correct, make sure the loss is zero
              double currLossScalar = as_scalar(cg.incremental_forward());
              assert(currLossScalar == 0.0);
            }
            loss_sum.push_back(currLoss);
          }

          else {//do the same for the root 
            float goldCost = cost[rc2ind(heads[mod-1], mod, slen + 1)];
            float currCost = cost[rc2ind(max_neg[mod], mod, slen + 1)];
            Expression goldCostExp = input(cg, goldCost);
            Expression currCostExp = input(cg, currCost);
            Expression currScore = root_scores[mod - 1] + currCostExp;
            Expression goldScore;
            if (heads[mod-1] > 0) {
             goldScore = result[rc2ind(heads[mod-1] - 1, mod - 1, slen)] + goldCostExp;
            } else {
             goldScore = root_scores[mod - 1] + goldCostExp;
            }
            Expression currLoss = max(currScore - goldScore, zero);
            if (heads[mod-1] == max_neg[mod]) { // if the predicted head is correct, make sure the loss is zero
              double currLossScalar = as_scalar(cg.incremental_forward());
              assert(currLossScalar == 0.0);
            }
            loss_sum.push_back(currLoss);
          }

        } else 
           assert (max_neg[mod] == -999);
      } 
      assert (loss_sum.size() == slen);

     // the total loss for the whole sentence is simply the sum of the loss of each arc
     Expression total_loss_exp = sum(loss_sum);
     double total_loss = as_scalar(cg.incremental_forward());
     assert (total_loss >= 0.0);
     //cerr << "Score difference (including cost) between the model's best tree and the gold tree is " << total_loss << endl;
     
     // part 1 component of the cost: difference between the model's best and gold tree, taking the cost into account
     Expression final_cost = total_loss_exp;
     // part 2: loss for the labeler 
     Expression MLP_labels = parameter(cg, p_MLP_labels); 
     Expression MLP_labels_bias = parameter(cg, p_MLP_labels_bias);
     Expression MLP_labels_softmax = parameter(cg, p_MLP_labels_softmax);
     Expression MLP_labels_softmax_bias = parameter(cg, p_MLP_labels_softmax_bias);
     // get the hidden layer representation for the labeller MLP
     vector<Expression> labels_cost;
     for (int mod = 0; mod < heads.size(); ++mod) {
       if (heads[mod] != 0) {
         assert (heads[mod]-1 != mod); //the head of a token should not be itself
         Expression MLP_labels_hidden = tanh(affine_transform({MLP_labels_bias, MLP_labels, concatenate({embedded[heads[mod]-1], embedded[mod]})})); 
         Expression labels_vec = log_softmax(affine_transform({MLP_labels_softmax_bias, MLP_labels_softmax, MLP_labels_hidden}));
         Expression label_cost = -pick(labels_vec, labels[mod]);
         double label_cost_scalar = as_scalar(cg.incremental_forward());
         assert (label_cost_scalar >= 0.0);
         labels_cost.push_back(label_cost);
       } else {
         labels_cost.push_back(zero);
       }
     }
     assert (labels_cost.size() == (heads.size()));
     Expression final_labels_cost = sum(labels_cost);
     assert (as_scalar(cg.incremental_forward()) >= 0.0);
     //part 3: regulariser
     Expression regulariser_cost_1 = sum(sum_MLP_squared_norm) * GAMMA / (sum_MLP_squared_norm.size()); 
     double regulariser_cost_1_scalar = as_scalar(cg.incremental_forward());
     Expression regulariser_cost_2 = BETA * squared_norm(MLP_output);
     double regulariser_cost_2_scalar = as_scalar(cg.incremental_forward());
     //cerr << "regulariser cost 1: " << regulariser_cost_1_scalar << " regulariser cost 2: " << regulariser_cost_2_scalar << endl;
     //Final part: add all the costs
     Expression truly_final_cost = final_cost + final_labels_cost + regulariser_cost_1 + regulariser_cost_2;
    } else {
      // we are not training, therefore the cost parts are not applicable  
    }
    return make_pair(root_scores_vector, result_matrix);
  }

  //Since this is a pipeline system, only label the head-modifier pairs predicted by the previous step. Returns whether the sentence contains multiple roots
  bool LabelDependencies(ComputationGraph& cg, const vector<Expression>& embedded, const vector<int>& heads, vector<int>& labels) {
    assert ((embedded.size() + 1) == heads.size());
    Expression MLP_labels = parameter(cg, p_MLP_labels);
    Expression MLP_labels_bias = parameter(cg, p_MLP_labels_bias);
    Expression MLP_labels_softmax = parameter(cg, p_MLP_labels_softmax);
    Expression MLP_labels_softmax_bias = parameter(cg, p_MLP_labels_softmax_bias); 
    int root_ctr = 0;
    //cerr << endl;
    for (int mod = 1; mod < heads.size(); ++mod) {
      assert (heads[mod] != mod); //a token cannot be the head of itself
      if (heads[mod] == 0) {
        labels.push_back(labels_dict.Convert("root"));
        ++root_ctr;
      } else {
         Expression MLP_labels_hidden = tanh(affine_transform({MLP_labels_bias, MLP_labels, concatenate({embedded[heads[mod]-1], embedded[mod-1]})}));
         Expression labels_vec = softmax(affine_transform({MLP_labels_softmax_bias, MLP_labels_softmax, MLP_labels_hidden}));      
         vector<float> labels_softmax = as_vector(cg.incremental_forward());
         float best_score = -9e+99;
         int best_idx = -999;
         assert (labels_softmax.size() == LABEL_SIZE);
         for (int i = 0; i < labels_softmax.size(); ++i) {
           if (labels_softmax[i] > best_score) {
             best_score = labels_softmax[i];
             best_idx = i;
           } 
         }
         assert (best_score > -9e+99 && best_idx >= 0 && best_idx < LABEL_SIZE);
         labels.push_back(best_idx);
      }
    }
    if (root_ctr == 0) cerr << "Error! No root of the sentence!" << endl;
    assert (labels.size() == embedded.size());
    if (root_ctr == 1) return false;
    return true;
  }
};


void signal_callback_handler(int /* signum */) {
  if (requested_stop) {
    cerr << "\nReceived SIGINT again, quitting.\n";
    _exit(1);
  }
  cerr << "\nReceived SIGINT terminating optimization early...\n";
  requested_stop = true;
}

void output_to_conll(const string& file, const vector<vector<int>>&  sentenceWordsTest, const vector<vector<int>>& sentencePosTest, const vector<vector<int>>& results, const vector<vector<int>>& labels) {
  assert (sentenceWordsTest.size() == sentencePosTest.size() && sentenceWordsTest.size() == results.size());
  ifstream in(file.c_str());
  string line;
  int sent_ctr = 0;
  int word_ctr = -1;
   while (getline(in, line)) {
      if (line.length() > 1) {
        ++word_ctr;
        istringstream iss(line);
        vector<string> elems;
        string word;
        // parse the CoNLL format, 10 entries per line, separated by tab 
        while(iss >> word) {
          elems.push_back(word);
        }
        assert (elems.size() == 10);
        for (int i = 0; i < elems.size(); ++i) {
          if (i != 6 && i != 7 && i != 9) {
            cout << elems[i] << "\t";
          } else if (i == 6) {
            cout << results[sent_ctr][word_ctr + 1] << "\t"; 
          } else if (i == 7) {
            cout << labels_dict.Convert(labels[sent_ctr][word_ctr]) << "\t";
          } else if (i == 9) {
            cout << elems[i] << endl;
          }
        }
      } else {
        word_ctr = -1;
        sent_ctr++;
        cout << endl;
      }
  }
}

pair<float, unsigned> readSentences(const string& file, vector<vector<int>>& sentenceWords, vector<vector<int>>& sentencePrets, vector<vector<int>>& sentencePos, vector<vector<int>>& sentenceHeads, vector<vector<int>>& sentenceLabels) {
    ifstream in(file.c_str());
    string line;
    vector<int> curr_word, curr_pos, curr_heads, curr_prets, curr_labels;
    float total_length = 0.0f;
    float num_sentences = 0.0f;
    unsigned max_length = 0;
    while (getline(in, line)) {
      if (line.length() > 1) {
        istringstream iss(line);
        vector<string> elems;
        string word;
        // parse the CoNLL format, 10 entries per line, separated by tab 
        while(iss >> word) {
          elems.push_back(word);
        }
        assert (elems.size() == 10);
        curr_word.push_back(words_dict.Convert(elems[1]));
        curr_prets.push_back(pret_dict.Convert(elems[1]));
        curr_pos.push_back(pos_dict.Convert(elems[4]));
        curr_heads.push_back(atoi(elems[6].c_str()));
        curr_labels.push_back(labels_dict.Convert(elems[7]));
      }
      else {
        // write out the sentence
        assert (curr_word.size() == curr_pos.size() && curr_pos.size() == curr_heads.size() && curr_prets.size() == curr_word.size() && curr_word.size() == curr_labels.size());
        sentenceWords.push_back(curr_word);
        sentencePos.push_back(curr_pos);
        sentenceHeads.push_back(curr_heads);
        sentencePrets.push_back(curr_prets);
        sentenceLabels.push_back(curr_labels);
        if (curr_word.size() > max_length) max_length = curr_word.size(); 
        total_length += ((float) curr_word.size());
        num_sentences += 1.0f;
        curr_word.clear();
        curr_pos.clear();
        curr_heads.clear();
        curr_prets.clear();
        curr_labels.clear();
      }
    }
    if (!curr_word.empty() || !curr_pos.empty() || !curr_heads.empty() || !curr_prets.empty() || !curr_labels.empty()) {
       assert (curr_word.size() == curr_pos.size() && curr_pos.size() == curr_heads.size() && curr_prets.size() == curr_word.size() && curr_word.size() == curr_labels.size());
       sentenceWords.push_back(curr_word);
       sentencePos.push_back(curr_pos);
       sentenceHeads.push_back(curr_heads);
       sentencePrets.push_back(curr_prets);
       sentenceLabels.push_back(curr_labels);
       if (curr_word.size() > max_length) max_length = curr_word.size(); 
       total_length += ((float) curr_word.size());
       num_sentences += 1.0f;
       curr_word.clear();
       curr_pos.clear();
       curr_heads.clear();
       curr_prets.clear();
       curr_labels.clear();
    }
  assert (sentenceWords.size() == sentencePos.size() && sentenceWords.size() == sentenceHeads.size() && sentenceWords.size() == sentencePrets.size() && sentenceWords.size() == sentenceLabels.size());
  if (N_WORDS == 0) 
  N_WORDS = total_length;
  return make_pair(total_length / num_sentences, max_length);
}

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  cerr << "COMMAND: ";
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;
  unsigned status_every_i_iterations = 100;
  // initialise the program options
  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);
  // initialise the dimensions according to the program options
  INPUT_DIM = conf["input_dim"].as<unsigned>();
  HIDDEN_DIM = conf["hidden_dim"].as<unsigned>();
  LSTM_INPUT_DIM = conf["lstm_input_dim"].as<unsigned>();
  POS_DIM = conf["pos_dim"].as<unsigned>();
  LAYERS = conf["layers"].as<unsigned>();
  MLP_DIM = conf["mlp_dim"].as<unsigned>();
  BETA = conf["beta"].as<float>();
  GAMMA = conf["gamma"].as<float>();
  if (conf.count("dropout"))
    DROPOUT = conf["dropout"].as<float>();

  vector<vector<int>> sentenceWords, sentencePos, sentenceHeads, sentencePrets, sentenceLabels;
  vector<vector<int>> sentenceWordsDev, sentencePosDev, sentenceHeadsDev, sentencePretsDev, sentenceLabelsDev;
  vector<vector<int>> sentenceWordsTest, sentencePosTest, sentenceHeadsTest, sentencePretsTest, sentenceLabelsTest;
  vector<vector<float>> cost_matrix;
  //load the training set 
  if (conf.count("learn") && conf.count("dev") == 0) {
    cerr << "You specified --learn but did not specify --dev FILE\n";
    return 1;
  }
  if ((conf.count("words") && conf.count("pretrained_dim") == 0) || (conf.count("pretrained_dim") && conf.count("words") == 0)) {
    cerr << "Either --words or --pretrained_dim is specified, but the other is not. Both must be specified" << endl;
    return 1;
  }
  //Load pre-trained word embedding
  if (conf.count("words")) {
    PRET_DIM = conf["pretrained_dim"].as<unsigned>();
    cerr << "Loading pre-trained word embedding from " << conf["words"].as<string>() << " with" << PRET_DIM << " dimensions\n";
    ifstream in(conf["words"].as<string>().c_str());
    string line;
    getline(in, line); // get the first line of the pre-trained word embedding, dummy element
    vector<float> v(PRET_DIM, 0);
    string word;
    unsigned total = 0;
    unsigned included = 0;
    while (getline(in, line)) {
      total++;
      istringstream lin(line);
      lin >> word;
      if (words_dict.Contains(word))
        included++; //TODO: fix this bit since it's outputting -nan
      for (unsigned i = 0; i < PRET_DIM; ++i) lin >> v[i];
      unsigned id = pret_dict.Convert(word);
      pretrained[id] = v;
    }
    pret_dict.Freeze();
    pret_dict.SetUnk("kUnk");
    cerr << "pretrained.size(): " << pretrained.size() << " total: " << total << " pret_dict.size(): " << pret_dict.size() << endl;
    assert (pretrained.size() == total && pret_dict.size() == (total + 1));
    cerr << "Total number of tokens in the pre-trained embedding: " << total << endl;
    cerr << included << " words in the vocabulary are covered in the pre-trained word embedding, from a total of " << VOCAB_SIZE << endl;
    cerr << "Word embedding coverage: " << ((float) included) / ((float) words_dict.size()) * 100.0f << "%" << endl;
    cerr << "Sanity check for total number of tokens in the pre-trained embedding: " << pretrained.size() << endl;
    //cerr << "Sanity check for word embedding coverage: " << ((float) pretrained.size()) / ((float) words_dict.size()) * 100.0f << "%" << endl;
    cerr << endl;
  } else {
    cerr << "Not using pre-trained embedding" << endl;
    pret_dict.Freeze();
    pret_dict.SetUnk("kUnk");
    cerr << "pretrained.size(): " << pretrained.size() << " pret_dict.size(): " << pret_dict.size() << endl;
  }

  //Load the training data
  cerr << "Now loading the training data from " << conf["train"].as<string>() << endl;
  auto avg_train = readSentences(conf["train"].as<string>(), sentenceWords, sentencePrets, sentencePos, sentenceHeads, sentenceLabels);
  if (conf.count("words")) {
    assert (!(sentencePrets == sentenceWords));
  }
  assert (sentenceWords.size() == sentencePos.size() && sentencePos.size() == sentenceHeads.size() && sentencePrets.size() == sentenceHeads.size() && sentenceWords.size() == sentenceLabels.size());
  cerr << "Number of train sentences " << sentenceWords.size() << " with average sentence length of " << avg_train.first << " and maximum length of " << avg_train.second << endl;
  cerr << "Total number of words in the training set: " << N_WORDS << endl;
  //cerr << "Length of the second sentence " << sentenceWords[2].size() << " " << words_dict.Convert(sentenceWords[2][8]) << " " << pos_dict.Convert(sentencePos[2][9]) << " " << sentenceHeads[2][10] << endl;
  VOCAB_SIZE = words_dict.size() + 1; //account for UNK
  POS_SIZE = pos_dict.size();
  LABEL_SIZE = labels_dict.size();
  cerr << endl;
  cerr << "Vocabulary size for the training data: " << VOCAB_SIZE << endl;
  cerr << "POS Tag vocabulary size for the training data: " << POS_SIZE << endl;
  cerr << "Number of labels in the training data: " << LABEL_SIZE << endl;

  words_dict.Freeze();
  words_dict.SetUnk("kUNK");
  pos_dict.Freeze();
  labels_dict.Freeze();

  singletons = getSingletons(sentenceWords);
  cerr << "Number of singletons: " << singletons.size() << endl;

  //Loading the dev data

  if (conf.count("dev")) {
    cerr << endl;
    cerr << "Now loading the dev data from " << conf["dev"].as<string>() << endl;
    auto avg_dev = readSentences(conf["dev"].as<string>(), sentenceWordsDev, sentencePretsDev, sentencePosDev, sentenceHeadsDev, sentenceLabelsDev);
    assert (sentenceWordsDev.size() == sentencePosDev.size() && sentencePosDev.size() == sentenceHeadsDev.size() && sentencePretsDev.size() == sentenceHeadsDev.size() && sentenceWordsDev.size() == sentenceLabelsDev.size());
    assert (sentenceWordsDev != sentencePretsDev);
    cerr << "Number of dev sentences " << sentenceWordsDev.size() << " with average sentence length of " << avg_dev.first << " and maximum length of " << avg_dev.second <<  endl;
    cerr << endl;
  }

  //Loading the test data
  if (conf.count("test")) {
    cerr << endl;
    cerr << "Now loading the test data from " << conf["test"].as<string>() << endl;
    auto avg_test = readSentences(conf["test"].as<string>(), sentenceWordsTest, sentencePretsTest, sentencePosTest, sentenceHeadsTest, sentenceLabelsTest);
    assert (sentenceWordsTest.size() == sentencePosTest.size() && sentencePosTest.size() == sentenceHeadsTest.size() && sentencePretsTest.size() == sentenceHeadsTest.size() && sentenceWordsTest.size() == sentenceLabelsTest.size());
    assert (sentenceWordsTest != sentencePretsTest);
    cerr << "Number of test sentences " << sentenceWordsTest.size() << " with average sentence length of " << avg_test.first << " and maximum length of " << avg_test.second <<  endl;
    cerr << endl;
  }

  if (conf.count("cost_matrix")) {
    cerr << "Specified the ensemble training mode" << endl;
    NUM_ENSEMBLE = ((unsigned) conf["num_ensemble"].as<float>());  
    USE_ENSEMBLE = true;
    cerr << "Training: ENSEMBLE MODE" << endl;
    cerr << "Loading the ensemble training file from: " << conf["cost_matrix"].as<string>() << endl;
    ifstream in(conf["cost_matrix"].as<string>().c_str());
    string line;
    int ctr = -1;
    while (getline(in, line)) {
      ++ctr;
      int curr_size = (sentenceWords[ctr].size() + 1) * (sentenceWords[ctr].size() + 1);
      vector<float> v(curr_size, -999.0);   
      istringstream lin(line);
      for (int i = 0; i < curr_size; ++i) {
        lin >> v[i];
        assert (v[i] >= 0.0 && v[i] <= ((float) NUM_ENSEMBLE));
      }
      assert (find(v.begin(), v.end(), -999.0) == v.end()); //assert that all the counts for the sentence have been covered
      cost_matrix.push_back(v);
    }
    assert(cost_matrix.size() == sentenceWords.size()); 
  }

  USE_POS = conf.count("use_pos_tags");
  cerr << "Do we use POS tags? " << USE_POS << endl;

  ETA_DECAY = conf["eta_decay"].as<float>();

  Model model;
  ParserBuilder<LSTMBuilder> parser(model, pretrained);  
  ostringstream os;
  os << "graphparse"
     << (USE_POS ? "_pos" : "")
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << '_' << LSTM_INPUT_DIM
     << '_' << MLP_DIM
     << '_' << DROPOUT 
     << '_' << BETA 
     << '_' << GAMMA 
     << '_' << USE_ENSEMBLE 
     << '-' << NUM_ENSEMBLE
     << '-' << ETA_DECAY 
     << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "PARAMETER FILE: " << fname << endl;
  bool softlinkCreated = false;

  if (conf.count("model")) {
    cerr << "Loading the model from " << conf["model"].as<string>() << endl;
    ifstream in(conf["model"].as<string>().c_str());
    boost::archive::text_iarchive ia(in);
    ia >> model;
  } 
  //train the model
  if (conf.count("learn")) {
    signal(SIGINT, signal_callback_handler); //TODO: handle sigint
    //SimpleSGDTrainer sgd(&model);
    AdamTrainer sgd(&model);
    //MomentumSGDTrainer sgd(&model);
    //sgd.eta_decay = 0.08;
    //sgd.eta_decay = 0.05;
    sgd.eta_decay = ETA_DECAY;
    cerr << "Training started."<<"\n";
    vector<unsigned> order(sentenceWords.size());
    for (unsigned i = 0; i < sentenceWords.size(); ++i)
      order[i] = i;
    double tot_seen = 0;
    if (status_every_i_iterations > sentenceWords.size()) status_every_i_iterations = sentenceWords.size(); 
    cerr << "Reporting train every " << status_every_i_iterations << " iterations" << endl; 
    unsigned si = sentenceWords.size();
    cerr << "NUMBER OF TRAINING SENTENCES: " << si << endl;
    unsigned trs = 0;
    double llh = 0;
    bool first = true;
    int iter = -1;
    double best_dev = -999.0;
    while(!requested_stop) {
      ++iter;
      double neg = 0.0f;
      for (unsigned sii = 0; sii < status_every_i_iterations; ++sii) {
           if (si == sentenceWords.size()) {
             si = 0;
             if (first) { first = false; } else { sgd.update_epoch(); }
             cerr << "**SHUFFLE\n";
             random_shuffle(order.begin(), order.end());
           }
           tot_seen += 1;
           int idx = order[si];
           const vector<int>& currWords = sentenceWords[idx];
           const vector<int>& currPos = sentencePos[idx];
           const vector<int>& currHeads = sentenceHeads[idx];
           const vector<int>& currPrets = sentencePrets[idx];
           const vector<int>& currLabels = sentenceLabels[idx];
           assert (currWords.size() == currPos.size() && currWords.size() == currHeads.size() && currPrets.size() == currWords.size() && currWords.size() == currLabels.size());
           ComputationGraph cg;
           vector<Expression> embedded;
           if (USE_ENSEMBLE) {
             const vector<float>& currCost = cost_matrix[idx];
             assert (currCost.size() == (currWords.size() + 1) * (currWords.size() + 1));
             parser.BuildGraph(currWords, currPrets, currPos, cg, currHeads, currLabels, true, embedded, currCost);
           } else {
             vector<float> dummy;
             parser.BuildGraph(currWords, currPrets, currPos, cg, currHeads, currLabels, true, embedded, dummy);
           }
           double lp = as_scalar(cg.incremental_forward());
           if (lp < 0 || std::isnan(lp)) {
             cerr << "Negative cost < 0 or NaN on sentence " << order[si] << ": lp=" << lp << endl;
             neg += 1.0;
             assert(lp >= 0.0);
           }
           if (lp >= 0 && !std::isnan(lp)) {
             cg.backward();
             sgd.update(1.0);
             llh += lp;
             trs += currWords.size();
           }
           ++si;
           trs += 1;
      }
      sgd.status();
      cerr << "update #" << iter << " (epoch " << (tot_seen / sentenceWords.size()) << ")\ttotal loss: "<< llh<<" average loss: " << (llh / ((float) trs)) << ", and " << neg / ((float) status_every_i_iterations) * 100.0f << " % of training instances for this update have <0 nll:" << endl; 
      llh = trs = neg = 0;
      static int logc = 0;
      ++logc;
      if (logc % 25 == 1) { // report on dev set
        cerr << "Validating on the dev set" << endl;
        unsigned dev_size = sentenceWordsDev.size();
        double llh = 0;
        auto t_start = std::chrono::high_resolution_clock::now();
        double trs = 0.0f;
        double neg_dev = 0.0f;
        double right = 0.0;
        for (unsigned sii = 0; sii < dev_size; ++sii) {
           const vector<int>& currWords = sentenceWordsDev[sii];
           const vector<int>& currPos = sentencePosDev[sii];
           const vector<int>& currHeads = sentenceHeadsDev[sii];
           const vector<int>& currPrets = sentencePretsDev[sii];
           const vector<int>& currLabels = sentenceLabelsDev[sii];
           assert (currWords.size() == currPos.size() && currWords.size() == currHeads.size() && currWords.size() == currPrets.size() && currLabels.size() == currWords.size());
           ComputationGraph cg;
           vector<Expression> embedded;
           vector<float> dummy;
           auto root_result_matrices = parser.BuildGraph(currWords, currPrets, currPos, cg, currHeads, currLabels, false, embedded, dummy);
           //double lp = as_scalar(cg.incremental_forward());
           double lp = 0.0f;
           if (lp < 0 || std::isnan(lp)) {
             cerr << "Negative log prob < 0 or NaN on dev sentence " << sii << ": lp=" << lp << endl;
             neg_dev += 1;
             assert(lp >= 0.0);
           } else {
             llh += lp;
             trs += currWords.size();
             vector<int> result_CLE = Eisner(root_result_matrices.first, root_result_matrices.second);             
             // Compute the accuracy
             assert (result_CLE.size() == (currHeads.size() + 1));
             for (int i = 0; i < currHeads.size(); ++i) {
               if (currHeads[i] == result_CLE[i+1]) right += 1.0;
             }
           }
        }
        auto t_end = std::chrono::high_resolution_clock::now();
        cerr << "  **dev (iter=" << iter << " epoch=" << (tot_seen / sentenceWords.size()) << ")\tllh=" << llh << " ppl: " << exp(llh / trs) << ", and " << neg_dev / ((float) dev_size) * 100.0f << " % of dev instances have <0 nll:" << endl; 
        cerr << " Accuracy: " << right / trs * 100.0 << endl;
        if ((right / trs * 100.0) > best_dev) {
          best_dev = right / trs * 100.0;
          cerr << "  new best...writing model to " << fname << " ...\n";
          ofstream out(fname);
          boost::archive::text_oarchive oa(out);
          oa << model;
          if (!softlinkCreated) {
            string softlink = " latest_model";
            if (system((string("rm -f ") + softlink).c_str()) == 0 &&
                system((string("ln -s ") + fname + " " + softlink).c_str()) == 0) {
              cerr << "Created " << softlink << " as a soft link to " << fname
                   << " for convenience." << endl;
            }
            softlinkCreated = true;
          } 
        }
        //cerr << "Time taken: " << t_end - t_start << endl;
       cerr << "End validating" << endl;
      }
    }
  }
  if (conf.count("test")) {
    if (conf.count("learn")) {
      cerr << "The model can't both learn and test at the same time. Please specify only one option, either --test or --learn" << endl;
      assert (!conf.count("learn"));
    }
    assert (sentenceHeadsTest.size() > 0); //assert that we have read all the sentences in the test set
    cerr << "Testing on the test set" << endl;
    unsigned test_size = sentenceWordsTest.size();
    double llh = 0;
    double right = 0.0;
    double total = 0.0;
    vector<vector<int>> results;
    vector<vector<int>> labels;
    int ctr_multiple_roots = 0;
    for (unsigned sii = 0; sii < test_size; ++sii) {
       const vector<int>& currWords = sentenceWordsTest[sii];
       const vector<int>& currPos = sentencePosTest[sii];
       const vector<int>& currHeads = sentenceHeadsTest[sii];
       const vector<int>& currPrets = sentencePretsTest[sii];
       const vector<int>& currLabels = sentenceLabelsTest[sii];
       assert (currWords.size() == currPos.size() && currWords.size() == currHeads.size() && currWords.size() == currPrets.size() && currWords.size() == currLabels.size());
       ComputationGraph cg;
       vector<Expression> embedded;
       vector<float> dummy;
       auto root_result_matrices = parser.BuildGraph(currWords, currPrets, currPos, cg, currHeads, currLabels, false, embedded, dummy);
       //double lp = as_scalar(cg.incremental_forward());
       double lp = 0.0f;
       if (lp < 0 || std::isnan(lp)) {
         cerr << "Negative log prob < 0 or NaN on dev sentence " << sii << ": lp=" << lp << endl;
         assert(lp >= 0.0);
       } else {
         llh += lp;
         vector<int> result_CLE = Eisner(root_result_matrices.first, root_result_matrices.second);
         //cerr << endl;
         vector<int> labels_sent;
         //cerr << "Current sentence: " << sii + 1 << endl;
         bool multi_root = parser.LabelDependencies(cg, embedded, result_CLE, labels_sent);
         if (multi_root) ++ctr_multiple_roots;
         assert (result_CLE.size() == (currHeads.size() + 1));
         assert (result_CLE[0] == -1);
         results.push_back(result_CLE);
         labels.push_back(labels_sent);
         assert (result_CLE.size() == (labels_sent.size() + 1));
         // Compute the accuracy
         for (int i = 0; i < currHeads.size(); ++i) {
           if (currHeads[i] == result_CLE[i+1]) right += 1.0;
           total += 1.0;
           //cerr << "Gold: " << currHeads[i] << " Predicted: " << result_CLE[i+1] << endl;
         }
       }
    }
    assert (results.size() == labels.size());
    cerr << "**test: Accuracy with punctuations: " << right / total * 100.0 << endl;
    cerr << "Number of sentences with multiple roots: " << ctr_multiple_roots << endl;
    cerr << "Writing down the output file" << endl;
    output_to_conll(conf["test"].as<string>(), sentenceWordsTest, sentencePosTest, results, labels);
    cerr << "All done!" << endl;
      
  }
  
}
