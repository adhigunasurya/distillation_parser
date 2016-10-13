#include <cstdlib>
#include <algorithm>
#include <vector>
#include <limits>
#include <cmath>
#include <chrono>
#include <ctime>
#include <iostream>

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>

#include <unordered_map>
#include "Algorithms.hpp"

#include <Eigen/Dense>

//int ROOT_HEAD = -999;

using namespace std;


vector<int> Eisner(const Eigen::MatrixXf& root_score_vectors, const Eigen::MatrixXf& result_matrix) {
  assert (result_matrix.rows() == result_matrix.cols() && root_score_vectors.rows() == 1 && root_score_vectors.cols() == result_matrix.cols());
  unsigned slen = result_matrix.rows();
  vector<double> scores; 
  vector<vector<int>> index_arcs;
  index_arcs.assign(slen + 1, vector<int>(slen + 1, -1));
  int ctr = -1;
  //cerr << "Milestone 1" << endl;
  for (unsigned i = 0; i < root_score_vectors.cols(); ++i) {
    ++ctr;
    scores.push_back(root_score_vectors(0, i)); 
    index_arcs[0][i+1] = ctr; 
  }
  //cerr << "Milestone 2" << endl;
  for (int h = 0; h < slen; ++h) {
    for (int m = 0; m < slen; ++m) {
      if (h != m) {
        ++ctr;
	scores.push_back(result_matrix(h, m));
        index_arcs[h+1][m+1] = ctr;
      }
      else {
        index_arcs[h+1][m+1] = -1;
      }
    }
  }
  for (int m = 0; m < slen; ++m) {
    index_arcs[m+1][0] = -1;
  }
  index_arcs[0][0] = -1;
  //cerr << "Milestone 3" << endl;
  assert (scores.size() == (slen * slen));
  vector<int> heads(slen + 1);
  double fscore = -9e+99;
  RunEisner(scores, &heads, &fscore, slen + 1, index_arcs);
  assert (fscore >= -9e+99);
  //cerr << "Score of the whole tree: " << fscore << endl;
  return heads;
}

vector<int> Eisner_recursive(const Eigen::MatrixXf& all_scores) {
  assert (all_scores.rows() == all_scores.cols());
  unsigned slen = all_scores.rows() - 1;
  vector<double> scores;
  vector<vector<int>> index_arcs;
  index_arcs.assign(slen + 1, vector<int>(slen + 1, -1));
  int ctr = -1;
  for (unsigned i = 1; i < all_scores.row(0).cols(); ++i) {
    ++ctr;
    scores.push_back(all_scores(0, i));
    index_arcs[0][i] = ctr;
  }
  for (int h = 1; h < (slen + 1); ++h) {
    for (int m = 1; m < (slen + 1); ++m) {
      if (h != m) {
        ++ctr;
        scores.push_back(all_scores(h, m));
        index_arcs[h][m] = ctr;
      }
    }
  }
  assert (scores.size() == (slen * slen));
  vector<int> heads(slen + 1);
  double fscore = -9e+99;
  RunEisner(scores, &heads, &fscore, slen + 1, index_arcs);
  assert (fscore >= -9e+99);
  //cerr << "Score of the whole tree: " << fscore << endl;
  return heads;
}
