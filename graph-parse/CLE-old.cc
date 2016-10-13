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

#include <Eigen/Dense>

int ROOT_HEAD = -999;

using namespace std;

void initialiseMatrix(Eigen::MatrixXf& all_scores) {
 all_scores.fill(-9e+99);
 /*
 for (int row = 0; row < all_scores.rows(); ++row) {
    for (int col = 0; col < all_scores.cols(); ++col) {
      all_scores(row, col) = -9e+99;
    }
  } */
}

bool detect_cycles(vector<int>& arg_max) {
  for (int i = 0; i < arg_max.size(); ++i) {
    if (i != 0) {
      if (arg_max[arg_max[i]] == i) {
        return true;
      }
    }
    else assert(arg_max[i] == ROOT_HEAD);
  }
  return false;
}

std::pair<int, int> return_cycles(vector<int>& arg_max) {
  for (int i = 0; i < arg_max.size(); ++i) {
    if (i != 0) {
      if (arg_max[arg_max[i]] == i) {
        return std::make_pair(i, arg_max[i]);
      }
    }
  }
  cerr << "Returning cycles when there is none" << endl;
  abort();
}

void condenseMatrix(const Eigen::MatrixXf& all_scores, const std::pair<int, int>& cycles, unordered_map<int, int>& mapping, unordered_map<int, int>& reverse_mapping, int& map_idx_result, Eigen::MatrixXf& condensedMatrix) {
  assert (all_scores.rows() == all_scores.cols());
  assert (cycles.first != cycles.second);
  int map_idx = -999;
  int curr_idx = 0;
  //map the indices
  for (int i = 0; i < all_scores.rows(); ++i) {
    if (i != cycles.first && i != cycles.second) { 
      mapping[i] = curr_idx; 
      ++curr_idx;
    } else {
      if (map_idx == -999) {
        mapping[i] = curr_idx;
        map_idx = curr_idx;
        ++curr_idx;
      } else {
        mapping[i] = map_idx;
      }
    }
  }
  assert (mapping.size() == all_scores.rows());
  assert (map_idx != -999);
  map_idx_result = map_idx;
  //reverse map the resulting mapping
  for (int i = 0; i < all_scores.rows(); ++i) {
    reverse_mapping[mapping[i]] = i;
  }
  assert (reverse_mapping.size() == (all_scores.rows() - 1));
  assert (mapping[0] == 0); // the dummy root symbol should never be involved in a cycle
  initialiseMatrix(condensedMatrix);
  //build the new matrix
  for (int row = 0; row < all_scores.rows(); ++row) {
    for (int col = 0; col < all_scores.cols(); ++col) {
      if (row != col) {
        if (mapping[row] != map_idx && mapping[col] != map_idx) { //neither the source nor the target is in the cycle node
          condensedMatrix(mapping[row], mapping[col]) = all_scores(row, col);
        } 
        else if (mapping[row] == map_idx && mapping[col] != map_idx) { //the source is in the cycle node but the target is not
          if (condensedMatrix(mapping[row], mapping[col]) < all_scores(row, col)) condensedMatrix(mapping[row], mapping[col]) = all_scores(row, col);
        } 
        else if (mapping[row] != map_idx && mapping[col] == map_idx) { //the source is not in the cycle node but the target is
          float head_score = 0.0f; 
          assert (col == cycles.first || col == cycles.second);
          if (col == cycles.first)
            head_score = all_scores(cycles.first, cycles.second);
          else
            head_score = all_scores(cycles.second, cycles.first);
          //cerr << "head score: " << head_score << endl;
          float curr_score = all_scores(row, col) + head_score;
          if (condensedMatrix(mapping[row], mapping[col]) < curr_score) condensedMatrix(mapping[row], mapping[col]) = curr_score;
        }
      }
    }
  } 
}

vector<int> CLE_recursive(const Eigen::MatrixXf& all_scores) {
  assert (all_scores.cols()  == all_scores.rows()); //assert it is a square matrix
  //find the highest scoring head for each modifier
  //auto& maxCols = all_scores.colwise().maxCoeff();
  vector<int> arg_max;
  arg_max.push_back(ROOT_HEAD);
  Eigen::VectorXf maxVal(all_scores.cols());
  Eigen::MatrixXf::Index maxIndex[all_scores.cols()];
  for (int i = 1; i < all_scores.cols(); ++i) {
    maxVal(i) = all_scores.col(i).maxCoeff(&maxIndex[i]); 
    arg_max.push_back(maxIndex[i]);
  }
  //for each node except the root, find the highest-scoring incoming edge
  /*
  for (int col = 1; col < all_scores.cols(); ++col) {
    bool found = false;
    for (int row = 0; row < all_scores.rows(); ++row) {
      if (all_scores(row, col) == maxCols(col)) {
        arg_max.push_back(row);
        found = true;
        assert(row != col);
        break;
      } 
    }
    assert(found);
  } */
  assert (arg_max.size() == all_scores.cols()) ;
  //check for cycles, if not, then the highest incoming edges are the best ones and form a complete tree, all set
  if (!detect_cycles(arg_max)) {
    //cerr << "No cycles" << endl;
    return arg_max;
  }
  //in case of cycles
  else {
    for (int i = 0; i < arg_max.size(); ++i) {
      //cerr << i << " : " << arg_max[i] << endl;
    }
    //cerr << all_scores << endl;
    // get the first cycle from the tree
    pair<int, int> cycles = return_cycles(arg_max);
    unordered_map<int, int> mapping;
    unordered_map<int, int> reverse_mapping;
    int map_idx = ROOT_HEAD;
    Eigen::MatrixXf condensedMatrix(all_scores.rows() - 1, all_scores.cols() - 1);
    //cerr << "cycles: " << cycles.first << " " << cycles.second << endl;
    condenseMatrix(all_scores, cycles, mapping, reverse_mapping, map_idx, condensedMatrix); 
    //cerr << "condensed matrix" << endl << condensedMatrix << endl;
    assert (map_idx != ROOT_HEAD);
    //apply the CLE algorithm recursively
    vector<int> recursive_arg_max = CLE_recursive(condensedMatrix);
    //cerr << "Printing the recursive result" << endl;
    for (int i = 0; i < recursive_arg_max.size(); ++i) {
      //cerr << i << " " << recursive_arg_max[i] << endl;
    }
    //cerr << "Done printing the recursive result" << endl;
    assert (recursive_arg_max.size() == (all_scores.rows() - 1));
    //map back the highest-scoring edges from the condensed space to the original space
    vector<int> result(all_scores.rows(), ROOT_HEAD);
     result[0] = ROOT_HEAD;
    for (int i = 1; i < recursive_arg_max.size(); ++i) {
      if (i != map_idx && recursive_arg_max[i] != map_idx) { //first case, neither the modifier nor the head is the cycle node. Straightforward
        result[reverse_mapping[i]] = reverse_mapping[recursive_arg_max[i]];
      } else if (i != map_idx && recursive_arg_max[i] == map_idx) { //second case, the modifier is not one of the cycle nodes but the head is one of the cycle nodes. Investigate which one is the head
        assert (all_scores(cycles.first, reverse_mapping[i]) == condensedMatrix(map_idx, i) || all_scores(cycles.second, reverse_mapping[i]) == condensedMatrix(map_idx, i));
        //cerr << "Score from: " << cycles.first << " " << all_scores(cycles.first, reverse_mapping[i]) << endl;
        //cerr << "Score from: " << cycles.second << " " << all_scores(cycles.second, reverse_mapping[i]) << endl;
        float option_1 = all_scores(cycles.first, reverse_mapping[i]);
        float option_2 = all_scores(cycles.second, reverse_mapping[i]);
        //cerr << "option 1: " << option_1 << " option 2: " << option_2 << endl;
        if (all_scores(cycles.first, reverse_mapping[i]) < all_scores(cycles.second, reverse_mapping[i])) {
          result[reverse_mapping[i]] = cycles.second;
        } else {
          result[reverse_mapping[i]] = cycles.first;
        } 
      } else if (i == map_idx && recursive_arg_max[i] != map_idx) { //third case, the modifier is one of the cycle nodes but the head is not any of the cycle nodes
          float curr_val = condensedMatrix(recursive_arg_max[i], map_idx);
          float first_possibility = all_scores(reverse_mapping[recursive_arg_max[i]], cycles.first) + all_scores(cycles.first, cycles.second);
          float second_possibility = all_scores(reverse_mapping[recursive_arg_max[i]], cycles.second) + all_scores(cycles.second, cycles.first);
          //cerr << "First, second possibilities: " << first_possibility << " " << second_possibility << endl;
          assert(curr_val == first_possibility || curr_val == second_possibility);
          if (curr_val == first_possibility) {
            result[cycles.first] = reverse_mapping[recursive_arg_max[i]];
            result[cycles.second] = cycles.first;
          } else {
            result[cycles.second] = reverse_mapping[recursive_arg_max[i]];
            result[cycles.first] = cycles.second;
          }
        }
      }    
    assert (result.size() == all_scores.rows());
    assert (std::find(result.begin(), result.end(), ROOT_HEAD) == result.begin() && std::count(result.begin(), result.end(), ROOT_HEAD) == 1);
    return result;
    } 
}

vector<int> CLE(const Eigen::MatrixXf& root_score_vectors, const Eigen::MatrixXf& result_matrix) {
  assert (result_matrix.rows() == result_matrix.cols() && root_score_vectors.rows() == 1 && root_score_vectors.cols() == result_matrix.cols());
  Eigen::MatrixXf all_scores(result_matrix.rows() + 1, result_matrix.cols() + 1);
  // initialise all elements with very small values
  initialiseMatrix(all_scores);
  // initialise the root row 
  all_scores.block(0, 1, 1, root_score_vectors.cols()) = root_score_vectors.row(0);
  // initialise all other rows
  for (int i = 0; i < result_matrix.rows(); ++i) {
    all_scores.block(i+1, 1, 1, result_matrix.cols()) = result_matrix.row(i);
  }
  //assign small values to the diagonals
  all_scores.diagonal().fill(-9e+99);
  /*
  for (int i = 0; i < all_scores.rows(); ++i) {
    all_scores(i, i) = -9e+99;
  } */
  //get the highest scoring 
  vector<int> result = CLE_recursive(all_scores);
  //cerr << "Printing the final result" << endl;
  for (int i = 0; i < result.size(); ++i)
    //cerr << "Head of " << i << " is " << result[i] << endl; 
  //cerr << "ALL DONE" << endl;
  return result;
}

