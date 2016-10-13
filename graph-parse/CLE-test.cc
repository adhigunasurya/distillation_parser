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
#include "CLE.cc"

using namespace std;

int main() {
   /*
  Eigen::MatrixXf mat(2,4);
  mat << 1, 2, 6, 9,
         3, 1, 7, 2;
  cout << mat << endl; 
  std::cout << "Column's maximum: " << std::endl
   << mat.colwise().maxCoeff() << std::endl;
  Eigen::MatrixXf::Index   maxIndex[4];
  Eigen::VectorXf maxVal(4);
  vector<int> idx(4,0);
  for (int i = 0; i < mat.cols(); ++i) {
    maxVal(i) = mat.col(i).maxCoeff(&maxIndex[i]);
    idx[i] = maxIndex[i];
    cout << "maxVal: " << maxVal(i) << " index: " << idx[i] << endl;
  }
  for (int i = 0; i < idx.size(); ++i)
  cout << idx[i] << endl;
   */

  // /*
  Eigen::MatrixXf root_score_vectors(1, 3);
  root_score_vectors << 9, 10, 9;
  Eigen::MatrixXf result_matrix(3, 3); 
  //initialiseMatrix(result_matrix);
  result_matrix.fill(-9e+99);
  result_matrix(0, 1) = 20;
  result_matrix(0, 2) = 3;
  result_matrix(1, 0) = 30;
  result_matrix(1, 2) = 30;
  result_matrix(2, 0) = 11;
  result_matrix(2, 1) = 0;
  vector<int> result = CLE(root_score_vectors, result_matrix); 
  for (int i = 0; i < result.size(); ++i) {
    cout << result[i] <<endl;
  }
  // */
}
