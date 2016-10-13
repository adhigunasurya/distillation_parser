# Graph-Based Distillation Parser

The code for a graph-based parser with discriminative training that distills an ensemble of greedy dependency parsers into a single parser. [Paper](https://arxiv.org/abs/1609.07561) pubslihed at EMNLP 2016 as a long paper.

#### Required software

 * A C++ compiler supporting the [C++11 language standard](https://en.wikipedia.org/wiki/C%2B%2B11)
 * [Boost](http://www.boost.org/) libraries
 * [Eigen](http://eigen.tuxfamily.org) (newer versions strongly recommended)
 * [CMake](http://www.cmake.org/)
 * [gcc](https://gcc.gnu.org/gcc-5/) (only tested with gcc version 5.3.0, may be incompatible with earlier versions)

#### Build instructions

    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen
    make -j2

# Code

The code and full instructions will be released before EMNLP 2016

# Citations

If you use the code, please cite the following work:

@inproceedings{kuncoro:2016emnlp,
  author={Adhiguna Kuncoro and Miguel Ballesteros and Lingpeng Kong and Chris Dyer and Noah A. Smith},
  title={Distilling an Ensemble of Greedy Dependency Parsers into One MST Parser},
  booktitle={Proc. EMNLP},
  year=2016,
} 

# Contact
For any questions or issues please e-mail adhiguna.kuncoro [ AT SYMBOL ] gmail [ DOT ] com
