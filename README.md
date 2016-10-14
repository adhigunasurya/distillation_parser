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

# Sample input file

The sample input file for English PTB-SD is at

`sample_input/sample_input_en.txt`

# Code

### Ensemble votes
The stack LSTM ensemble votes (used to compute the distillation cost) are provided by the following files:    
`costs/matrices_PTB_SD.txt` (English)     
`costs/matrices_chinese.txt` (Chinese)     

The English ensemble votes were computed with 21 models, while the Chinese ensemble votes were computed with 17. 

### Training the distillation parser (English)
Assuming current directory is `build/`      
`nohup graph-parse/graph-parse-new-cost-mbr --cnn_mem 1700 -t [train.conll]  -d [dev.conll] -P --pretrained_dim [pretrained word embedding dimension] -w [pretrained word embedding file] --cost_matrix ../costs/matrices_PTB_SD.txt --eta_decay 0.05 --num_ensemble 21 -x > log_en.txt`      

For the best results train the model for 3-4 days.

### Training the distillation parser (Chinese)

`nohup graph-parse/graph-parse-new-cost-mbr --cnn_mem 1700 -t [chinese_train.conll]  -d [chinese_dev.conll] -P --pretrained_dim [pretrained word embedding dimension] -w [pretrained word embedding file] --cost_matrix ../costs/matrices_chinese.txt --eta_decay 0.05 --num_ensemble 17 -x > log_chinese.txt`     

For the best results train the model for 3-4 days.

### Parameter files

To find out where the parameter file is saved to, look at the `log_en.txt` and `log_chinese.txt`       

### Decoding with the distillation parser (English and Chinese)
`nohup graph-parse/graph-parse-new-cost-mbr --cnn_mem 1700 -t [train.conll]  -p [test.conll] -P --pretrained_dim [pretrained word embedding dimension] -w [pretrained word embedding file] -m [parameter file (.params extension)] > output.txt`     

### Evaluation (without punctuations)
`perl eval.pl -s output.txt -g [test.conll] -q`     

German results were reported with punctuations and used a different evaluation script    

# Citations

If you use the code, please cite the following work:

@inproceedings{kuncoro:2016emnlp,
  author={Adhiguna Kuncoro and Miguel Ballesteros and Lingpeng Kong and Chris Dyer and Noah A. Smith},
  title={Distilling an Ensemble of Greedy Dependency Parsers into One MST Parser},
  booktitle={Proc. EMNLP},
  year=2016,
} 

# Contact
For any questions, issues, or reproducing the German results, please e-mail adhiguna.kuncoro [ AT SYMBOL ] gmail [ DOT ] com
