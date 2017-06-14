//
// Created by Han Zhao on 11/19/15.
//

#include "src/SPNNode.h"
#include "src/SPNetwork.h"
#include "src/utils.h"
#include "src/BatchParamLearning.h"

#include <fstream>
#include <queue>
#include <random>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

using namespace std;
using SPN::SPNNode;
using SPN::SumNode;
using SPN::ProdNode;
using SPN::VarNode;
using SPN::SPNNodeType;
using SPN::SPNetwork;
using SPN::BatchParamLearning;
using SPN::ProjectedGD;
using SPN::ExpoGD;
using SPN::SMA;
using SPN::ExpectMax;
using SPN::CollapsedVB;
using SPN::LBFGS;
using SPN::utils::split;

std::vector<int> getLabels(const std::vector<std::vector<double>> &inputs) {

    int K = 10;
    int argmax = -1;
    int LABEL_VAR_START_DIM = 784;

    int num_inputs = inputs.size();
    std::vector<int> labels(num_inputs,-1);
    for (size_t n = 0; n < num_inputs; ++n) {
        for (int k=0; k < K; k++) {
            if (inputs[n][LABEL_VAR_START_DIM + k] == 1) {
                labels[n] = k;
            }

        }
            
    }

    return labels;
}

int main(int argc, char *argv[]) {
    // Positional program parameters.
    std::string model_filename, test_masks_filename, test_filename, algo_name;
    std::string output_model_filename;
    // Hyperparameters for projected gradient descent algorithm.
    uint seed = 42;
    int num_iters = 50;
    uint history_window = 5;
    double stop_thred = 1e-2;
    double lap_smooth = 1e-3;
    double proj_eps = 1e-2;
    double lrate = 1e-1;
    double shrink_weight = 8e-1;
    double prior_scale = 100.0;
    double train_fraction = 1.0;
    // Building command line parser
    po::options_description desc("Please specify the following options");
    desc.add_options()
            // Positional program parameters.
            ("masks", po::value<std::string>(&test_masks_filename), "file path of test masks data")
            ("test", po::value<std::string>(&test_filename), "file path of test data")
            ("model", po::value<std::string>(&model_filename), "file path of SPN")
            ("output_model", po::value<std::string>(&output_model_filename), "file path of SPN to save")
            ("algo", po::value<std::string>(&algo_name), "batch algorithm")
            ("train_fraction", po::value<double>(&train_fraction), "fraction of training data")
            // Hyperparameters for training algorithms.
            ("seed", po::value<uint>(&seed), "random seed")
            ("num_iters", po::value<int>(&num_iters), "maximum number of iterations")
            ("stop_thred", po::value<double>(&stop_thred), "stop criterion for consecutive function values")
            ("proj_eps", po::value<double>(&proj_eps), "projection constant for ProjectedGD algorithm")
            ("shrink_weight", po::value<double>(&shrink_weight), "shrinking weight during line search")
            ("lrate", po::value<double>(&lrate), "learning rate")
            ("lap_smooth", po::value<double>(&lap_smooth), "smoothing parameter")
            ("prior_scale", po::value<double>(&prior_scale), "scale parameter the prior distritbuion");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (!vm.count("masks") || !vm.count("test") ||
        !vm.count("model") || !vm.count("algo")) {
        std::cout << desc << std::endl;
        return -1;
    }
    // Load training and test data sets

    std::vector<std::vector<double>> test_data = SPN::utils::load_data(test_filename);

    // Load masks for test data

    std::vector<std::vector<bool>> test_data_masks = SPN::utils::load_masks(test_masks_filename);

    std::cout << "Loaded model: " << model_filename << std::endl;
    std::cout << "Number of instances in test set = " << test_data.size() << std::endl;
    std::cout << "Number of instances in masks set = " << test_data_masks.size() << std::endl;

    std::cout << "Dim of test set = " << test_data[0].size() << std::endl;
    std::cout << "Dim of masks set = " << test_data_masks[0].size() << std::endl;

    size_t num_test_masks = test_data_masks.size(), num_test = test_data.size();
    if (test_data_masks[0].size() != test_data[0].size()) {
        std::cerr << "Test data and masks are not consistent in dimension" << std::endl;
        return -1;
    }

    

    // Load and simplify SPN
    SPNetwork *spn = SPN::utils::load(model_filename);
    spn->init();
    std::cout << "Network statistics after initialization: " << std::endl;
    cout << "Network height: " << spn->height() << endl;
    cout << "Network size: " << spn->size() << endl;
    cout << "Network number of nodes: " << spn->num_nodes() << endl;
    cout << "Network number of edges: " << spn->num_edges() << endl;
    cout << "Network number of varnodes: " << spn->num_var_nodes() << endl;
    cout << "Network number of sumnodes: " << spn->num_sum_nodes() << endl;
    cout << "Network number of prodnodes: " << spn->num_prod_nodes() << endl;
    cout << "**********************************" << endl;

    cout << "Classify test set ... " << endl;
    vector<int> labels = getLabels(test_data);
    
    // std::vector<bool> mask_false(training_data[0].size(), false);
    std::clock_t tt_start = std::clock();
    vector<int> preds = spn->batchClassify(test_data,test_data_masks,true);
    cout << "Done ... " << endl;
    std::clock_t tt_end = std::clock();


    std::cout << "size "<< preds.size() << ", CPU time = " << 1000.0 * (tt_end - tt_start) / CLOCKS_PER_SEC << " milliseconds\n";
    double count = 0.0;
    for (int i=0 ; i<labels.size() ; i++){
        std::cout << "labels["<< i <<"] = "<< labels[i] << ", preds["<< i <<"] = " << preds[i] << std::endl;
        if (labels[i] == preds[i]) {
            count += 1;
        }
    }
    std::cout << "accuracy = "<< (count / labels.size()) << std::endl;

    delete spn;
    return 0;
}
