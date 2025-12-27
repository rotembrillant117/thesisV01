import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import torch
import json
import random
from pathlib import Path
from thesis.core.stats import *
from thesis.utils.find_ff_all_words_all_languages import get_same_words_across_languages
from thesis.core.experiment import Experiment

from thesis.tokenizers.sage import MySageTokenizer
from thesis.tokenizers.hf import HFTokenizer
from thesis.utils.log_utils import setup_logger
from thesis.utils.analysis_utils import get_categories, get_ex_couple, compare_trials



def parse_args(path):
    """
    Reads the .json file
    :param path: path to json file
    :return: data
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data


PROJECT_ROOT = Path(__file__).resolve().parent


def fix_path(path):
    # If 'path' is absolute, the / operator ignores the left side (PROJECT_ROOT)
    # If 'path' is relative, it joins them.
    return str((PROJECT_ROOT / path.strip()).resolve())


def create_multi_text_file(path1, path2, file_name, num_rows=300_000, seed=42):
    """
    Creates a .txt file that combines two different text files by randomly sampling half of the lines
    from each input file using a specific random seed.

    :param path1: Path to file of first language
    :param path2: Path to file of second language
    :param file_name: Name of the combined output file
    :param num_rows: Total number of rows in the output file (half from each input)
    :param seed: Random seed for reproducibility
    """
    rows_from_each = num_rows // 2
    
    with open(path1, 'r', encoding='utf-8') as f1:
        lines1 = f1.readlines()
    with open(path2, 'r', encoding='utf-8') as f2:
        lines2 = f2.readlines()
    
    random.seed(seed)
    sampled1 = random.sample(lines1, rows_from_each)
    random.seed(seed + 1)
    sampled2 = random.sample(lines2, rows_from_each)
    
    with open(file_name, 'w', encoding='utf-8') as f_out:
        f_out.writelines(sampled1 + sampled2)


def init_experiments(data, l1_tokenizers):
    """
    This function instantiates the Experiment objects
    :param data: the .json data
    :param l1_tokenizers: dictionary of l1 tokenizers {algo: Tokenizer}
    :return: dictionary of {l2: [Experiment1, Experiment2...]}
    """
    experiments = dict()
    l1_data = data["l1"]
    l1 = l1_data["language"]
    l1_training_corpus_dir = fix_path(l1_data["training_data"])
    l1_words_dir = fix_path(l1_data["words"])
    l2_experiments = data["l2"]
    for l2_data in l2_experiments:
        l2 = l2_data["language"]
        experiments[l2] = []
        l2_training_corpus_dir = fix_path(l2_data["training_data"])
        l2_words_dir = fix_path(l2_data["words"])
        ff_words_path = fix_path(l2_data["ff"])
        l1_l2_training_corpus_dir = fix_path(f"./data/raw/training_data/{l2}/{l1}_{l2}_corpus.txt")
        create_multi_text_file(l1_training_corpus_dir, l2_training_corpus_dir, l1_l2_training_corpus_dir)
        for algo in data["algos"]:
            l1_tokenizer = l1_tokenizers[algo]
            if "SAGE" in algo:
                cur_exp = Experiment(l1, l2, l1_training_corpus_dir, l2_training_corpus_dir, l1_words_dir, l2_words_dir,
                                     l1_l2_training_corpus_dir, algo, data["vocab_size"], ff_words_path, l1_tokenizer,
                                     embedding_schedule=data["embedding_schedule"], full_vocab_schedule=data["full_vocab_schedule"])
            else:
                cur_exp = Experiment(l1, l2, l1_training_corpus_dir, l2_training_corpus_dir, l1_words_dir, l2_words_dir,
                                     l1_l2_training_corpus_dir, algo, data["vocab_size"], ff_words_path, l1_tokenizer)
            experiments[l2].append(cur_exp)
    
    return experiments


def train_l1_tokenizers(data):
    """
    This function trains the l1 (English) tokenizers
    :param data: data from the .json file
    :return: dictionary {algo: Tokenizer}
    """
    algos = data['algos']
    vocab_size = data['vocab_size']
    full_vocab_schedule = data['full_vocab_schedule']
    embedding_schedule = data['embedding_schedule']
    l1_data = data['l1']
    l1 = l1_data["language"]
    l1_tokenizers = dict()
    training_corpus_dir = fix_path(l1_data['training_data'])
    for algo in algos:
        if "SAGE" in algo:
            dir = fix_path(f"./outputs/results/{l1}_{algo}_{vocab_size}")
            os.makedirs(dir, exist_ok=True)
            logger.info(f"created directory {dir}")
            tokenizer = MySageTokenizer(l1_data["language"], training_corpus_dir, vocab_size, algo, embedding_schedule,
                                        full_vocab_schedule)
        else:
            tokenizer = HFTokenizer(l1_data["language"], training_corpus_dir, vocab_size, algo)
        tokenizer.train_tokenizer()
        l1_tokenizers[algo] = tokenizer
    return l1_tokenizers


def start_experiments(experiments):
    """
    Start the experiments
    :param experiments: dictionary of Experiment objects
    :return:
    """
    for l2, exp_list in experiments.items():
        for exp in exp_list:
            exp.start_experiment()


def save_experiments(experiments):
    """
    This functon saves the Experiment objects
    :param experiments: dictionary of Experiment objects
    :return:
    """
    for l2, exp_list in experiments.items():
        for exp in exp_list:
            exp.save_experiment()


def load_experiments(path):
    """
    This function loads the Experiment objects
    :param path: path to experiment objects
    :return: dictionary of {l2: [list of Experiment objects]}
    """
    experiments = dict()
    for l2 in os.listdir(path):
        experiments[l2] = []
        for pickle_file in os.listdir(f"{path}/{l2}"):
            experiments[l2].append(Experiment.load_experiment(f"{path}/{l2}/{pickle_file}"))
    return experiments


if __name__ == '__main__':
    logger = setup_logger("Main")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    train, args_path = sys.argv[1:]
    logger.info(sys.argv[1:])
    data = parse_args(fix_path(args_path))
    if train == "True":
        l1_tokenizers = train_l1_tokenizers(data)
        logger.info("Finished training l1 tokenizers")
        experiments = init_experiments(data, l1_tokenizers)
        logger.info("Finished creating experiments")
        start_experiments(experiments)
        logger.info("Finished starting experiments")
        save_experiments(experiments)
        logger.info("Finished saving experiments")
    else:
        vocab_size = str(data["vocab_size"])
        experiments = load_experiments(fix_path(f"./outputs/experiments/{vocab_size}"))
        logger.info("Finished loading experiments")
    
    for l2, exp_list in experiments.items():
        for ex in exp_list:
            graphs_path = f"{ex.analysis_dir}/graphs"
            tokenization_path = f"{ex.analysis_dir}/tokenization"
            categories = get_categories(ex)
            ff_tokenization_cases = analyze_tokenization(ex.get_tokenizers_list(), ex.get_ff_words(), ex.l1, ex.l2,
                                                         categories)
            homographs = get_same_words_across_languages(ex.l1, ex.l2)
            homographs_tokenization_cases = analyze_tokenization(ex.get_tokenizers_list(), homographs, ex.l1, ex.l2,
                                                                 categories)
            # ff_intrinsic_analysis(ex.l1, ex.l2, ex)
            plot_tokenization_cases(ff_tokenization_cases, ex.algo_name, ex.l1, ex.l2, categories, "ff",
                                    graphs_path)
            write_tokenization_split(ex.get_tokenizers_list(), ex.get_ff_words(), ex.l1, ex.l2, ex.algo_name,
                                     tokenization_path)
            plot_average_word_length(ff_tokenization_cases, ex.algo_name, graphs_path, ex.l1, ex.l2, categories)
            plot_average_num_tokens(ex.get_tokenizers_list(), ff_tokenization_cases, ex.algo_name, graphs_path,
                                    ex.l1, ex.l2, categories)
            plot_frequency_comparison(ff_tokenization_cases, ex.algo_name, graphs_path, ex.l1, ex.l2,
                                      ex.get_corpus_words(ex.l1), ex.get_corpus_words(ex.l2), categories)
            plot_pos_data(ff_tokenization_cases, ex.l1, ex.l2, ex.l1_training_corpus_dir, ex.l2_training_corpus_dir,categories, ex.algo_name, graphs_path)
            # chi_square_test(ff_tokenization_cases, homographs_tokenization_cases, ex.l1, ex.l2, ex.algo_name)
            # print("#########################################################################################################")
        compare_trials(exp_list, "BPE", "different_splits")
        compare_trials(exp_list, "UNI", "different_splits")






