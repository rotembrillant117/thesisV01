
from thesis.core.stats import (
    analyze_tokenization,
    earth_movers_dist,
    get_avg_chars_per_token,
    get_token_length_distribution,
    words_moved_to_target,
    words_removed_from_target,
    words_moved_to_target_ff
)
from thesis.utils.find_ff_all_words_all_languages import get_same_words_across_languages

def get_categories(experiment):
    """
    This functon returns the category of a specific experiment
    :param experiment: an Experiment object
    :return: list of tokenization categories
    """
    l1 = experiment.l1
    l2 = experiment.l2
    categories = [f"{l1}_t==multi_t", f"{l2}_t==multi_t", f"{l1}_t=={l2}_t", "same_splits", "different_splits"]
    return categories

def get_ex_couple(exp_list, algo):
    """
    This function returns from the Experiment list of the same language (exp_list) 2 different Experiments:
    either BPE and BPE_SAGE or UNI and UNI_SAGE, depending on the algo chosen
    :param exp_list: a list of experiments of the same language
    :param algo: BPE or UNI
    :return: BPE and BPE_SAGE or UNI and UNI_SAGE Experiments
    """
    reg_ex, sage_ex = None, None
    for ex in exp_list:
        if ex.algo_name == algo:
            reg_ex = ex
        elif ex.algo_name == f"{algo}_SAGE":
            sage_ex = ex
    return reg_ex, sage_ex


def compare_trials(exp_list, baseline_algo, track_target):
    """
    This function compares between BPE vs BPE_SAGE and UNI vs UNI_SAGE, and saves a file with the results
    :param exp_list: a list of experiments of the same language
    :param baseline_algo: BPE or UNI
    :param track_target: which target to track from tokenization category
    :return:
    """
    ex_reg, ex_sage = get_ex_couple(exp_list, baseline_algo)
    path = f"{ex_reg.analysis_dir}/tokenization/{baseline_algo}_baseline_vs_SAGE.txt"
    categories = get_categories(ex_reg)
    homographs = get_same_words_across_languages(ex_reg.l1, ex_reg.l2)
    tokenization_cases = analyze_tokenization(ex_reg.get_tokenizers_list(), homographs, ex_reg.l1, ex_reg.l2, categories)
    sage_tokenization_cases = analyze_tokenization(ex_sage.get_tokenizers_list(), homographs, ex_sage.l1, ex_sage.l2, categories)
    # Tokenization Cases
    distribution1 = {c: len(tokenization_cases[c]) for c in categories}
    sage_distributions = {c: len(sage_tokenization_cases[c]) for c in categories}
    # Earth Movers Distance
    emd, moved = earth_movers_dist(categories, ex_reg.l1, ex_reg.l2, distribution1, sage_distributions, track_target)
    # Percent moved to tracked target
    total_mass_of_target = sum([moved[c] for c in categories])
    moved_normalized = {c: moved[c]/total_mass_of_target for c in categories}
    # Character Per Token
    cpt1, cpt2, cpt3 = get_avg_chars_per_token(ex_reg.l1_tokenizer), get_avg_chars_per_token(ex_reg.l2_tokenizer), get_avg_chars_per_token(ex_reg.l1_l2_tokenizer)
    sage_cpt1, sage_cpt2, sage_cpt3 = get_avg_chars_per_token(ex_sage.l1_tokenizer), get_avg_chars_per_token(ex_sage.l2_tokenizer), get_avg_chars_per_token(ex_sage.l1_l2_tokenizer)
    # Token Length distribution
    dis1, dis2, dis3 = get_token_length_distribution(ex_reg.l1_tokenizer), get_token_length_distribution(ex_reg.l2_tokenizer), get_token_length_distribution(ex_reg.l1_l2_tokenizer)
    sage_dis1, sage_dis2, sage_dis3 =  get_token_length_distribution(ex_sage.l1_tokenizer), get_token_length_distribution(ex_sage.l2_tokenizer), get_token_length_distribution(ex_sage.l1_l2_tokenizer)
    # Homographs added and removed from target
    added  = words_moved_to_target(tokenization_cases, sage_tokenization_cases, categories, track_target)
    removed = words_removed_from_target(tokenization_cases, sage_tokenization_cases, categories, track_target)
    # False Friends added to target
    added_ff = words_moved_to_target_ff(tokenization_cases, sage_tokenization_cases, ex_sage.get_ff_words(), categories, track_target)
    with open(path, "w", encoding="utf-8") as f:
        title = (f"Homographs across languages {ex_reg.l1} and {ex_reg.l2} - Baseline tokenizer: {baseline_algo}\n"
                 f"Difference between experiment {ex_reg.l1}_{ex_reg.algo_name}, {ex_reg.l2}_{ex_reg.algo_name}, multilingual_{ex_reg.algo_name} AND experiment "
                          f"{ex_sage.l1}_{ex_sage.algo_name}, {ex_sage.l2}_{ex_sage.algo_name}, multilingual_{ex_sage.algo_name}\n")
        distributions = f"{ex_reg.algo_name}: {distribution1}\n{ex_sage.algo_name}: {sage_distributions}\n"
        earth_movers = f"Earth Movers Distance: {emd}\nMass moved to {track_target}: {moved}\n"
        earth_movers2_percent = f"Earth Movers Distance - Normalized: {track_target}: {moved_normalized}\n"
        cpt_line1 = f"Average Chars Per Token {ex_reg.algo_name}: {ex_reg.l1}_tokenizer:{cpt1}, {ex_reg.l2}_tokenizer: {cpt2}, multilingual_tokenizer: {cpt3}\n"
        cpt_line2 = f"Average Chars Per Token {ex_sage.algo_name}: {ex_sage.l1}_tokenizer:{sage_cpt1}, {ex_sage.l2}_tokenizer: {sage_cpt2}, multilingual_tokenizer: {sage_cpt3}\n"
        token_dis_line1 = f"Token Length Distribution {ex_reg.algo_name}: {ex_reg.l1}_tokenizer:{dis1}, {ex_reg.l2}_tokenizer: {dis2}, multilingual_tokenizer: {dis3}\n"
        token_dis_line2 = f"Token Length Distribution {ex_sage.algo_name}: {ex_sage.l1}_tokenizer:{sage_dis1}, {ex_sage.l2}_tokenizer: {sage_dis2}, multilingual_tokenizer: {sage_dis3}\n"
        f.write(title)
        f.write(distributions)
        f.write(earth_movers)
        f.write(earth_movers2_percent)
        f.write(cpt_line1)
        f.write(cpt_line2)
        f.write(token_dis_line1)
        f.write(token_dis_line2)
        f.write(f"False Friends moved from source distribution to {track_target} in target distribution:\n")
        for c, words in added_ff.items():
            f.write(f"{c}: {words}\n")
        for c, words in added.items():
            f.write(f"Words added from source {c} to target {track_target}: {len(words)}\n")
        f.write("\n")
        for c, words in removed.items():
            f.write(f"Words removed from source {track_target} to target {c}: {len(words)}\n")
        f.write("\n")
        for c, words in added.items():
            f.write(f"Words added from source {c} to target {track_target}\n")
            for w in words:
                f.write(f'{w}\n')
            f.write("###################################################################################################################################################\n")
        f.write("\n")
        for c, words in removed.items():
            f.write(f"Words removed from source {track_target} to target {c}\n")
            for w in words:
                f.write(f'{w}\n')
            f.write("###################################################################################################################################################\n")
