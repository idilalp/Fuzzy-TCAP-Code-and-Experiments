import csv
from itertools import combinations
from collections import defaultdict

# ==============================================================
# Utility functions
# ==============================================================

def get_data_set(source_file_name, has_header):
    """
    Loads a CSV file into a list-of-lists structure.
    If 'has_header' is True, the first row is stored separately as the header.
    Returns [header, dataset]
    """
    header = []
    dataset = []
    with open(source_file_name, 'r', encoding='Latin1') as f:
        reader = csv.reader(f)
        if has_header:
            header = next(reader)
        for row in reader:
            if row:
                dataset.append(row)
    return [header, dataset]

def convert_list_to_csv(lis):
    """
    Joins a list of strings into a comma-separated single string.
    Used to create keys for equivalence class dictionaries.
    """
    return ",".join(str(x) for x in lis)

def get_equivalence_classes(dataset, k):
    """
    Forms equivalence classes based on the first k columns of the dataset.
    Returns a dictionary {key: count}, where key is a combination of k values.
    """
    eq_class_dict = {}
    for rec in dataset:
        key = convert_list_to_csv(rec[0:k])
        eq_class_dict[key] = eq_class_dict.get(key, 0) + 1
    return eq_class_dict

def get_univariate_frequency_table(dataset, column):
    """
    Returns frequency counts for a single column.
    Used to compute the baseline probability distribution for the target variable.
    """
    vals = []
    for r in dataset:
        try:
            vals.append(int(float(r[column])))
        except:
            continue
    if not vals:
        return [0]
    counts = [0] * (max(vals) + 1)
    for v in vals:
        counts[v] += 1
    return counts

# ==============================================================
# Stage 1: run TCAP per combination and gather per-record lists
# ==============================================================

def run_classic_tcap_per_record(source_file, synth_file, keys, target, tau=1.0, eqmax=None):
    """
    Runs classic TCAP logic for a given key combination.
    Returns:
    - CAP-undef, CAP-zero, baseline
    - per-record attribution values (None, 0, or proportion)
    - list of eligible record indices for later CAP-zero calculation
    """
    # Loading real and synthetic data
    header, real = get_data_set(source_file, True)
    _, synth = get_data_set(synth_file, True)

    key_indices = [header.index(k) for k in keys]
    target_index = header.index(target)

    # Creating key+target combinations from selected columns
    d1 = [[row[i] for i in key_indices] + [row[target_index]] for row in real]
    d2 = [[row[i] for i in key_indices] + [row[target_index]] for row in synth]

    nvars = len(d1[0])  # total number of variables (keys + target)
    eqks = get_equivalence_classes(d1, nvars - 1)  # key counts in real data
    eqk  = get_equivalence_classes(d2, nvars - 1)  # key counts in synth data
    eqkt = get_equivalence_classes(d2, nvars)      # key+target counts in synth data
    uvd  = get_univariate_frequency_table(d1, nvars - 1)  # frequency of each target value

    eqmaxcount = 0           # number of eligible records (filtered by eqmax if applicable)
    uvt = 0                  # cumulative baseline contribution (used for CAP baseline)
    matches = 0              # how many records found in synthetic key space
    dcaptotal = 0.0          # total CAP value for matched records
    per_record = [None] * len(d1)   # store CAP value per real record (None = ineligible)
    eligible_indices = []    # store indices of eligible records (needed for CAP-zero calculation when eqmax is set)

    for idx, rec in enumerate(d1):
        recordk  = convert_list_to_csv(rec[0:nvars - 1])  # key only
        recordkt = convert_list_to_csv(rec[0:nvars])      # key + target

        # Filter by eqmax if set: skip if equivalence class too large in real data
        if eqmax is not None and recordk in eqks and eqks[recordk] > eqmax:
            continue

        eqmaxcount += 1
        eligible_indices.append(idx)

        # Add baseline (uninformed guess probability)
        if sum(uvd) > 0:
            uvt += uvd[int(float(rec[nvars - 1]))] / sum(uvd)

        # If the record's key doesn't exist in synthetic, it's unmatched
        if recordk not in eqk:
            continue

        matches += 1
        prop = 0.0
        if recordkt in eqkt and eqk[recordk] > 0:
            prop = eqkt[recordkt] / eqk[recordk]  # conditional prob of target given key in synth

        # Assign per-record CAP (proportion if >= tau; otherwise 0)
        if prop >= tau:
            dcaptotal += prop
            per_record[idx] = prop
        else:
            per_record[idx] = 0

    # Calculate CAP metrics
    cap_undef = dcaptotal / matches if matches else 0
    cap_zero  = dcaptotal / eqmaxcount if eqmaxcount else 0
    baseline  = uvt / eqmaxcount if eqmaxcount else 0

    return cap_undef, cap_zero, baseline, per_record, eligible_indices

def stage1_gather_per_record_caps(source_file, synth_file, all_vars, target,
                                  tau=1.0, eqmax=None,
                                  min_depth=1, max_depth=None):
    """
    Iterates over all key combinations (from max_depth to min_depth).
    Stores per-record CAP lists and corresponding combinations for Stage 2.
    """
    if max_depth is None:
        max_depth = len(all_vars)

    stage1_lists = []  # list of per-record CAP values for each combination
    stage1_combos = [] # list of variable combinations used
    eligible_record_sets = []  # stores sets of eligible indices for CAP-zero

    print("\n=== Stage 1: Gathering per-record outputs ===")
    for k in range(max_depth, min_depth - 1, -1):
        for combo in combinations(all_vars, k):
            cap_undef, cap_zero, baseline, per_record, eligible_indices = run_classic_tcap_per_record(
                source_file, synth_file, list(combo), target, tau=tau, eqmax=eqmax
            )
            stage1_lists.append(per_record)
            stage1_combos.append(combo)
            eligible_record_sets.append(set(eligible_indices))
            print(f"[Stage1] keys={combo} | CAP undef={cap_undef:.4f} | CAP zero={cap_zero:.4f} | baseline={baseline:.4f}")
    return stage1_lists, stage1_combos, eligible_record_sets

# ==============================================================
# Stage 2: integrate using weights
# ==============================================================

def stage2_integrate_with_weights(stage1_lists, stage1_combos, eligible_record_sets, key_weights):
    """
    Implements weighted TCAP logic.
    Sorts combinations by total key weight, and selects the first non-None value per record.
    Also computes CAP-undef and CAP-zero based on final results.
    """
    weighted_meta = []
    for i in range(len(stage1_combos)):
        combo = stage1_combos[i]
        per_record = stage1_lists[i]
        weight_sum = sum(key_weights.get(k, 0) for k in combo)
        weighted_meta.append((combo, weight_sum, per_record, eligible_record_sets[i]))

    # Sorting combinations by total weight in descending order
    weighted_meta.sort(key=lambda x: x[1], reverse=True)

    n_records = len(weighted_meta[0][2])
    final_results = [None] * n_records
    chosen_combos = [None] * n_records

    # For each record, picking the first (for example, highest-weighted) combination with a defined value
    for rec_idx in range(n_records):
        for combo, weight_sum, per_record, _ in weighted_meta:
            if per_record[rec_idx] is not None:
                final_results[rec_idx] = per_record[rec_idx]
                chosen_combos[rec_idx] = combo
                break

    # Computing CAP-undef: average over matched (non-None) records
    non_none_vals = [v for v in final_results if v is not None]
    matches = len(non_none_vals)
    correct_sum = sum(non_none_vals)
    correct_count = sum(1 for v in non_none_vals if v > 0)
    cap_undef = correct_sum / matches if matches else 0

    # CAP-zero: averaging over all eligible records (union of eligible indices)
    eligible_union = set.union(*eligible_record_sets) if eligible_record_sets else set()
    cap_zero_vals = [final_results[i] if final_results[i] is not None else 0 for i in eligible_union]
    cap_zero = sum(cap_zero_vals) / len(eligible_union) if eligible_union else 0

    # Output summary
    print("\n=== Stage 2 Integrated Results ===")
    print(f"The target dataset CAP score with non-matches undefined is : {cap_undef:.4f}")
    print(f"The target dataset CAP score with non-matches scored as zero is: {cap_zero:.4f}")
    print(f"Matches = {matches}")
    print(f"Correct matches (count) = {correct_count}")
    print(f"(Sum of proportions used internally = {correct_sum:.4f}")

    # Tracking usage of combinations
    combo_match_counts = {}
    combo_correct_counts = {}
    for val, combo in zip(final_results, chosen_combos):
        if combo is not None:
            combo_match_counts[combo] = combo_match_counts.get(combo, 0) + 1
            if val and val > 0:
                combo_correct_counts[combo] = combo_correct_counts.get(combo, 0) + 1

    print("\n=== Top 10 combos used ===")
    for combo, count in sorted(combo_match_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{combo}: matches={count}, correct_matches={combo_correct_counts.get(combo, 0)}")

    return final_results, cap_undef, cap_zero, matches, correct_count, chosen_combos

# ==============================================================
# Variable risk contribution analysis
# ==============================================================

def compute_variable_risk_contributions(final_results, chosen_combos, all_vars):
    """
    Computes how much each variable contributed to final attribution risk.
    Uses counts and average CAP to build a per-variable risk score.
    """
    var_counts = defaultdict(int)
    var_risk_sums = defaultdict(float)
    n = len(final_results)

    for cap_val, combo in zip(final_results, chosen_combos):
        if combo is None or cap_val is None:
            continue
        for var in combo:
            var_counts[var] += 1
            var_risk_sums[var] += cap_val

    print("\n=== Variable-Level Risk Contribution Scores ===")
    for var in all_vars:
        v = var_counts[var]
        avg_cap = var_risk_sums[var] / v if v > 0 else 0
        score = (avg_cap * v) / n  # average contribution scaled by frequency
        print(f"{var}: Count={v}, Avg CAP={avg_cap:.4f}, Risk Contribution Score={score:.4f}")

# ==============================================================
# Example Run
# ==============================================================

if __name__ == "__main__":
    # Define all potential key variables and target
    all_vars = ['age', 'sex', 'marstat', 'minority', 'empstat', 'birthplace']
    target = 'relig'

    # Set up variable weights (used in Stage 2)
    key_weights = {
        'age': 1,
        'sex': 1,
        'minority': 1,
        'marstat': 1,
        'empstat': 1,
        'birthplace': 1
    }

    # File paths for real and synthetic datasets
    source_file = "/Users/idilalp/Desktop/canada2011_tcap_real_relig.csv"
    synth_file = "/Users/idilalp/Desktop/canada2011_tcap_synth_relig.csv"

    # Computing baseline from univariate distribution
    header, real_data = get_data_set(source_file, True)
    target_index = header.index(target)
    uvd = get_univariate_frequency_table(real_data, target_index)
    baseline = sum((count / sum(uvd))**2 for count in uvd if sum(uvd) > 0)
    print(f"\n=== Baseline (from source target distribution) ===")
    print(f"Baseline = {baseline:.4f}")

    # Stage 1: gathering all per-record CAPs for different key sets
    stage1_lists, stage1_combos, eligible_sets = stage1_gather_per_record_caps(
        source_file, synth_file, all_vars, target,
        tau=1.0,
        eqmax=None,
        min_depth=6,
        max_depth=6
    )

    # Stage 2: integrating using weighted logic
    final_results, cap_undef, cap_zero, matches, correct_count, chosen_combos = stage2_integrate_with_weights(
        stage1_lists, stage1_combos, eligible_sets, key_weights
    )

    print(f"\nFinal overall CAP (undef): {cap_undef:.4f}")
    print(f"Final overall CAP (zero):  {cap_zero:.4f}")
    print(f"Total matches: {matches}, Correct matches: {correct_count}, Baseline: {baseline:.4f}")

    # Computing variable-level risk scores
    compute_variable_risk_contributions(final_results, chosen_combos, all_vars)
