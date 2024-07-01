import argparse
import numpy as np
import os
import pickle


# Calculates forgetting statistics per example
#
# diag_stats: dictionary created during training containing 
#             loss, accuracy, and missclassification margin 
#             per example presentation
# npresentations: number of training epochs
#
# Returns 4 dictionaries with statistics per example
#
# def compute_forgetting_statistics(diag_stats, npresentations):
#     presentations_needed_to_learn = {}
#     unlearned_per_presentation = {}
#     margins_per_presentation = {}
#     first_learned = {}
#
#     for example_id, example_stats in diag_stats.items():
#         # Skip 'train' and 'test' keys of diag_stats
#         if not isinstance(example_id, str):
#             # Forgetting event is a transition in accuracy from 1 to 0
#             presentation_acc = np.array(example_stats[1][:npresentations])
#             transitions = presentation_acc[1:] - presentation_acc[:-1]
#
#             # Find all presentations when forgetting occurs
#             if len(np.where(transitions == -1)[0]) > 0:
#                 unlearned_per_presentation[example_id] = np.where(transitions == -1)[0] + 2
#             else:
#                 unlearned_per_presentation[example_id] = []
#             print(
#                 f"Debug: Example ID {example_id}, Transitions: {transitions}, Unlearned Times: {unlearned_per_presentation[example_id]}")
#
#             # Find number of presentations needed to learn example
#             if len(np.where(presentation_acc == 0)[0]) > 0:
#                 presentations_needed_to_learn[example_id] = np.where(presentation_acc == 0)[0][-1] + 1
#             else:
#                 presentations_needed_to_learn[example_id] = 0
#
#             # Find the misclassification margin for each presentation of the example
#             margins_per_presentation[example_id] = np.array(example_stats[2][:npresentations])
#
#             # Find the presentation at which the example was first learned
#             if len(np.where(presentation_acc == 1)[0]) > 0:
#                 first_learned[example_id] = np.where(presentation_acc == 1)[0][0]
#             else:
#                 first_learned[example_id] = np.nan
#
#             # 调试打印
#             print(f"Example {example_id} stats:")
#             print(f"  Presentation Accuracy: {presentation_acc}")
#             print(f"  Transitions: {transitions}")
#             print(f"  Unlearned Presentations: {unlearned_per_presentation[example_id]}")
#             print(f"  Presentations Needed to Learn: {presentations_needed_to_learn[example_id]}")
#             print(f"  First Learned Presentation: {first_learned[example_id]}")
#     print("Debugging unlearned_per_presentation dictionary:")
#     for example_id, unlearned_times in list(unlearned_per_presentation.items())[:10]:
#         print(f"Example ID: {example_id}, Unlearned Times: {unlearned_times}")
#
#     return presentations_needed_to_learn, unlearned_per_presentation, margins_per_presentation, first_learned

def compute_forgetting_statistics(diag_stats, npresentations):
    presentations_needed_to_learn = {}
    unlearned_per_presentation = {}
    margins_per_presentation = {}
    first_learned = {}

    for example_id, example_stats in diag_stats.items():
        # Extract accuracies for each presentation
        presentation_acc = [int(acc) for (_, acc) in example_stats[:npresentations]]  # 转换为整数

        # Compute transitions (1 to 0 indicates forgetting)
        transitions = [presentation_acc[i + 1] - presentation_acc[i] for i in range(len(presentation_acc) - 1)]

        # Detect forgetting events
        forgetting_events = [i + 2 for i, t in enumerate(transitions) if t == -1]
        unlearned_per_presentation[example_id] = forgetting_events

        # Detect learning events
        learning_events = [i for i, acc in enumerate(presentation_acc) if acc == 1]
        first_learned[example_id] = learning_events[0] if learning_events else None

    # Print debugging information for the first five samples
    for i, (example_id, example_stats) in enumerate(diag_stats.items()):
        if i >= 5:
            break
        presentation_acc = [int(acc) for (_, acc) in example_stats[:npresentations]]
        transitions = [presentation_acc[i + 1] - presentation_acc[i] for i in range(len(presentation_acc) - 1)]
        forgetting_events = [i + 2 for i, t in enumerate(transitions) if t == -1]
        learning_events = [i for i, acc in enumerate(presentation_acc) if acc == 1]
        first_learned_sample = learning_events[0] if learning_events else None

        print(f"Example {example_id}:")
        print(f"  Accuracies: {presentation_acc}")
        # print(f"  Transitions: {transitions}")
        # print(f"  Forgetting Events: {forgetting_events}")
        # print(f"  First Learned at Presentation: {first_learned_sample}")

    return presentations_needed_to_learn, unlearned_per_presentation, margins_per_presentation, first_learned

# def compute_forgetting_statistics(diag_stats, npresentations):
#     presentations_needed_to_learn = {}
#     unlearned_per_presentation = {}
#     margins_per_presentation = {}
#     first_learned = {}
#
#     for example_id, example_stats in diag_stats.items():
#         # Extract accuracies for each presentation
#         presentation_acc = [acc for (_, acc) in example_stats[:npresentations]]
#
#         # Compute transitions (1 to 0 indicates forgetting)
#         transitions = [presentation_acc[i+1] - presentation_acc[i] for i in range(len(presentation_acc)-1)]
#
#         # Detect forgetting events
#         forgetting_events = [i+2 for i, t in enumerate(transitions) if t == -1]
#         unlearned_per_presentation[example_id] = forgetting_events
#
#         # Detect learning events
#         learning_events = [i for i, acc in enumerate(presentation_acc) if acc == 1]
#         first_learned[example_id] = learning_events[0] if learning_events else None
#
#         # Print debugging information
#         # print(f"Example {example_id}:")
#         # print(f"  Accuracies: {presentation_acc}")
#         # print(f"  Transitions: {transitions}")
#         # print(f"  Forgetting Events: {forgetting_events}")
#         # print(f"  First Learned at Presentation: {first_learned[example_id]}")
#
#     return presentations_needed_to_learn, unlearned_per_presentation, margins_per_presentation, first_learned


# Sorts examples by number of forgetting counts during training, in ascending order
# If an example was never learned, it is assigned the maximum number of forgetting counts
# If multiple training runs used, sort examples by the sum of their forgetting counts over all runs
#
# unlearned_per_presentation_all: list of dictionaries, one per training run
# first_learned_all: list of dictionaries, one per training run
# npresentations: number of training epochs
#
def sort_examples_by_forgetting(unlearned_per_presentation_all, npresentations):
    print("Number of presentations:", npresentations)
    #print("Unlearned per presentation:", unlearned_per_presentation_all)
    example_original_order = []
    example_stats = []

    for example_id, stats in unlearned_per_presentation_all.items():
        example_original_order.append(example_id)
        if len(stats) > 0:
            example_stats.append(len(stats))
        else:
            example_stats.append(0)

    sorted_indices = np.argsort(example_stats)
    sorted_example_ids = np.array(example_original_order)[sorted_indices]
    sorted_stats = np.array(example_stats)[sorted_indices]

    # for idx, example_id in enumerate(sorted_example_ids):
    #     if sorted_stats[idx] > 0:
    #         print(f"Example ID: {example_id}, Forgetting Count: {sorted_stats[idx]}")

    return sorted_example_ids, sorted_stats


# Checks whether a given file name matches a list of specified arguments
#
# fname: string containing file name
# args_list: list of strings containing argument names and values, i.e. [arg1, val1, arg2, val2,..]
#
# Returns 1 if filename matches the filter specified by the argument list, 0 otherwise
#
def check_filename(fname, args_list):

    # If no arguments are specified to filter by, pass filename
    if args_list is None:
        return 1

    for arg_ind in np.arange(0, len(args_list), 2):
        arg = args_list[arg_ind]
        arg_value = args_list[arg_ind + 1]

        # Check if filename matches the current arg and arg value
        if arg + '_' + arg_value + '__' not in fname:
            print('skipping file: ' + fname)
            return 0

    return 1


# if __name__ == "__main__":
#
#     parser = argparse.ArgumentParser(description="Options")
#     parser.add_argument('--input_dir', type=str, required=True)
#     parser.add_argument(
#         '--input_fname_args',
#         nargs='+',
#         help=
#         'arguments and argument values to select input filenames, i.e. arg1 val1 arg2 val2'
#     )
#     parser.add_argument('--output_dir', type=str, required=True)
#     parser.add_argument(
#         '--output_name',
#         type=str,
#         required=True)
#     parser.add_argument('--epochs', type=int, default=200)
#
#     args = parser.parse_args()
#     print(args)
#
#     # Initialize lists to collect forgetting stastics per example across multiple training runs
#     unlearned_per_presentation_all, first_learned_all = [], []
#
#     for d, _, fs in os.walk(args.input_dir):
#         for f in fs:
#
#             # Find the files that match input_fname_args and compute forgetting statistics
#             if f.endswith('stats_dict.pkl') and check_filename(
#                     f, args.input_fname_args):
#                 print('including file: ' + f)
#
#                 # Load the dictionary compiled during training run
#                 with open(os.path.join(d, f), 'rb') as fin:
#                     loaded = pickle.load(fin)
#
#                 # Compute the forgetting statistics per example for training run
#                 _, unlearned_per_presentation, _, first_learned = compute_forgetting_statistics(
#                     loaded, args.epochs)
#
#                 unlearned_per_presentation_all.append(
#                     unlearned_per_presentation)
#                 first_learned_all.append(first_learned)
#
#     if len(unlearned_per_presentation_all) == 0:
#         print('No input files found in {} that match {}'.format(
#             args.input_dir, args.input_fname_args))
#     else:
#
#         # Sort examples by forgetting counts in ascending order, over one or more training runs
#         ordered_examples, ordered_values = sort_examples_by_forgetting(
#             unlearned_per_presentation_all, first_learned_all, args.epochs)
#
#         # Save sorted output
#         if args.output_name.endswith('.pkl'):
#             with open(os.path.join(args.output_dir, args.output_name),
#                       'wb') as fout:
#                 pickle.dump({
#                     'indices': ordered_examples,
#                     'forgetting counts': ordered_values
#                 }, fout)
#         else:
#             with open(
#                     os.path.join(args.output_dir, args.output_name + '.pkl'),
#                     'wb') as fout:
#                 pickle.dump({
#                     'indices': ordered_examples,
#                     'forgetting counts': ordered_values
#                 }, fout)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument(
        '--input_fname_args',
        nargs='+',
        help=
        'arguments and argument values to select input filenames, i.e. arg1 val1 arg2 val2'
    )
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument(
        '--output_name',
        type=str,
        required=True)
    parser.add_argument('--epochs', type=int, default=200)

    args = parser.parse_args()
    print(args)

    # Initialize lists to collect forgetting statistics per example across multiple training runs
    unlearned_per_presentation_all, first_learned_all = [], []

    for d, _, fs in os.walk(args.input_dir):
        for f in fs:
            # Find the files that match input_fname_args and compute forgetting statistics
            if f.endswith('stats_dict.pkl') and check_filename(f, args.input_fname_args):
                print('including file: ' + f)

                # Load the dictionary compiled during training run
                with open(os.path.join(d, f), 'rb') as fin:
                    loaded = pickle.load(fin)

                # Compute the forgetting statistics per example for training run
                _, unlearned_per_presentation, _, first_learned = compute_forgetting_statistics(
                    loaded, args.epochs)

                unlearned_per_presentation_all.append(unlearned_per_presentation)
                first_learned_all.append(first_learned)

    if len(unlearned_per_presentation_all) == 0:
        print('No input files found in {} that match {}'.format(args.input_dir, args.input_fname_args))
    else:
        # Sort examples by forgetting counts in ascending order, over one or more training runs
        ordered_examples, ordered_values = sort_examples_by_forgetting(
            unlearned_per_presentation_all, args.epochs)

        # Save sorted output
        if args.output_name.endswith('.pkl'):
            with open(os.path.join(args.output_dir, args.output_name), 'wb') as fout:
                pickle.dump({
                    'indices': ordered_examples,
                    'forgetting counts': ordered_values
                }, fout)
        else:
            with open(os.path.join(args.output_dir, args.output_name + '.pkl'), 'wb') as fout:
                pickle.dump({
                    'indices': ordered_examples,
                    'forgetting counts': ordered_values
                }, fout)
