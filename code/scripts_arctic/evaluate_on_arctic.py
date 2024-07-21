import numpy as np
import json
import zipfile
import tempfile
import torch
import os
import argparse
import os.path as op
from tqdm import tqdm
import sys
sys.path = [".", '..'] + sys.path
import src.utils.io.gt_arctic as gt
import src.utils.eval_modules_arctic as eval_m

eval_fn_dict = {
    "mpjpe_ra_r": eval_m.eval_mpjpe_right,
    "mpjpe_ra_l": eval_m.eval_mpjpe_left,
    "mpjpe_ra_h": eval_m.eval_mpjpe_hand,
    "cd_f_r": eval_m.eval_cd_f_right_arctic,
    "cd_f_l": eval_m.eval_cd_f_left_arctic,
    "cd_h": eval_m.eval_cd_f_hand_arctic,
}


def eval_seq(pred_p, output_dir):
    print("Evaluating: ", pred_p)
    data_pred = torch.load(pred_p).to_std_precision()
    data_gt = gt.load_data(data_pred["full_seq_name"])
    seq_name = data_pred["full_seq_name"]
    out_p = op.join(output_dir, seq_name)
    os.makedirs(out_p, exist_ok=True)
    eval_fn_dict["icp"] = eval_m.eval_icp_first_frame_arctic

    print("------------------")
    print("Involving the following eval_fn:")
    for eval_fn_name in eval_fn_dict.keys():
        print(eval_fn_name)
    print("------------------")

    # Initialize the metrics dictionaries
    metric_dict = {}
    # Evaluate each metric using the corresponding function
    pbar = tqdm(eval_fn_dict.items())
    for eval_fn_name, eval_fn in pbar:
        pbar.set_description(f"Evaluating {eval_fn_name}")
        metric_dict = eval_fn(data_pred, data_gt, metric_dict)

    # Dictionary to store mean values of metrics
    mean_metrics = {}

    # Print out the mean of each metric and store the results
    for metric_name, values in metric_dict.items():
        mean_value = float(
            np.nanmean(values)
        )  # Convert mean value to native Python float
        mean_metrics[metric_name] = mean_value

    # sort by key
    mean_metrics = dict(sorted(mean_metrics.items(), key=lambda item: item[0]))

    for metric_name, mean_value in mean_metrics.items():
        print(f"{metric_name.upper()}: {mean_value:.2f}")

    # Define the file paths
    json_path = out_p + ".metric.json"
    npy_path = out_p + ".metric_all.npy"

    from datetime import datetime

    current_time = datetime.now()
    time_str = current_time.strftime("%m-%d %H:%M")
    mean_metrics["timestamp"] = time_str
    mean_metrics["seq_name"] = seq_name
    print("Units: CD (cm), F-score (percentage), MPJPE (mm)")
    
    with open(json_path, "w") as f:
        json.dump(mean_metrics, f, indent=4)
        print(f"Saved mean metrics to {json_path}")

    # Save the metric_all numpy array
    np.save(npy_path, metric_dict)
    print(f"Saved metric_all numpy array to {npy_path}")

# sequences to evaluate
test_seqs = [
    'arctic_s05_espressomachine_grab_01_8'
]


def main(args):
    input_zip_p = args.zip_p
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Extract the zip file to the temporary directory
        with zipfile.ZipFile(input_zip_p, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)

        # Get the list of file paths within the temporary directory
        file_paths = []
        for root, dirs, files in os.walk(tmp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

        # Call the function and print the list of file paths
        pred_ps = file_paths
        pred_seqs = [op.basename(pred_p).replace('.pt', '') for pred_p in pred_ps]
        assert set(pred_seqs) == set(test_seqs), "Mismatch between test sequences and predictions."
        for idx, pred_p in enumerate(pred_ps):
            print("[%d/%d]" % (idx + 1, len(pred_ps)))
            eval_seq(pred_p, args.output)
        
        results = []
        for seq_name in test_seqs:
            #with open(f"./arctic_results/{seq_name}.metric.json", "r") as f:
            with open(op.join(args.output, f"{seq_name}.metric.json"), "r") as f:
                metric_dict = json.load(f)
            results.append(metric_dict)
            
        # average results
        avg_results = {}
        for key in results[0].keys():
            if key in ["seq_name"]:
                continue
            elif key in ["timestamp"]:
                avg_results[key] = results[0][key]
            else:
                avg_results[key] = np.mean([result[key] for result in results])
        print("Average results:")
        print(avg_results)
            
        # write
        with open(op.join(args.output, "avg_results.json"), "w") as f:
            json.dump(avg_results, f, indent=4)
        print(f"Saved average results to avg_results.json")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip_p",
        type=str,
        default=None,
        help="Path to the zip file containing the predictions.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output directory.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
