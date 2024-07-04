import sys
import json


def read_json_file(file_path):
    """Reads a JSON file and returns the data as a dictionary."""
    with open(file_path, "r") as file:
        data = json.load(file)
    return {k: v for k, v in data.items() if isinstance(v, (int, float))}


def main(hash_codes):
    results = []

    # Construct file paths from hash codes and read the JSON data
    for hash_code in hash_codes:
        file_path = f"logs/{hash_code}/checkpoints/last.ckpt.metric.json"
        try:
            results.append(read_json_file(file_path))
        except FileNotFoundError:
            print(f"Error: File not found for hash code {hash_code}")
            return
        except json.JSONDecodeError:
            print(f"Error: File content is not valid JSON for hash code {hash_code}")
            return
    # Initialize a dictionary to store total sums
    total_metrics = {key: 0 for key in results[0].keys()}

    # Sum up all metrics
    for result in results:
        for key, value in result.items():
            total_metrics[key] += value

    # Calculate averages
    average_metrics = {
        key: total / len(results) for key, total in total_metrics.items()
    }

    # Print the averages
    print("Average metrics:")
    for metric, average in average_metrics.items():
        print(f"{metric}: {average:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summarize_metrics.py <hash_code1> <hash_code2> ...")
    else:
        main(sys.argv[1:])
