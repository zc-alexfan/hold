def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_p", type=str, required=False)
    args = parser.parse_args()
    from easydict import EasyDict

    args = EasyDict(vars(args))
    return args


def main():
    args = parse_args()
    import sys

    sys.path = [".", "../"] + sys.path
    import src.utils.io.ours as ours
    from common.xdict import xdict
    from src.arctic.extraction.keys import keys

    data_pred = ours.load_data(args.sd_p)
    data_pred = data_pred.to_16_bits().detach().to("cpu")
    out = xdict()
    for key in keys:
        out[key] = data_pred[key]
    seq_name = out["full_seq_name"]
    out_p = f"./arctic_preds/{seq_name}.pt"
    out.save(out_p)


if __name__ == "__main__":
    main()
