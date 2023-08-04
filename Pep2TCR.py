from .model.code.tcr_pre import main
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="single", choices=["single", "batch"])
    parser.add_argument('--cdr3', type=str)
    parser.add_argument('--pep', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--outdir', default= '.', type=str)
    args = parser.parse_args()
    
    main(args)
