import argparse
from stage2_build_corrmat import build_Cij

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_dir", required=True)
    parser.add_argument("--irrep", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    build_Cij(
        stage1_dir=args.stage1_dir,
        irrep=args.irrep,
        output_file=args.output
    )

