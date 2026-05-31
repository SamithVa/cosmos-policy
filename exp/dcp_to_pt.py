from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--dcp_dir", type=str, help="Directory containing the DCP checkpoint")
parser.add_argument("--pt_output_dir", type=str, help="Output directory for the PyTorch checkpoint")

args = parser.parse_args()


dcp_to_torch_save(
    args.dcp_dir,
    args.pt_output_dir,
)