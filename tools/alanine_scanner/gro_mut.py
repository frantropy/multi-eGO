import argparse
import os
import sys
import biotite
import biotite.structure as struc
import biotite.structure.io as strucio

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='Input file')
parser.add_argument('-o', '--output', required=True, help='Output file')
parser.add_argument('-r', '--resid', required=True, help='Residue ID')

args = parser.parse_args()

file = args.input
if not os.path.isfile(file):
    sys.exit(f"Input file '{file}' does not exist")
structure = strucio.load_structure(file)

resid = args.resid
if not resid.isdigit():
    sys.exit(f"Residue ID '{resid}' is not a number")

resid = int(resid)
residue_names = structure.res_name
resid_idx = structure.res_id
residue_names[resid_idx == resid] = "ALA"
structure.set_annotation("res_name", residue_names)
ala_filter = (structure.res_id == resid) \
    & (structure.atom_name != 'CA') \
    & (structure.atom_name != 'CB') \
    & (structure.atom_name != 'C') \
    & (structure.atom_name != 'O') \
    & (structure.atom_name != 'N')
structure = structure[~ala_filter]
strucio.save_structure(args.output, structure)
