import numpy as np
import pandas as pd
import argparse
import enum
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from src.multiego import io
from src.multiego.resources import type_definitions

c12_mapper = { k: v for k, v in zip(type_definitions.gromos_atp['name'], type_definitions.gromos_atp['c12']) }
ALA_CB_C12 = c12_mapper['CH3']
ALA_CB_MASS = 15.0350

class Section(enum.Enum):
    HEADER = 0
    ATOMS = 1
    BONDS = 2
    ANGLES = 3
    DIHEDRALS = 4
    PAIRS = 5
    EXCLUSIONS = 6
    SYSTEM = 7
    MOLECULES = 8


def read_ffnonbonded(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.startswith('[ nonbond_params ]'):
            first_data_line = i
            break

    first_data_line += 2
    last_data_line = len(lines)

    attypes = lines[:first_data_line]
    header  = [ x for x in attypes if x != '' and x.startswith(';') and not x.startswith('[') ]
    attypes = [ x for x in attypes if x != '' and not x.startswith(';') and not x.startswith('[') ]
    attypes = [ x.replace('\n', '') for x in attypes ]
    header  = [ x.replace('\n', '') for x in header ]
    attypes = [ x.split() for x in attypes ]
    attypes = [ x for x in attypes if len(x) != 0 and x[0] != '' and x[0] != ';' ]
    attypes = pd.DataFrame(attypes)
    attypes.columns = ['sb_type', 'atomic_number', 'mass', 'charge', 'ptype', 'c6', 'c12']
    attypes['c6'] = attypes['c6'].astype(float)
    attypes['c12'] = attypes['c12'].astype(float)

    ffnb = pd.read_csv(path, sep='\s+', skiprows=first_data_line, nrows=last_data_line-first_data_line, header=None)
    ffnb.columns = [
        'ai', 'aj', 'type', 'c6', 'c12', ';', 'sigma',
        'epsilon', 'probability', 'rc_probability', 'md_threshold',
        'rc_threshold', 'rep', 'cutoff', 'same_chain', 'source',
        'number_ai', 'number_aj'
    ]
    # ffnb = ffnb.drop(columns=[';'])

    return header, attypes, ffnb

def read_topol(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    sections = {}
    sections['header'] =     []
    sections['atoms'] =      []
    sections['bonds'] =      []
    sections['angles'] =     []
    sections['dihedrals'] =  []
    sections['pairs'] =      []
    sections['exclusions'] = []
    sections['system'] =     []
    sections['molecules'] =  []
    section = Section.HEADER
    for i, line in enumerate(lines):
        if '\n' in line: line = line.replace('\n', '')
        line = line.strip()
        if Section.HEADER == section and line.startswith(';') or line.startswith('#'):
            print(line)
            sections['header'].append(line)
        if line == '': continue
        if line.startswith(';'): continue
        if line.startswith('#'): continue
        if line.startswith('[ atoms ]'):
            section = Section.ATOMS
            continue
        elif line.startswith('[ bonds ]'):
            section = Section.BONDS
            continue
        elif line.startswith('[ angles ]'):
            section = Section.ANGLES
            continue
        elif line.startswith('[ dihedrals ]'):
            section = Section.DIHEDRALS
            continue
        elif line.startswith('[ pairs ]'):
            section = Section.PAIRS
            continue
        elif line.startswith('[ exclusions ]'):
            section = Section.EXCLUSIONS
            continue
        elif line.startswith('[ system ]'):
            section = Section.SYSTEM
            continue
        elif line.startswith('[ molecules ]'):
            section = Section.MOLECULES
            continue

        match section:
            case Section.HEADER: sections['header'].append(line)
            case Section.ATOMS: sections['atoms'].append(line)
            case Section.BONDS: sections['bonds'].append(line)
            case Section.ANGLES: sections['angles'].append(line)
            case Section.DIHEDRALS: sections['dihedrals'].append(line)
            case Section.PAIRS: sections['pairs'].append(line)
            case Section.EXCLUSIONS: sections['exclusions'].append(line)
            case Section.SYSTEM: sections['system'].append(line)
            case Section.MOLECULES: sections['molecules'].append(line)
            case _: print(f"WARNING: line {i} not in any section")
    
    topol = {}
    topol['header'] = sections['header']
    atms = pd.DataFrame([ x.split() for x in sections['atoms'] ])
    atms.columns = ['number', 'sb_type', 'resnum', 'resname', 'name', 'cgnr']
    atms.astype({'number': int, 'sb_type': str, 'resnum': int, 'resname': str, 'name': str, 'cgnr': int})
    topol['atoms'] = atms
    bnds = pd.DataFrame([ x.split() for x in sections['bonds'] ])
    bnds.columns = ['ai', 'aj', 'func', 'req', 'k']
    bnds.astype({'ai': int, 'aj': int, 'func': int, 'req': float, 'k': float})
    topol['bonds'] = bnds
    angs = pd.DataFrame([ x.split() for x in sections['angles'] ])
    angs.columns = ['ai', 'aj', 'ak', 'func', 'theteq', 'k']
    angs.astype({'ai': int, 'aj': int, 'ak': int, 'func': int, 'theteq': float, 'k': float})
    topol['angles'] = angs
    dihs = pd.DataFrame([ x.split() for x in sections['dihedrals'] ])
    dihs.columns = ['ai', 'aj', 'ak', 'al', 'func', 'phase', 'phi_k', 'per']
    dihs_proper = dihs[~dihs['per'].isna()]
    dihs_proper.astype({'ai': int, 'aj': int, 'ak': int, 'al': int, 'func': int, 'phase': float, 'phi_k': float, 'per': int})
    dihs_improper = dihs[dihs['per'].isna()]
    dihs_improper = dihs_improper.drop(columns=['per'])
    dihs_improper.columns = ['ai', 'aj', 'ak', 'al', 'func', 'psi_eq', 'psi_k']
    dihs_improper.astype({'ai': int, 'aj': int, 'ak': int, 'al': int, 'func': int, 'psi_eq': float, 'psi_k': float})
    topol['dihedrals'] = dihs_proper
    topol['improper_dihedrals'] = dihs_improper
    prs = pd.DataFrame([ x.split() for x in sections['pairs'] ])
    prs.columns = ['ai', 'aj', 'func', 'c6', 'c12', ';', 'probability', 'rc_probability', 'source']
    prs = prs.drop(columns=[';'])
    prs.astype({'ai': int, 'aj': int, 'func': int, 'c6': float, 'c12': float, 'probability': float, 'rc_probability': float, 'source': str})
    topol['pairs'] = prs
    exc = pd.DataFrame([ x.split() for x in sections['exclusions'] ])
    exc.columns = ['ai', 'aj']
    exc.astype({'ai': int, 'aj': int})
    topol['exclusions'] = exc
    topol['system'] = sections['system']
    topol['molecules'] = sections['molecules']

    return topol

def mutate_residue(topol, att, ffnb, resi, sysname):
    print(f'Mutating residue {resi} in system {sysname}')
    # mutate topol
    atoms = topol['atoms']
    atoms = atoms.astype({'number': int, 'sb_type': str, 'resnum': int, 'resname': str, 'name': str, 'cgnr': int})
    atoms['sb_type_mut'] = atoms['sb_type']
    mutation_site = atoms[(atoms['resnum'] == resi)]['resname'].unique()[0]
    print(f'Mutation will be form {mutation_site}_{resi} to ALA_{resi}')
    
    # get indices of atoms to remove
    indices = atoms[(atoms['resnum'] == resi) & (
        ~(atoms['sb_type'].str.startswith('N_'))
        & ~(atoms['sb_type'].str.startswith('CA_'))
        & ~(atoms['sb_type'].str.startswith('C_'))
        & ~(atoms['sb_type'].str.startswith('O_'))
        & ~(atoms['sb_type'].str.startswith('CB_'))
        )
    ]['number'].values

    # remove atoms from topology
    topol['header'] = [f'; mutated topology generated with {sys.argv[0]}', 
                       f'; mutated residue {resi} in system {sysname} to ALA'] + topol['header']
    topol['atoms'] = [ x for i, x in enumerate(topol['atoms']) if i not in indices ]
    atoms.loc[(atoms['resnum'] == resi) & (
        ~(atoms['sb_type'].str.startswith('CB_'))
        & ~(atoms['sb_type'].str.startswith('N_'))
        & ~(atoms['sb_type'].str.startswith('CA_'))
        & ~(atoms['sb_type'].str.startswith('C_'))
        & ~(atoms['sb_type'].str.startswith('O_'))
    ), 'sb_type_mut'] = np.nan
    print(f'Removing {np.sum(atoms["sb_type_mut"].isna())} atoms from topology')
    atoms = atoms.dropna()
    atoms['number_mut'] = np.arange(1, len(atoms)+1)
    atoms.loc[atoms['resnum'] == resi, 'resname'] = 'ALA'
    num_mapper = {k: v for k, v in zip(atoms['number'], atoms['number_mut'])}
    atoms['number'] = atoms['number_mut']
    atoms = atoms.drop(columns=['number_mut'])
    topol['atoms'] = atoms

    # map all indices in topology to new indices
    print('Mapping indices in topology')
    topol['bonds']['ai'] = topol['bonds']['ai'].astype(int)
    topol['bonds']['aj'] = topol['bonds']['aj'].astype(int)
    # map bonds
    topol['bonds'] = topol['bonds'][~topol['bonds']['ai'].isin(indices)]
    topol['bonds'] = topol['bonds'][~topol['bonds']['aj'].isin(indices)]
    topol['bonds']['ai'] = topol['bonds']['ai'].map(num_mapper)
    topol['bonds']['aj'] = topol['bonds']['aj'].map(num_mapper)
    # map angles
    topol['angles']['ai'] = topol['angles']['ai'].astype(int)
    topol['angles']['aj'] = topol['angles']['aj'].astype(int)
    topol['angles']['ak'] = topol['angles']['ak'].astype(int)
    topol['angles'] = topol['angles'][~topol['angles']['ai'].isin(indices)]
    topol['angles'] = topol['angles'][~topol['angles']['aj'].isin(indices)]
    topol['angles'] = topol['angles'][~topol['angles']['ak'].isin(indices)]
    topol['angles']['ai'] = topol['angles']['ai'].map(num_mapper)
    topol['angles']['aj'] = topol['angles']['aj'].map(num_mapper)
    topol['angles']['ak'] = topol['angles']['ak'].map(num_mapper)
    # map dihedrals
    topol['dihedrals']['ai'] = topol['dihedrals']['ai'].astype(int)
    topol['dihedrals']['aj'] = topol['dihedrals']['aj'].astype(int)
    topol['dihedrals']['ak'] = topol['dihedrals']['ak'].astype(int)
    topol['dihedrals']['al'] = topol['dihedrals']['al'].astype(int)
    topol['dihedrals'] = topol['dihedrals'][~topol['dihedrals']['ai'].isin(indices)]
    topol['dihedrals'] = topol['dihedrals'][~topol['dihedrals']['aj'].isin(indices)]
    topol['dihedrals'] = topol['dihedrals'][~topol['dihedrals']['ak'].isin(indices)]
    topol['dihedrals'] = topol['dihedrals'][~topol['dihedrals']['al'].isin(indices)]
    topol['dihedrals']['ai'] = topol['dihedrals']['ai'].map(num_mapper)
    topol['dihedrals']['aj'] = topol['dihedrals']['aj'].map(num_mapper)
    topol['dihedrals']['ak'] = topol['dihedrals']['ak'].map(num_mapper)
    topol['dihedrals']['al'] = topol['dihedrals']['al'].map(num_mapper)
    # map pairs
    topol['pairs']['ai'] = topol['pairs']['ai'].astype(int)
    topol['pairs']['aj'] = topol['pairs']['aj'].astype(int)
    topol['pairs'] = topol['pairs'][~topol['pairs']['ai'].isin(indices)]
    topol['pairs'] = topol['pairs'][~topol['pairs']['aj'].isin(indices)]
    topol['pairs']['ai'] = topol['pairs']['ai'].map(num_mapper)
    topol['pairs']['aj'] = topol['pairs']['aj'].map(num_mapper)
    # map exclusions
    topol['exclusions']['ai'] = topol['exclusions']['ai'].astype(int)
    topol['exclusions']['aj'] = topol['exclusions']['aj'].astype(int)
    topol['exclusions'] = topol['exclusions'][~topol['exclusions']['ai'].isin(indices)]
    topol['exclusions'] = topol['exclusions'][~topol['exclusions']['aj'].isin(indices)]
    topol['exclusions']['ai'] = topol['exclusions']['ai'].map(num_mapper)
    topol['exclusions']['aj'] = topol['exclusions']['aj'].map(num_mapper)

    # mutate ffnb
    print('Mutating ffnb')
    ffnb_mut = ffnb.copy()
    # remove all interactions that are not backbone so N, CA, C, O and CB
    # ffnb_mut[(ffnb_mut['ai'].str.contains(f'{sysname}_{resi}')) | ffnb_mut['aj'].str.contains(f'{sysname}_{resi}') & ~ffnb_mut['ai'].str.contains('\bN_|\bCA_|\bC_|\bO_|\bCB_') & ~ffnb_mut['aj'].str.contains('\bN_|\bCA_|\bC_|\bO_|\bCB_')] = np.nan
    ffnb_mut.loc[(ffnb_mut['ai'].str.split('_').str[2] == str(resi)) & (
        ~(ffnb_mut['ai'].str.startswith('CB_'))
        & ~(ffnb_mut['ai'].str.startswith('N_'))
        & ~(ffnb_mut['ai'].str.startswith('CA_'))
        & ~(ffnb_mut['ai'].str.startswith('C_'))
        & ~(ffnb_mut['ai'].str.startswith('O_'))
    ), 'ai'] = np.nan
    ffnb_mut.loc[(ffnb_mut['aj'].str.split('_').str[2] == str(resi)) & (
        ~(ffnb_mut['aj'].str.startswith('CB_'))
        & ~(ffnb_mut['aj'].str.startswith('N_'))
        & ~(ffnb_mut['aj'].str.startswith('CA_'))
        & ~(ffnb_mut['aj'].str.startswith('C_'))
        & ~(ffnb_mut['aj'].str.startswith('O_'))
    ), 'aj'] = np.nan
    print(f'Removing {np.sum(ffnb_mut["ai"].isna()) + np.sum(ffnb_mut["aj"].isna())} interactions from ffnb')
    ffnb_mut = ffnb_mut.dropna()


    att_mut = att.copy()
    att_mut.loc[(att_mut['sb_type'].str.split('_').str[2] == str(resi)) & (
        ~(att_mut['sb_type'].str.startswith('CB_'))
        & ~(att_mut['sb_type'].str.startswith('N_'))
        & ~(att_mut['sb_type'].str.startswith('CA_'))
        & ~(att_mut['sb_type'].str.startswith('C_'))
        & ~(att_mut['sb_type'].str.startswith('O_'))
    ), 'sb_type'] = np.nan
    print(f'Removing {np.sum(att_mut["sb_type"].isna())} atoms from ffnonbonded')
    att_mut = att_mut.dropna()
    # set the correct mass and c12 for the CB atom
    att_mut.loc[ ( np.array([ x[2] for x in np.array(att_mut['sb_type'].str.split('_')) ]) == str(resi) )
            & ( att_mut['sb_type'].str.startswith('CB') ), 'mass'] = ALA_CB_MASS
    att_mut.loc[ ( np.array([ x[2] for x in np.array(att_mut['sb_type'].str.split('_')) ]) == str(resi) )
            & ( att_mut['sb_type'].str.startswith('CB') ), 'c12'] = ALA_CB_C12
    
    ffnb_mut['molecule_name_ai'] = ffnb_mut['ai'].str.split('_').str[1]
    ffnb_mut['molecule_name_aj'] = ffnb_mut['aj'].str.split('_').str[1]

    # reset indices
    att_mut = att_mut.reset_index(drop=True)
    ffnb_mut = ffnb_mut.reset_index(drop=True)

    return topol, att_mut, ffnb_mut

def write_topology(topol, output_dir):
    with open(f'{output_dir}/topol_GRETA.top', 'w') as f:
        for line in topol['header']:
            f.write(f'{line}\n')
        f.write('\n\n')
        f.write('[ atoms ]\n; ')
        f.write(topol['atoms'].to_string(index=False))
        f.write('\n\n')
        f.write('[ bonds ]\n; ')
        f.write(topol['bonds'].to_string(index=False))
        f.write('\n\n')
        f.write('[ angles ]\n; ')
        f.write(topol['angles'].to_string(index=False))
        f.write('\n\n')
        f.write('[ dihedrals ]\n; ')
        f.write(topol['dihedrals'].to_string(index=False))
        f.write('\n\n')
        f.write('[ dihedrals ]\n; ')
        f.write(topol['improper_dihedrals'].to_string(index=False))
        f.write('\n\n')
        f.write('[ pairs ]\n; ')
        f.write(topol['pairs'].to_string(index=False))
        f.write('\n\n')
        f.write('[ exclusions ]\n; ')
        f.write(topol['exclusions'].to_string(index=False))
        f.write('\n\n')
        for line in topol['system']:
            f.write(f'{line}\n')
        f.write('\n\n')
        for line in topol['molecules']:
            f.write(f'{line}\n')
        f.write('\n')

def write_nonbonded(header, att, ffnb, output_dir):
    att["c6"] = att["c6"].map(lambda x: "{:.6e}".format(x))
    att["c12"] = att["c12"].map(lambda x: "{:.6e}".format(x))
    ffnb["c6"] = ffnb["c6"].map(lambda x: "{:.6e}".format(x))
    ffnb["c12"] = ffnb["c12"].map(lambda x: "{:.6e}".format(x))
    with open(f'{output_dir}/ffnonbonded.itp', 'w') as f:
        f.write(f'; mutated ffnonbonded generated with {sys.argv[0]}\n')
        for line in header:
            f.write(f'{line}\n')
        f.write('\n\n')
        f.write('[ atomtypes ]\n ; ')
        f.write(att.to_string(index=False))
        f.write('\n\n')
        f.write('[ nonbond_params ]\n; ')
        f.write(ffnb.to_string(index=False))

def write_output(topol, header, att, ffnb, output_dir):
    if not os.path.isdir(output_dir): os.mkdir(output_dir)
    write_topology(topol, output_dir)
    write_nonbonded(header, att, ffnb, output_dir)

if __name__ == '__main__':
    # read in topology and ffnb
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--topol', type=str, help='topology file')
    parser.add_argument('-f', '--ffnb', type=str, help='ffnonbonded file')
    parser.add_argument('-r', '--resi', type=int, help='residue number to mutate')
    parser.add_argument('-s', '--sysname', type=str, help='system name')
    parser.add_argument('-o', '--output', type=str, help='output file')

    args = parser.parse_args()

    topol = read_topol(args.topol)
    header, att, ffnb = read_ffnonbonded(args.ffnb)

    topol, att, ffnb = mutate_residue(topol, att, ffnb, args.resi, args.sysname)
    
    write_output(topol, header, att, ffnb, args.output)