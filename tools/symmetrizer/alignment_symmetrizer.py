import pandas as pd
import numpy as np
import argparse

def read_ffnonbonded(path):
    '''
    Read the ffnonbonded.itp file and create a dataframe with the data

    Parameters
    ----------
    path : str
        The path to the ffnonbonded.itp file

    Returns
    -------
    df : pandas.DataFrame
        The dataframe with the data
    '''
    with open(path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.startswith('[ nonbond_params ]'):
            first_data_line = i
            break

    first_data_line += 2
    last_data_line = len(lines)

    df = pd.read_csv(path, sep='\s+', skiprows=first_data_line, nrows=last_data_line-first_data_line, header=None)
    df.columns = [
        'ai', 'aj', 'type', 'c6', 'c12', ';', 'sigma',
        'epsilon', 'probability', 'rc_probability', 'md_threshold',
        'rc_threshold', 'rep', 'cutoff', 'same_chain', 'source',
        'number_ai', 'number_aj'
    ]

    return df

def read_alignment(path, sections):
    '''
    Read the alignment file and create a dataframe with the alignment and the sections

    Parameters
    ----------
    path : str
        The path to the alignment file

    sections : list of tuples
        The sections of the protein sequence that correspond to each domain
    
    Returns
    -------
    df : pandas.DataFrame
        The dataframe with the alignment and the sections
    '''
    n_domains = len(sections)
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = lines[3:]
    lines = [ x for x in lines if x != '\n' ]
    prefix_end = lines[0].find(' ')
    data_start = len(lines[0][prefix_end:]) - len(lines[0][prefix_end:].lstrip()) + prefix_end
    df = pd.DataFrame()
    for i in range(n_domains):
        pro_seq = lines[i::n_domains+1]
        pro_seq = [ x.replace('\n', '')[data_start:] for x in pro_seq ]
        seq = np.array([ x for x in ''.join(pro_seq) ])
        df[f'domain_{i+1}'] = seq.T
        n_gaps = np.sum(seq == '-')
        df[f'domain_{i+1}_index'] = -1
        try:
            df.loc[df[f'domain_{i+1}'] != '-', f'domain_{i+1}_index'] = np.arange(sections[i][0], sections[i][1])
        except ValueError:
            sum = np.sum(seq != '-')
            print(f'Error: the number of gaps in domain {i+1} is {n_gaps} and expected number of residues is {sum} but the number of residues is {sections[i][1]-sections[i][0]}')
            exit(1)
 
    alignment = lines[n_domains::n_domains+1]
    alignment = "".join([ x.replace('\n', '')[data_start:] for x in alignment ])
    alignment = np.array([ x for x in alignment ])
    df['alignment'] = alignment

    return df

def recuperate_header(path):
    '''
    Recuperate the header of the ffnonbonded.itp file
    
    Parameters
    ----------
    path : str
        The path to the original ffnonbonded.itp file

    Returns
    -------
    header : str
        The header of the ffnonbonded.itp file
    '''
    with open(path, 'r') as f:
        lines = f.readlines()
    header = lines[:lines.index('[ nonbond_params ]\n')]

    return header

def symmetrize_ffnonbonded(ffnb, pairs):
    '''
    Symmetrize the ffnonbonded.itp file according to the alignment file and the sections

    Parameters
    ----------
    ffnb : pandas.DataFrame
        The ffnonbonded dataframe

    pairs : numpy.ndarray
        The pairs of residues to symmetrize

    Returns
    -------
    ffnb : pandas.DataFrame
        The symmetrized ffnonbonded dataframe
    '''
    ffnb_sym = pd.DataFrame(columns=ffnb.columns)
    ffnb['sym'] = 'no'
    sbtype_to_number = { x[0]: int(x[1]) for x in ffnb[['ai', 'number_ai']].to_numpy() }

    ffnb['ri'] = [ int(x[2]) for x in ffnb['ai'].str.split('_') ]
    ffnb['rj'] = [ int(x[2]) for x in ffnb['aj'].str.split('_') ]

    for i, pair in enumerate(pairs):
        ffnb_sym_i = ffnb.loc[(ffnb['ri'] == pair[0]) | (ffnb['rj'] == pair[0])]
        ffnb_sym_j = ffnb.loc[(ffnb['ri'] == pair[1]) | (ffnb['rj'] == pair[1])]
        ffnb_sym_i = ffnb_sym_i.loc[ffnb_sym_i['epsilon'] > 0]
        ffnb_sym_j = ffnb_sym_j.loc[ffnb_sym_j['epsilon'] > 0]

        ffnb_sym_i_only = ffnb_sym_i.loc[(~ffnb_sym_i['ri'].isin(ffnb_sym_j['ri'])) & (~ffnb_sym_i['rj'].isin(ffnb_sym_j['rj']))]
        ffnb_sym_j_only = ffnb_sym_j.loc[(~ffnb_sym_j['ri'].isin(ffnb_sym_i['ri'])) & (~ffnb_sym_j['rj'].isin(ffnb_sym_i['rj']))]
        ffnb_sym_i_only.loc[:, 'ai'] = ffnb_sym_i_only['ai'].str.replace(f'_{pair[0]}', f'_{pair[1]}')
        ffnb_sym_i_only.loc[:, 'aj'] = ffnb_sym_i_only['aj'].str.replace(f'_{pair[0]}', f'_{pair[1]}')
        ffnb_sym_j_only.loc[:, 'ai'] = ffnb_sym_j_only['ai'].str.replace(f'_{pair[1]}', f'_{pair[0]}')
        ffnb_sym_j_only.loc[:, 'aj'] = ffnb_sym_j_only['aj'].str.replace(f'_{pair[1]}', f'_{pair[0]}')
        ffnb_sym_i_only.loc[:, 'number_ai'], ffnb_sym_i_only.loc[:, 'number_aj'] = ffnb_sym_i_only['number_aj'], ffnb_sym_i_only['number_ai']
        ffnb_sym_j_only.loc[:, 'number_ai'], ffnb_sym_j_only.loc[:, 'number_aj'] = ffnb_sym_j_only['number_aj'], ffnb_sym_j_only['number_ai']

        ffnb_sym = pd.concat([ffnb_sym, ffnb_sym_i_only, ffnb_sym_j_only], ignore_index=True)
        ##############################
        # WARNING not tested yet     #
        ##############################`
        ffnb_sym_i_overlap = ffnb_sym_i.loc[(ffnb_sym_i['ri'].isin(ffnb_sym_j['ri'])) & (ffnb_sym_i['rj'].isin(ffnb_sym_j['rj']))]
        ffnb_sym_j_overlap = ffnb_sym_j.loc[(ffnb_sym_j['ri'].isin(ffnb_sym_i['ri'])) & (ffnb_sym_j['rj'].isin(ffnb_sym_i['rj']))]
        ffnb_sym_i_overlap = ffnb_sym_i_overlap.sort_values(by=['ai', 'aj'])
        ffnb_sym_j_overlap = ffnb_sym_j_overlap.sort_values(by=['ai', 'aj'])
        ffnb_sym_i_smaller = ffnb_sym_i_overlap.loc[ffnb_sym_i_overlap['sigma'] < ffnb_sym_j_overlap['sigma']]
        ffnb_sym_j_smaller = ffnb_sym_j_overlap.loc[ffnb_sym_j_overlap['sigma'] < ffnb_sym_i_overlap['sigma']]
        ffnb_sym_i_smaller['ai'] = ffnb_sym_i_smaller['ai'].str.replace(f'_{pair[0]}', f'_{pair[1]}')
        ffnb_sym_i_smaller['aj'] = ffnb_sym_i_smaller['aj'].str.replace(f'_{pair[0]}', f'_{pair[1]}')
        ffnb_sym_j_smaller['ai'] = ffnb_sym_j_smaller['ai'].str.replace(f'_{pair[1]}', f'_{pair[0]}')
        ffnb_sym_j_smaller['aj'] = ffnb_sym_j_smaller['aj'].str.replace(f'_{pair[1]}', f'_{pair[0]}')
        ##############################
        # END WARNING                #
        ##############################

    # create a sym entry from the ri and rj columns
    ffnb_sym['sym'] = ffnb_sym['ri'].astype(int).astype(str) + '_' + ffnb_sym['rj'].astype(int).astype(str)
    ffnb = pd.concat([ffnb_sym, ffnb], ignore_index=True)
    ffnb = ffnb.sort_values(by=['sigma', 'epsilon'], ascending=[True, True])
    ffnb = ffnb.drop_duplicates(subset=['ai', 'aj', 'type'], keep='first')
    ffnb = ffnb.drop(columns=['ri', 'rj'])
    # ffnb = ffnb.sort_values(by=['number_ai', 'number_aj'])
    # sort the values by number_ai and number_aj symmetrically (i.e. 1_2 and 2_1 are considered the same)
    # ffnb['number_ai'] = ffnb['number_ai'].astype(int)
    # ffnb['number_aj'] = ffnb['number_aj'].astype(int)
    # ffnb['number_ai'] = [ sbtype_to_number[x] for x in ffnb['ai'] ]
    # ffnb['number_aj'] = [ sbtype_to_number[x] for x in ffnb['aj'] ]
    ffnb['number_ai'], ffnb['number_aj'] = np.sort(ffnb[['number_ai', 'number_aj']].to_numpy(), axis=1).T
    dropper = np.sort(ffnb[['ai', 'aj']], axis=1)
    dropper = [ x[0] + '_' + x[1] for x in dropper ]
    ffnb['dropper'] = dropper
    ffnb = ffnb.drop_duplicates(subset=['dropper'], keep='first')
    ffnb = ffnb.sort_values(by=['number_ai', 'number_aj'])

    return ffnb

def sort_ffnb(ffnb):
    '''
    Sort the ffnonbonded.itp file by atomtypes

    Parameters
    ----------
    ffnb : pandas.DataFrame
        The ffnonbonded dataframe

    Returns
    -------
    ffnb : pandas.DataFrame
        The sorted ffnonbonded dataframe
    '''
    ffnb['sort_ai'] = [ int(x.split('_')[2]) for x in ffnb['ai'] ]
    ffnb['sort_aj'] = [ int(x.split('_')[2]) for x in ffnb['aj'] ]

    ffnb['swap'] = ffnb['sort_ai'] > ffnb['sort_aj']
    ffnb.loc[ffnb['swap'] == True, 'ai'], ffnb.loc[ffnb['swap'] == True, 'aj'] = ffnb.loc[ffnb['swap'] == True, 'aj'], ffnb.loc[ffnb['swap'] == True, 'ai']
    
    ffnb = ffnb.drop(columns=['swap'])
    ffnb = ffnb.sort_values(by=['sort_ai', 'sort_aj', 'ai', 'aj'])
    ffnb = ffnb.drop(columns=['sort_ai', 'sort_aj'])

    return ffnb

def check_att_presence(ffnb_header, ffnb_data):
    '''
    Check if all atomtypes in [ nonbond_params ] are present in [ atomtypes ]

    Parameters
    ----------
    ffnb_header : str
        The header of the ffnonbonded.itp file

    ffnb_data : str
        The data of the ffnonbonded.itp file

    Returns
    -------
    True : bool
        If all atomtypes in [ nonbond_params ] are present in [ atomtypes ]
    '''
    header_lines = ffnb_header.split('\n')
    ffnb_lines = ffnb_data.split('\n')

    for i, line in enumerate(header_lines):
        if line.startswith('[ atomtypes ]'):
            atomtypes_start = i
            break

    for i, line in enumerate(header_lines[atomtypes_start:]):
        if line == '':
            atomtypes_end = i
            break
    
    header_lines = header_lines[atomtypes_start+2:atomtypes_start+atomtypes_end]
    sbtypes = [ x.split()[0] for x in header_lines ]
    sbtypes = np.array(sbtypes)

    ffnb_lines = ffnb_lines[1:]
    pairs_i = [ x.split()[0] for x in ffnb_lines ]
    pairs_i = np.array(pairs_i)
    pairs_j = [ x.split()[1] for x in ffnb_lines ]
    pairs_j = np.array(pairs_j)

    sum_i = np.sum(~np.isin(pairs_i, sbtypes))
    sum_j = np.sum(~np.isin(pairs_j, sbtypes))

    if sum_i > 0:
        raise ValueError(f'''Error: some atomtypes in the first column of [ nonbond_params ] are not present in [ atomtypes ]
                         Check the --sections argument
                         Wrong pairs ::\n{pairs_i[~np.isin(pairs_i, sbtypes)].tolist()}''')
    if sum_j > 0:
        raise ValueError(f'''Error: some atomtypes in the second column of [ nonbond_params ] are not present in [ atomtypes ]
                         Check the --sections argument
                         Wrong pairs ::\n{pairs_j[~np.isin(pairs_j, sbtypes)].tolist()}''')

    return True

def remove_intra_sym(ffnb, sections):
    '''
    Remove intra-domain interactions that resulted from the symmetrization process

    Parameters
    ----------
    ffnb : pandas.DataFrame
        The ffnonbonded dataframe

    sections : list of tuples
        The sections of the protein sequence that correspond to each domain

    Returns
    -------
    ffnb : pandas.DataFrame
        The ffnonbonded dataframe without intra-domain interactions
    '''

    ffnb['ri'] = [ int(x[2]) for x in ffnb['ai'].str.split('_') ]
    ffnb['rj'] = [ int(x[2]) for x in ffnb['aj'].str.split('_') ]
    ffnb = sort_ffnb(ffnb)
    ffnb['intra_domain'] = False
    for section in sections:
        ffnb.loc[(ffnb['ri'] >= section[0]) & (ffnb['rj'] <= section[1]), 'intra_domain'] = True
        ffnb.loc[(ffnb['ri'] <= section[0]) & (ffnb['rj'] >= section[1]), 'intra_domain'] = True
    ffnb = ffnb.loc[(ffnb['sym'] == 'no') | (ffnb['intra_domain'] == False)]
    ffnb = ffnb.drop(columns=['ri', 'rj', 'intra_domain'])

    return ffnb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ffnonbonded', type=str, help='path to ffnonbonded.itp', required=True)
    parser.add_argument('--alignment', type=str, help='path to clustalw alignment file', required=True)
    parser.add_argument('--output', default='ffnonbonded.itp', type=str, help='path to output file', required=True)
    parser.add_argument('--sections', help='sections to symmetrize (i.e.: 1:40,41:50)', required=True)
    parser.add_argument('--no_intra', action='store_true', help='do not symmetrize intra-domain interactions')
    args = parser.parse_args()

    # check args section for proper format
    if not all([ x.split(':')[0].isdigit() and x.split(':')[0].isdigit() and len(x.split(':'))==2 for x in args.sections.split(',') ]):
        raise ValueError(f'Error: --sections argument must be in the form of 1:40,41:50,51:60,61:70')
    args.sections = [ ( int(x.split(':')[0]), int(x.split(':')[1]) ) for x in args.sections.split(',') ]
    n_domains = len(args.sections)
    header = recuperate_header(args.ffnonbonded)
    ffnb = read_ffnonbonded(args.ffnonbonded)
    alignment = read_alignment(args.alignment, args.sections)

    alignment['symmetrize'] = alignment['alignment'] == '*'
    sym_pairs = alignment[alignment['symmetrize'] == True][[ f'domain_{i+1}_index' for i in range(n_domains) ]].to_numpy()

    print(alignment.to_string())
    ffnb_sym = symmetrize_ffnonbonded(ffnb, sym_pairs)
    ffnb_sym = sort_ffnb(ffnb_sym)

    if args.no_intra: ffnb_sym = remove_intra_sym(ffnb_sym, args.sections)

    check_att_presence(''.join(header), ffnb_sym.to_string(index=False))

    ffnb_string = ''.join(header)
    ffnb_string += '[ nonbond_params ]\n;'
    ffnb_string += ffnb_sym.to_string(index=False)
    ffnb_string += '\n'

    with open(args.output, 'w') as f:
        f.write(ffnb_string)
