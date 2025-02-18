import sys
import argparse
from pathlib import Path

import h5py
import scipy as sp
import numpy as np
import pandas as pd
import magic


def h5_tree(val, pre='', out=''):
    '''
    This code is modified based on the impelemtation by
    https://stackoverflow.com/questions/61133916/is-there-in-python-a-single-function-that-shows-the-full-structure-of-a-hdf5-fi
    '''
    length = len(val)
    for key, val in val.items():
        length -= 1
        if length == 0:
            if type(val) == h5py._hl.group.Group:
                out += pre + '└── ' + key + '\n'
                out = h5_tree(val, pre+'    ', out)

            else:
                out += pre + '└── ' + key + f' {val.shape}\n'

        else:
            if type(val) == h5py._hl.group.Group:
                out += pre + '├── ' + key + '\n'
                out = h5_tree(val, pre+'│   ', out)

            else:
                out += pre + '├── ' + key + f' {val.shape}\n'

    return out


def get_obj(f, keys):
    for key in keys:
        f = f[key]

    return f


def verify(args):     
    # Verify file format
    # DB: should return file output rather than care about ending
    # print(magic.from_file(args.input))
    test_str = magic.from_file(args.input)
    if 'Hierarchical Data Format (version 5) data' not in test_str:
        print(
            f"Error 128: The input file format ({args.input.split('.')[1]}) is invalid." \
            ' Please upload hdf5 file.'
        )
        sys.exit(128)

    with h5py.File(args.input, mode='r') as f:
        keys = list(f.keys())
        tree = h5_tree(f)

        if len(keys) == 0:
            print(
                "Error 129: Faild to read the input file. The input file hierarchy doesn't include any objects (keys)."
            )
            sys.exit(129)
        
        elif len(keys) == 1 and args.mat_obj is None and args.obs_tree is None and args.var_tree is None:
            try:
                # Verify mat, var, and obs objects
                data = pd.read_hdf(args.input, key=keys[0])
                mat = data.values
                var = data.columns.to_numpy(str)
                obs = data.index.to_numpy(str)
            except:
                print(
                    'Error 130: Faild to read the input file.' \
                    ' Please provide the correct HDF5 object names (keys) by setting the arguments -obs and -var'
                )
                print('input structure' + '\n' + tree)
                sys.exit(130)

            # Verify the shape
            if obs.shape[0] != mat.shape[0] or var.shape[0] != mat.shape[1]:
                print(
                    f'Error 131: The shape of obs and var  objects {obs.shape[0], var.shape[0]} must match the shape of the matrix {mat.shape}.' \
                    ' Please provide the correct HDF5 object names (keys) by setting the arguments -obs and -var'
                )
                print('input structure' + '\n' + tree)
                sys.exit(131)

        else:
            # Verify var object
            if args.var_tree is None:
                try:
                    var_key1 = 'var_names' if 'var_names' in keys else 'features' if 'features' in keys else 'var'
                    if var_key1 != 'var':
                        var = np.array(f[var_key1], dtype=str)
                    else:
                        var_key2 = 'feature_name' if 'feature_name' in f[var_key1].keys() else '_index'
                        var = np.array(f[var_key1][var_key2], dtype=str)
                except:
                    print(
                        "Error 132: Failed to read the var object, the input file hierarchy doesn't include" \
                        ' any of the default objects (e.g. var_names, var -> feature_name, or var -> _index).' \
                        ' Please provide the correct HDF5 object names (keys) by setting the argument -var'
                    )
                    print('input structure' + '\n' + tree)
                    sys.exit(132)

            else:
                try:
                    var = np.array(get_obj(f, args.var_tree), dtype=str)
                except:
                    print(
                        f"Error 133: The object names (keys) provided ({' -> '.join(args.var_tree)}) don't exist." \
                        ' Please provide the correct HDF5 object names (keys) by setting the argument -var'
                    )
                    print('input structure' + '\n' + tree)
                    sys.exit(133)

            # Verify obs object
            if args.obs_tree is None:
                try:
                    obs_key1 = 'obs_names' if 'obs_names' in keys else 'barcodes' if 'barcodes' in keys else 'obs'
                    if obs_key1 != 'obs':
                        obs = np.array(f[obs_key1], dtype=str)
                    else:
                        obs_key2 = 'barcode' if 'barcode' in f[obs_key1].keys() else '_index'
                        obs = np.array(f[obs_key1][obs_key2], dtype=str)
                except:
                    print(
                        "Error 134: Failed to read the obs object, the input file hierarchy doesn't include" \
                        ' any of the default objects (e.g. obs_names, barcodes, obs -> barcode, or obs -> _index).' \
                        ' Please provide the correct HDF5 object names (keys) by setting the argument -obs'
                    )
                    print('input structure' + '\n' + tree)
                    sys.exit(134)

            else:
                try:
                    obs = np.array(get_obj(f, args.obs_tree), dtype=str)
                except:
                    print(
                        f"Error 135: The object names (keys) provided ({' -> '.join(args.obs_tree)}) don't exist." \
                        ' Please provide the correct HDF5 object names (keys) by setting the argument -obs'
                    )
                    print('input structure' + '\n' + tree)
                    sys.exit(135)

            # Verify mat object
            if args.mat_obj is None:
                mat_key = 'exprs' if 'exprs' in f.keys() else 'matrix' if 'matrix' in f.keys() else 'X'
                
                try:
                    if isinstance(f[mat_key], h5py.Group):
                        try:
                            mat = sp.sparse.csr_matrix((f[mat_key]['data'], f[mat_key]['indices'], f[mat_key]['indptr']), shape=f[mat_key]['shape'])
                        except:
                            mat = sp.sparse.csr_matrix((f[mat_key]['data'], f[mat_key]['indices'], f[mat_key]['indptr']))

                        mat = np.array(mat.toarray(), dtype=np.float32)
                    else:
                        mat = np.array(f[mat_key], dtype=np.float32)
                except Exception as e:
                    print(
                        "Error 136: Failed to read the mat object, the input file hierarchy doesn't include" \
                        ' any of the default objects (e.g. exprs or X).' \
                        ' Please provide the correct HDF5 object name (key) by setting the argument -mat' + e
                    )
                    print('input structure' + '\n' + tree)
                    sys.exit(136)

            else:
                try:
                    if isinstance(f[args.mat_obj], h5py.Group):
                        try:
                            mat = sp.sparse.csr_matrix((f[args.mat_obj]['data'], f[args.mat_obj]['indices'], f[args.mat_obj]['indptr']), shape=f[args.mat_obj]['shape'])
                        except:
                            mat = sp.sparse.csr_matrix((f[args.mat_obj]['data'], f[args.mat_obj]['indices'], f[args.mat_obj]['indptr']))

                        mat = np.array(mat.toarray(), dtype=np.float32)
                    else:
                        mat = np.array(f[args.mat_obj], dtype=np.float32)
                except:
                    print(
                        f"Error 137: The matrix object name (key) provided (--mat_obj {args.mat_obj}) doesn't exist." \
                        ' Please provide the correct HDF5 object name (key) by setting the argument -mat'
                    )
                    print('input structure' + '\n' + tree)
                    sys.exit(137)

            # Verify the shape
            if obs.shape[0] != mat.shape[0] or var.shape[0] != mat.shape[1]:
                print(
                    f'Error 131: The shape of obs and the shape of var {obs.shape[0], var.shape[0]} must match the shape of the matrix {mat.shape}.' \
                    ' Please provide the correct HDF5 object names (keys) by setting the arguments -obs and -var'
                )
                print('input structure' + '\n' + tree)
                sys.exit(131)
                
    # Cross referencing genes
    query_genes = obs if args.transpose == 1 else var
    query_genes = query_genes.flatten()
    if np.unique(query_genes).shape[0] != query_genes.shape[0]:
        print(
            f'Error 138: The set of query genes has duplicates (e.g. {np.unique(query_genes)[:5]}).' \
            ' Please provide the correct HDF5 object names (keys) by setting the argument -var'
        )
        print('input structure' + '\n' + tree)
        sys.exit(138)

    ref_genes = pd.read_csv(Path(args.model_path, args.reference, f'{args.reference}-reference-genes.csv'), header=None, index_col=False)
    ref_genes = ref_genes.values.flatten()
    match = np.isin(query_genes, ref_genes, assume_unique=True)
    if match.sum() / query_genes.shape[0] <= 0.5:
        print(
            f'Error 139: {round((match.sum() / query_genes.shape[0]) * 100)}% of query genes are found in reference genes,' \
            f' this will affect the output reliability. Please try to trasnpose the input matrix or try a different reference.'
        )
        sys.exit(139)

    np.save(Path(args.output, 'mat.npy'), mat)
    np.save(Path(args.output, 'obs.npy'), obs)
    np.save(Path(args.output, 'var.npy'), var)
    np.save(Path(args.output, 'match.npy'), match)
    sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', help='the full path to the input file')
    parser.add_argument('-o', '--output', default='.', help='the full path to the output directory')
    parser.add_argument('-r', '--reference', help='the name of the reference dataset')
    parser.add_argument('-t', '--transpose', type=int, help='transpose the matrix if rows are genes and columns are cells')
    parser.add_argument('-mp', '--model_path', help='the path to the reference models and the set of reference genes')
    parser.add_argument('-mat', '--mat_obj', required=False, help='the key to the matrix object')
    parser.add_argument('-obs', '--obs_tree', required=False, nargs='+', help='the full tree to the obs object')
    parser.add_argument('-var', '--var_tree', required=False, nargs='+', help='the full tree to the var object')

    args = parser.parse_args()
    verify(args)