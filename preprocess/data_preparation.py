import pandas as pd
import os

from sklearn.model_selection import train_test_split

import utils.file as uf
import configuration.config_default as cfgd
import preprocess.property_change_encoder as pce

SEED = 42
SPLIT_RATIO = 0.8

# import packages
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from e3fp.pipeline import fprints_from_smiles, fprints_from_mol

from multiprocessing import Pool


NUM_WORKERS = 32



# define function that transforms SMILES strings into E3FPs
def e3fp_from_smiles(smiles_string):

    """Function to create E3FP from a SMILES string."""
    # mol = Chem.MolFromSmiles(smiles_string)
    # print(mol)
    # # fprint_params = {'bits': 2**10, 'radius_multiplier': radius_multiplier, 'rdkit_invariants': True}
    # fpdict = fprints_dict_from_mol(mol)
    # print(fpdict)
    # # return np.array(fp[5][0].to_rdkit())
    fprint_params = {'bits': 2**7, 'rdkit_invariants': True}
    confgen_params = {'max_energy_diff': 20, 'first': 1}
    #look into details of conformation generation of e3fp

    feature_list = fprints_from_smiles(smiles_string, "placeholder_name",  fprint_params=fprint_params,confgen_params=confgen_params, save=False)
    # confgen_params=confgen_params
    return np.array(feature_list[0].to_rdkit())


def get_smiles_list(file_name):
    """
    Get smiles list for building vocabulary
    :param file_name:
    :return:
    """
    pd_data = pd.read_csv(file_name, sep=",")

    print("Read %s file" % file_name)
    smiles_list = pd.unique(pd_data[['Source_Mol', 'Target_Mol']].values.ravel('K'))
    print("Number of SMILES in chemical transformations: %d" % len(smiles_list))

    return smiles_list

def split_data(input_transformations_path, LOG=None):
    """
    Split data into training, validation and test set, write to files
    :param input_transformations_path:L
    :return: dataframe
    """
    data = pd.read_csv(input_transformations_path, sep=",")
    if LOG:
        LOG.info("Read %s file" % input_transformations_path)

    train, test = train_test_split(
        data, test_size=0.1, random_state=SEED)
    train, validation = train_test_split(train, test_size=0.1, random_state=SEED)
    if LOG:
        LOG.info("Train, Validation, Test: %d, %d, %d" % (len(train), len(validation), len(test)))

    parent = uf.get_parent_dir(input_transformations_path)
    train.to_csv(os.path.join(parent, "e3fp_train.csv"), index=False)
    # np.save("ecfp_train.npy", train.to_numpy())
    validation.to_csv(os.path.join(parent, "e3fp_validation.csv"), index=False)
    # np.save("ecfp_validation.npy", train.to_numpy())
    test.to_csv(os.path.join(parent, "e3fp_test.csv"), index=False)
    # np.save("ecfp_test.npy", train.to_numpy())

    return train, validation, test

def save_df_property_encoded(file_name, property_change_encoder, LOG=None):
    data = pd.read_csv(file_name, sep=",")

    #smiles to e3fp
    print('just source smiles to e3fp')
    LOG.info('smiles to e3fp')
    with Pool(NUM_WORKERS) as p:
        data['Source_Mol'] = p.map(e3fp_from_smiles,data['Source_Mol'])
    # data['Target_Mol'] = data['Target_Mol'].map(ECFP_from_smiles)


    for property_name in cfgd.PROPERTIES:
        if property_name == 'LogD':
            encoder, start_map_interval = property_change_encoder[property_name]
            data['Delta_{}'.format(property_name)] = \
                data['Delta_{}'.format(property_name)].apply(lambda x:
                                                                 pce.value_in_interval(x, start_map_interval), encoder)
        elif property_name in ['Solubility', 'Clint']:
            data['Delta_{}'.format(property_name)] = data.apply(
                lambda row: prop_change(row['Source_Mol_{}'.format(property_name)],
                                        row['Target_Mol_{}'.format(property_name)],
                                        cfgd.PROPERTY_THRESHOLD[property_name]), axis=1)

    output_file = file_name.split('.csv')[0] + '_encoded.csv'
    LOG.info("Saving encoded property change to file: {}".format(output_file))
    data.to_csv(output_file, index=False)
    return output_file

def prop_change(source, target, threshold):
    if source <= threshold and target > threshold:
        return "low->high"
    elif source > threshold and target <= threshold:
        return "high->low"
    elif source <= threshold and target <= threshold:
        return "no_change"
    elif source > threshold and target > threshold:
        return "no_change"

    #after data_prep mmp_prop_encoded is the result