import os
import argparse
import random
from sys import path

path.append(os.getcwd())
from experiments.common_utils import dump_rows
from data_utils.task_def import DataFormat
from data_utils.log_wrapper import create_logger
from experiments.glue.glue_utils import *

logger = create_logger(__name__, to_disk=True, log_file='glue_prepro.log')


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing pawsx/lcqmc/bq dataset.')

    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--root_dir', type=str, default='data')
    parser.add_argument('--old_glue', action='store_true', help='whether it is old GLUE, refer official GLUE webpage for details')
    args = parser.parse_args()
    return args


def main(args):
    is_old_glue = args.old_glue
    root = args.root_dir
    assert os.path.exists(root)


    ######################################
    # GLUE tasks
    ######################################
    lcqmc_train_path = os.path.join(root, 'LCQMC/train.tsv')
    lcqmc_dev_path = os.path.join(root, 'LCQMC/dev.tsv')
    lcqmc_test_path = os.path.join(root, 'LCQMC/test.tsv')

    bq_train_path = os.path.join(root, 'BQ/train.tsv')
    bq_dev_path = os.path.join(root, 'BQ/dev.tsv')
    bq_test_path = os.path.join(root, 'BQ/test.tsv')

    pawsx_train_path = os.path.join(root, 'PAWSX/train.tsv')
    pawsx_dev_path = os.path.join(root, 'PAWSX/dev.tsv')
    pawsx_test_path = os.path.join(root, 'PAWSX/test.tsv')

    ######################################
    # Loading DATA
    ######################################

    lcqmc_train_data = load_test(lcqmc_train_path, is_train=True)
    lcqmc_dev_data = load_test(lcqmc_dev_path, is_train=True)
    lcqmc_test_data = load_test(lcqmc_test_path, is_train=False)
    logger.info('Loaded {} LCQMC train samples'.format(len(lcqmc_train_data)))
    logger.info('Loaded {} LCQMC dev samples'.format(len(lcqmc_dev_data)))
    logger.info('Loaded {} LCQMC test samples'.format(len(lcqmc_test_data)))

    bq_train_data = load_test(bq_train_path, is_train=True)
    bq_dev_data = load_test(bq_dev_path, is_train=True)
    bq_test_data = load_test(bq_test_path, is_train=False)
    logger.info('Loaded {} BQ train samples'.format(len(bq_train_data)))
    logger.info('Loaded {} BQ dev samples'.format(len(bq_dev_data)))
    logger.info('Loaded {} BQ test samples'.format(len(bq_test_data)))

    pawsx_train_data = load_test(pawsx_train_path, is_train=True)
    pawsx_dev_data = load_test(pawsx_dev_path, is_train=True)
    pawsx_test_data = load_test(pawsx_test_path, is_train=False)
    logger.info('Loaded {} PAWSX train samples'.format(len(pawsx_train_data)))
    logger.info('Loaded {} PAWSX dev samples'.format(len(pawsx_dev_data)))
    logger.info('Loaded {} PAWSX test samples'.format(len(pawsx_test_data)))
    canonical_data_suffix = "canonical_data"
    canonical_data_root = os.path.join(root, canonical_data_suffix)
    if not os.path.isdir(canonical_data_root):
        os.mkdir(canonical_data_root)

    lcqmc_train_fout = os.path.join(canonical_data_root, 'lcqmc_train.tsv')
    lcqmc_dev_fout = os.path.join(canonical_data_root, 'lcqmc_dev.tsv')
    lcqmc_test_fout = os.path.join(canonical_data_root, 'lcqmc_test.tsv')
    dump_rows(lcqmc_train_data, lcqmc_train_fout, DataFormat.SimPair)
    dump_rows(lcqmc_dev_data, lcqmc_dev_fout, DataFormat.SimPair)
    dump_rows(lcqmc_test_data, lcqmc_test_fout, DataFormat.SimPairTest)
    logger.info('done with lcqmc')

    bq_train_fout = os.path.join(canonical_data_root, 'bq_train.tsv')
    bq_dev_fout = os.path.join(canonical_data_root, 'bq_dev.tsv')
    bq_test_fout = os.path.join(canonical_data_root, 'bq_test.tsv')
    dump_rows(bq_train_data, bq_train_fout, DataFormat.SimPair)
    dump_rows(bq_dev_data, bq_dev_fout, DataFormat.SimPair)
    dump_rows(bq_test_data, bq_test_fout, DataFormat.SimPairTest)
    logger.info('done with bq')

    pawsx_train_fout = os.path.join(canonical_data_root, 'pawsx_train.tsv')
    pawsx_dev_fout = os.path.join(canonical_data_root, 'pawsx_dev.tsv')
    pawsx_test_fout = os.path.join(canonical_data_root, 'pawsx_test.tsv')
    dump_rows(pawsx_train_data, pawsx_train_fout, DataFormat.SimPair)
    dump_rows(pawsx_dev_data, pawsx_dev_fout, DataFormat.SimPair)
    dump_rows(pawsx_test_data, pawsx_test_fout, DataFormat.SimPairTest)
    logger.info('done with pawsx')

if __name__ == '__main__':
    args = parse_args()
    main(args)
