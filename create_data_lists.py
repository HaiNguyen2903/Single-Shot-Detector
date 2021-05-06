from utils import create_data_lists
import argparse

if __name__ == '__main__':
    # create_data_lists(voc07test_path = './data/test/VOC2007',
    #                   voc07trainval_path = './data/trainval/VOC2007',
    #                   voc12trainval_path = './data/trainval/VOC2012',
    #                   output_folder='./')

                      

    parser = argparse.ArgumentParser(description = 'evaluate arguments')
    parser.add_argument('-te2007', '--test2007', help='voc 2007 test')
    parser.add_argument('-tr2007', '--train2007', help='voc 2007 train')
    parser.add_argument('-tr2012', '--train2012', help='voc 2012 train')
    parser.add_argument('-of', '--outputfolder', help= 'folder to store data')
    args = parser.parse_args()

    create_data_lists(voc07test_path = args.test2007, voc07trainval_path = args.train2007, 
                      voc12trainval_path = args.train2012, output_folder = args.outputfolder)