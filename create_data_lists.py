from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(voc07test_path = './data/test/VOC2007',
                      voc07trainval_path = './data/trainval/VOC2007',
                      voc12trainval_path = './data/trainval/VOC2012',
                      output_folder='./')

                      