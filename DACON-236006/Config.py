img_size = 224
class_num = 50

print_model = False

epoch = 50
batch_size = 128

learning_rate = 5e-4

data_loader_worker_num = 4

use_mixup = False

use_fold = False
fold_k = 2

fixed_randomness = True
seed = 2023

test_run = False
test_run_data_size = 1000

train_model_name_list = ['efficientnetb0']
test_input_model_list = ['efficientnetb0_fold_1_result.pt']
