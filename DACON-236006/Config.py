img_size = 224
class_num = 50

print_model = False

epoch = 10
batch_size = 64

learning_rate = 3e-4

data_loader_worker_num = 4

use_fold = False
fold_k = 2

fixed_randomness = True
seed = 2023

test_run = True
test_run_data_size = 1000

train_model_name_list = ['efficientnetb0']
test_input_model_list = ['efficientnetb0_fold_1_result.pt']
