img_size = 224
class_num = 10

epoch = 2
batch_size = 20

learning_rate = 1e-4

data_loader_worker_num = 2
fold_k = 2

fixed_randomness = True
seed = 2023

test_run = True
test_run_data_size = 1000

train_model_name_list = ['alexnet1', 'alexnet2']
test_input_model_list = ['alexnet1_fold_1_result.pt', 'alexnet2_fold_2_result.pt']
