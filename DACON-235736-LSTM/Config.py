input_size = 10
window_size = 28
output_size = 1

print_model = False

epoch = 100
batch_size = 128

learning_rate = 1e-2

data_loader_worker_num = 4

fixed_randomness = True
seed = 2023

test_run = False
test_run_data_size = 100

train_model_name_list = ['simplelstm']
test_input_model_list = ['simplelstm_fold_1_result.pt']
