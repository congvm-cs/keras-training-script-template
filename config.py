'''
> Anotational model name:
    ml-{model_name}-ft-{finetune_version}-bm-{basemodel_name}-is-{input_size}-dr-{dropout}-bs-{batch_size}-lr-{learning_rate}
'''
MODEL_NAME = 'ZAGNET'
BASED_MODEL_NAME = 'MobileNet'
FINETUNING_VERSION = 1

# ROOT_IMAGE_DIR = '/data1/LABELING_SERVER_DATABASE/label_system/6f11e1d43be14649b3a1a611e31153cc/'
ROOT_IMAGE_DIR = '/home/zdeploy/AILab/congvm/Workspace/Zalo-AI/Gender_Age/megaage_asian'
# ROOT_MODELS_DIR = '/home/zdeploy/AILab/congvm/Workspace/Zalo-AI/Age-Regression_V1/models_v8/'


NUM_CLASSES = 1
ALPHA = 1.0
SHAPE = (1, 1, int(1280 * ALPHA))
DROPOUT = 0.2

INPUT_SIZE = 128
INPUT_CHANNEL = 3
INPUT_SHAPE = (INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL)
BATCH_SIZE = 128
EPOCHS = 100
INITIAL_EPOCH = 0

AGE_LOSS_WEIGHT = 0.5
GENDER_LOSS_WEIGHT = 1

LOSS_FUNC = {
    'age_out': 'mse',
    'gender_out': 'binary_crossentropy'
}

METRICS = {
    'age_out': 'mse',
    'gender_out': 'binary_accuracy'
}

LEARNING_RATE = 0.05
OPTIMIZER = 'SGD'
WEIGHT_DECAY = 0


