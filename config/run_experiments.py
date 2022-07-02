from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment, Dataset

EXPERIMENT_DIR_PATH = '../modules'
EXPERIMENT_TRAIN_SCRIPT = 'train.py'
ENV_PATH = '../environment.yml'
ENV_NAME = 'img_seg_env_conda'


def connect():
    try:
        return Workspace.from_config()
    except:
        print(f'An exception occured trying to connect to azure ml workspace')


def create_environment_if_not_exists():
    try:
        env = Environment.get(ws, ENV_NAME)
        print('Found existing environment')
    except:
        env = Environment.from_conda_specification(ENV_NAME, ENV_PATH)
        env.register(ws)
        print(f'Registered new environment with name: {ENV_NAME}')
    return env


def register_datasets_if_not_exist(ws):
    data_store = ws.get_default_datastore()

    if 'image dataset' not in ws.datasets:
        data_store.upload_files(files=['../Oxford_pets/images/Abyssinian_81.jpg'],
                                target_path='images/',
                                overwrite=True,
                                show_progress=True)
        data_store.upload_files(files=['../Oxford_pets/annotations/trimaps/Abyssinian_81.png'],
                                target_path='annotations/trimaps/',
                                overwrite=True,
                                show_progress=True)

        # Create a tabular dataset from the path on the datastore (this may take a short while)
        img_file_ds = Dataset.File.from_files(path=(data_store, 'images/*.jpg'))
        mask_file_ds = Dataset.File.from_files(path=(data_store, 'annotations/trimaps/*.png'))

        # Register the tabular dataset
        try:
            img_file_ds = img_file_ds.register(workspace=ws,
                                                 name='image dataset',
                                                 description='images data',
                                                 tags={'format': 'JPG'},
                                                 create_new_version=True)
            print(f'Registered images as dataset')
            mask_file_ds = mask_file_ds.register(workspace=ws,
                                                 name='mask dataset',
                                                 description='image masks data',
                                                 tags={'format': 'PNG'},
                                                 create_new_version=True)
            print(f'Registered masks as dataset')
        except:
            print(f'An exception occured while trying to register the data set')

        return  img_file_ds, mask_file_ds


if __name__ == '__main__':
   ws = connect()

   env = create_environment_if_not_exists()

   img_files_ds, mask_files_ds = register_datasets_if_not_exist(ws)

   script_config = ScriptRunConfig(source_directory=EXPERIMENT_DIR_PATH,
                                   script=EXPERIMENT_TRAIN_SCRIPT,
                                   environment=env,
                                   arguments=[ '--input-data', img_files_ds.as_named_input('training_files').as_download(),
                                               '--masks-data', mask_files_ds.as_named_input('mask_files').as_download()],
                                   compute_target='compute-ml-train')

   experiment = Experiment(workspace=ws, name='img-segmentation-pipe')
   run = experiment.submit(config=script_config)
   run.wait_for_completion(show_output=True)
