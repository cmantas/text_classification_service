import click, json
from tools.wrappers import create_and_train_text_model, list_saved_models,\
                           predict_with_model


@click.group()
#@click.version_option(version='1.0.0')
def my_main():
    """
    CLI interface for training, saving and using models
    """
    pass

@my_main.command()
def list_models():
    """
    Shows a list of available (saved) models
    """
    models = list_saved_models()
    print('Saved Models Info:')
    present(models)

@my_main.command()
@click.argument('dataset_tsv')
@click.argument('model_name')
@click.option('--epochs', default=1, help='how many epochs to train for')
@click.option('--limit', default=0, help='limit the ammount of training data')
@click.option('--validation-size', default=10_000,
              help='the size of the validation (when training)')
def train(**kwargs):
    """
    Trains a model with a dataset and saves the model file(s)
    """

    train_fname = kwargs['dataset_tsv']
    model_name = kwargs['model_name']
    epochs = kwargs['epochs']
    limit = kwargs['limit']
    val_size = kwargs['validation_size']

    print('training on:', train_fname, 'saved as', model_name, 'with', epochs,
          'epochs', 'limit', limit)
    description = create_and_train_text_model(train_fname, model_name, epochs,
                                              val_size=val_size, limit=limit)

    print('Trained Model Info:')
    present(description)

@my_main.command()
@click.argument('inference_set_file')
@click.option('--limit', default=0, help='limit the ammount of inference data')
@click.argument('model_name')
def predict(**kwargs):
    """
    Predicts the labels for each line of given file based on the given model
    """
    model_name = kwargs['model_name']
    inf_fname = kwargs['inference_set_file']
    limit = kwargs['limit']
    predictions = predict_with_model(model_name, inf_fname, limit=limit)
    present(predictions)



def present(h):
    print(json.dumps(h,indent=2, ensure_ascii=False))

if __name__ == '__main__':
    my_main()
