
# we will train the nlu model here.
from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer, Metadata, Interpreter
from rasa_nlu import config
from rasa_nlu.components import ComponentBuilder

builder = ComponentBuilder(use_cache=True)

def StartModelTraining():
    # loading the nlu training samples
    training_data = load_data("NLU_Train.md")
    # trainer to educate our pipeline
    trainer = Trainer(config.load("NLU_model_Config.yml"))
    # train the model!
    interpreter = trainer.train(training_data)
    # store it for future use
    model_directory = trainer.persist("./models/nlu", fixed_model_name="botModelTrained")

def RunNLU():
    interpreter=Interpreter.load('./models/nlu/default/botModelTrained',builder)
    print(interpreter.parse("Suggest Food"))

if __name__=='__main__':
    StartModelTraining()
    RunNLU()
