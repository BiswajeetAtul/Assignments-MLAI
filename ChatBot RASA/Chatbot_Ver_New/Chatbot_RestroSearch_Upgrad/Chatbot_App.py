import logging, io, json, warnings
#logging.basicConfig(level="INFO")
warnings.filterwarnings('ignore')
import rasa_nlu
import rasa_core
import spacy
from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config

from rasa_core.actions import Action
from rasa_core.events import SlotSet
from rasa_core.policies import FallbackPolicy, KerasPolicy, MemoizationPolicy
from rasa_core.agent import Agent

print("Initializing the ChatBot:")

print("STEP 1:Training the NLU Model")
#Training the NLU MODEL:
# loading the nlu training samples
training_data = load_data("NLU_Train.md")
# trainer to create the pipeline
trainer = Trainer(config.load("NLU_model_Config.yml"))
# training the model
interpreter = trainer.train(training_data)
# storeing it for future
model_directory = trainer.persist("./models/nlu", fixed_model_name="current")
print("Done")

print("STEP 2: Training the CORE model")
fallback = FallbackPolicy(fallback_action_name="utter_default",
                          core_threshold=0.2,
                          nlu_threshold=0.1)

agent = Agent('restaurant_domain.yml', policies=[MemoizationPolicy(), KerasPolicy(), fallback])
training_data = agent.load_data('Core_Stories.md')
agent.train(
    training_data,
    validation_split=0.0,
    epochs=200
)
agent.persist('models/dialogue')
print("Done")
print("STEP 3: Starting the Bot")
from rasa_core.agent import Agent
agent = Agent.load('models/dialogue', interpreter=model_directory)

print("Your bot is ready to talk! Type your messages here or send 'stop'")
while True:
    a = input()
    if a == 'stop':
        break
    responses = agent.handle_message(a)
    for response in responses:
        print(response["text"])