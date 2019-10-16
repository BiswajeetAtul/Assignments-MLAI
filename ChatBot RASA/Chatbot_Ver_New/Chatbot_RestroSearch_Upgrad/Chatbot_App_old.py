#this is the main File
#here the program enters the main method and starts the whole process:


from rasa_core.agent import Agent
import Core_Train
import Nlu_model

#start training of the nlu model:
model_directory=StartModelTraining()
RunNLU()
#start training of the core model:
TrainCore()
#load agent:
agent = Agent.load('models/dialogue', interpreter=model_directory)

#bot has been instantiated, and it will start the conversation from here:
print("Your bot is ready to talk! Type your messages here or send 'stop'")
while True:
    a = input()
    if a == 'stop':
        break
    responses = agent.handle_message(a)
    for response in responses:
        print(response["text"])
