from rasa_core.policies import FallbackPolicy, KerasPolicy, MemoizationPolicy
from rasa_core.agent import Agent

def TrainCore():
    fallback = FallbackPolicy(fallback_action_name="utter_unclear",core_threshold=0.2,nlu_threshold=0.1)

    agent = Agent('domain.yml', policies=[MemoizationPolicy(), KerasPolicy(), fallback])
    training_data = agent.load_data('stories.md')

    agent.train(training_data,validation_split=0.0,epochs=500)
    agent.persist('models/dialogue')
