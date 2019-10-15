
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from rasa_core.actions.action import Action
from rasa_core.events import SlotSet
from ZomatoAPI import ZomatoApi
import MailModule
import json

class ActionSearchRestaurants(Actions):
    pass

class ActionGetCuisines(Action):
    pass

class ActionGetTop10Restaurants(Action):
    pass

class ActionGetAvgBudgetFor2People(Action):
    pass

class ActionSendEmail(Action):
    pass

class ActionGetTypesOfResturants(Action):
    pass

