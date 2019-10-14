import requests

class ZomatoApi:
    #Constructor
    #Give your Api Key or use my key by default
    def __init__(self,api_key='ab130b87323aa68eb2747b5fd1c1b188'):
        self.api_key= api_key
        self.api_url="https://developers.zomato.com/api/v2.1/"
        

    def GetCategories(self):
        response= requests.get(self.api_url+"categories")
        pass

    def GetCities(self):
        response= requests.get(self.api_url+"cities")
        pass
    
    def GetCollections(self):
        response= requests.get(self.api_url+"collections")
        pass

    def GetCuisines(self):
        response= requests.get(self.api_url+"cuisines")
        pass

    def GetEstablishmentTypes(self):
        response= requests.get(self.api_url+"establishments")
        pass
    
    def GetFoodieNightLifeIndex(self):
        # Finds NightLife Index
        # Using location_details
        response= requests.get(self.api_url+"location_details")
        pass

    def GetLocations(self):
        response= requests.get(self.api_url+"locations")
        pass

    def GetMenuOfRestro(self):
        response= requests.get(self.api_url+"dailymenu")
        pass

    def GetRestroInfo(self):
        response= requests.get(self.api_url+"restaurant")
        pass

    def SearchRestros(self):
        response= requests.get(self.api_url+"reviews")
        pass

    def GetReviews(self):
        response= requests.get(self.api_url+"search")
        pass