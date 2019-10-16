import requests
import json

class ZomatoApi:
    #Constructor
    #Give your Api Key or use my key by default
    def __init__(self,api_key='ab130b87323aa68eb2747b5fd1c1b188'):
        self.api_key= api_key
        self.api_url="https://developers.zomato.com/api/v2.1/"
        self.header={'Accept': 'application/json','user-key': api_key}
        

    def GetCategories(self):
        '''
        Categories {
            category_id (integer): ID of the category type ,
            category_name (string): Name of the category type
        }
        '''
        response= requests.get(url=self.api_url+"categories",headers=self.header)


    def GetCities(self,city="",city_id=""):
        '''
        City {
            id (integer): ID of the city ,
            name (string): City name ,
            country_id (integer, optional): ID of the country ,
            country_name (string, optional): Name of the country ,
            is_state (boolean, optional): Whether this location is a state ,
            state_id (integer, optional): ID of the state ,
            state_name (string, optional): Name of the state ,
            state_code (string, optional): Short code for the state
        }
        '''

        task={"q": city, "city_ids": city_id}
        if(city==""): #search by city id
            task={"city_ids": city_id}
        elif(city_id==""): #search by city name
            task={"q": city}

        #response= requests.get(self.api_url+"cities",headers=header,params=task)
        response=requests.get(url=self.api_url+"cities",headers=self.header,params=task)
        
    
    #def GetCollections(self):
    #    response= requests.get(self.api_url+"collections")
    #    pass

    
    def GetCuisines(self, city_id):
        '''
        Cuisine {
            cuisine_id (integer): ID of the cuisine ,
            cuisine_name (string): Name of the cuisine
        }

        '''
        task={"city_id":city_id}
        response= requests.get(self.api_url+"cuisines",headers=self.header,params=task)
        

    def GetEstablishmentTypes(self):
        '''
        Establishment {
            establishment_id (integer): ID of the establishment type ,
            establishment_name (string): Name of the establishment type
        }
        '''
        task={}
        response= requests.get(self.api_url+"establishments",headers=self.header,params=task)
        
    
    def GetFoodieNightLifeIndex(self):
        # Finds NightLife Index
        # Using location_details
        task={}
        response= requests.get(self.api_url+"location_details",headers=self.header,params=task)
        

    def GetLocations(self,location_id, location_type):
        #Can be used to obtain the list of best resturants.
        '''
        Location {
            entity_type (string, optional): Type of location; one of [city, zone, subzone, landmark, group, metro, street] ,
            entity_id (integer, optional): ID of location; (entity_id, entity_type) tuple uniquely identifies a location ,
            title (string, optional): Name of the location ,
            latitude (number, optional): Coordinates of the (centre of) location ,
            longitude (number, optional): Coordinates of the (centre of) location ,
            city_id (integer, optional): ID of city ,
            city_name (string, optional): Name of the city ,
            country_id (integer, optional): ID of country ,
            country_name (string, optional): Name of the country
        }

        '''
        task={"entity_id":location_id, "entity_type":location_type}
        response= requests.get(self.api_url+"locations",headers=self.header,params=task)
        

    
    def GetMenuOfRestro(self):
        '''
        DailyMenu {
            daily_menu (Array[DailyMenuCategory], optional): List of restaurant's menu details
        }
        DailyMenuCategory {
            daily_menu_id (integer, optional): ID of the restaurant ,
            name (string, optional): Name of the restaurant ,
            start_date (string, optional): Daily Menu start timestamp ,
            end_date (string, optional): Daily Menu end timestamp ,
            dishes (Array[DailyMenuItem], optional): Menu item in the category
        }
        DailyMenuItem {
            dish_id (integer, optional): Menu Item ID ,
            name (string, optional): Menu Item Title ,
            price (string, optional): Menu Item Price
        }
        '''
        task={}
        response= requests.get(self.api_url+"dailymenu",headers=self.header,params=task)
        
    
    def GetRestroInfo(self):
        task={}
        response= requests.get(self.api_url+"restaurant",headers=self.header,params=task)
        

    def SearchRestros(self,restro_id):
        '''
        RestaurantL3 {
            id (integer, optional): ID of the restaurant ,
            name (string, optional): Name of the restaurant ,
            url (string, optional): URL of the restaurant page ,
            location (ResLocation, optional): Restaurant location details ,
            average_cost_for_two (integer, optional): Average price of a meal for two people ,
            price_range (integer, optional): Price bracket of the restaurant (1 being pocket friendly and 4 being the costliest) ,
            currency (string, optional): Local currency symbol; to be used with price ,
            thumb (string, optional): URL of the low resolution header image of restaurant ,
            featured_image (string, optional): URL of the high resolution header image of restaurant ,
            photos_url (string, optional): URL of the restaurant's photos page ,
            menu_url (string, optional): URL of the restaurant's menu page ,
            events_url (string, optional): URL of the restaurant's events page ,
            user_rating (UserRating, optional): Restaurant rating details ,
            has_online_delivery (boolean, optional): Whether the restaurant has online delivery enabled or not ,
            is_delivering_now (boolean, optional): Valid only if has_online_delivery = 1; whether the restaurant is accepting online orders right now ,
            has_table_booking (boolean, optional): Whether the restaurant has table reservation enabled or not ,
            deeplink (string, optional): Short URL of the restaurant page; for use in apps or social shares ,
            cuisines (string, optional): List of cuisines served at the restaurant in csv format ,
            all_reviews_count (integer, optional): [Partner access] Number of reviews for the restaurant ,
            photo_count (integer, optional): [Partner access] Total number of photos for the restaurant, at max 10 photos for partner access ,
            phone_numbers (string, optional): [Partner access] Restaurant's contact numbers in csv format ,
            photos (Array[Photo], optional): [Partner access] List of restaurant photos ,
            all_reviews (Array[Review], optional): [Partner access] List of restaurant reviews
        }
        ResLocation {
            address (string, optional): Complete address of the restaurant ,
            locality (string, optional): Name of the locality ,
            city (string, optional): Name of the city ,
            latitude (number, optional): Coordinates of the restaurant ,
            longitude (number, optional): Coordinates of the restaurant ,
            zipcode (string, optional): Zipcode ,
            country_id (integer, optional): ID of the country
        }
        UserRating {
            aggregate_rating (number, optional): Restaurant rating on a scale of 0.0 to 5.0 in increments of 0.1 ,
            rating_text (string, optional): Short description of the rating ,
            rating_color (string, optional): Color hex code used with the rating on Zomato ,
            votes (integer, optional): Number of ratings received
        }
        Photo {
            id (string, optional): ID of the photo ,
            url (string, optional): URL of the image file ,
            thumb_url (string, optional): URL for 200 X 200 thumb image file ,
            user (User, optional): User who uploaded the photo ,
            res_id (integer, optional): ID of restaurant for which the image was uploaded ,
            caption (string, optional): Caption of the photo ,
            timestamp (integer, optional): Unix timestamp when the photo was uploaded ,
            friendly_time (string, optional): User friendly time string; denotes when the photo was uploaded ,
            width (integer, optional): Image width in pixel; usually 640 ,
            height (integer, optional): Image height in pixel; usually 640 ,
            comments_count (integer, optional): Number of comments on photo ,
            likes_count (integer, optional): Number of likes on photo
        }
        Review {
            rating (number, optional): Rating on scale of 0 to 5 in increments of 0.5 ,
            review_text (string, optional): Review text ,
            id (integer, optional): ID of the review ,
            rating_color (string, optional): Color hex code used with the rating on Zomato ,
            review_time_friendly (string, optional): User friendly time string corresponding to time of review posting ,
            rating_text (string, optional): Short description of the rating ,
            timestamp (integer, optional): Unix timestamp for review_time_friendly ,
            likes (integer, optional): No of likes received for review ,
            user (User, optional): User details of author of review ,
            comments_count (integer, optional): No of comments on review
        }
        User {
            name (string, optional): User's name ,
            zomato_handle (string, optional): User's @handle; uniquely identifies a user on Zomato ,
            foodie_level (string, optional): Text for user's foodie level ,
            foodie_level_num (integer, optional): Number to identify user's foodie level; ranges from 0 to 10 ,
            foodie_color (string, optional): Color hex code used with foodie level on Zomato ,
            profile_url (string, optional): URL for user's profile on Zomato ,
            profile_deeplink (string, optional): short URL for user's profile on Zomato; for use in apps or social sharing ,
            profile_image (string, optional): URL for user's profile image
        }
        '''

        task={"res_id":restro_id}
        response= requests.get(self.api_url+"reviews",headers=self.header,params=task)
        

    def GetReviews(self, restro_id):
        '''
        Inline Model [
            Review
        ]
        Review {
            rating (number, optional): Rating on scale of 0 to 5 in increments of 0.5 ,
            review_text (string, optional): Review text ,
            id (integer, optional): ID of the review ,
            rating_color (string, optional): Color hex code used with the rating on Zomato ,
            review_time_friendly (string, optional): User friendly time string corresponding to time of review posting ,
            rating_text (string, optional): Short description of the rating ,
            timestamp (integer, optional): Unix timestamp for review_time_friendly ,
            likes (integer, optional): No of likes received for review ,
            user (User, optional): User details of author of review ,
            comments_count (integer, optional): No of comments on review
        }
        User {
            name (string, optional): User's name ,
            zomato_handle (string, optional): User's @handle; uniquely identifies a user on Zomato ,
            foodie_level (string, optional): Text for user's foodie level ,
            foodie_level_num (integer, optional): Number to identify user's foodie level; ranges from 0 to 10 ,
            foodie_color (string, optional): Color hex code used with foodie level on Zomato ,
            profile_url (string, optional): URL for user's profile on Zomato ,
            profile_deeplink (string, optional): short URL for user's profile on Zomato; for use in apps or social sharing ,
            profile_image (string, optional): URL for user's profile image
        }
        '''
        task={"res_id":restro_id}
        response= requests.get(self.api_url+"search",headers=self.header,params=task)
        