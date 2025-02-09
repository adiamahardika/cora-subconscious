from flask import jsonify, request
from app import db
from app.models.city_model import City
from app.utils import response

def get_all_cities():
    try:
        page = request.args.get('page', default=1, type=int)
        page_size = request.args.get('page_size', default=10, type=int)

        cities = City.query.paginate(page=page, per_page=page_size, error_out=False)
        result = [city.to_dict() for city in cities]

        # If there are no job orders, return error status with a message
        if not result:
            return response([], cities, status="error", message="No data found.")

        # Return successful response
        return response(result, cities, status="success", message="Request successful.")

    except Exception as e:
        return response([], None, status="error", message=f"An error occurred: {str(e)}")

def get_city(city_id):
    try:
        city = City.query.get(city_id)
        if city:
            return response(city.to_dict(), None, status="success", message="City retrieved successfully")
        else:
            return response([], None, status="error", message="City not found")
    except Exception as e:
        return response([], None, status="error", message=f"An error occurred: {str(e)}")

def create_city():
    try:
        data = request.get_json()
        new_city = City(name=data['name'], country=data['country'])
        db.session.add(new_city)
        db.session.commit()
        return response(new_city.to_dict(), None, status="success", message="City created successfully")
    except Exception as e:
        return response([], None, status="error", message=f"An error occurred: {str(e)}")

def update_city(city_id):
    try:
        data = request.get_json()
        city = City.query.get(city_id)
        if city:
            city.name = data['name']
            city.country = data['country']
            db.session.commit()
            return response(city.to_dict(), None, status="success", message="City updated successfully")
        else:
            return response([], None, status="error", message="City not found")
    except Exception as e:
        return response([], None, status="error", message=f"An error occurred: {str(e)}")

def delete_city(city_id):
    try:
        city = City.query.get(city_id)
        if city:
            db.session.delete(city)
            db.session.commit()
            return response([], None, status="success", message="City deleted successfully")
        else:
            return response([], None, status="error", message="City not found")
    except Exception as e:
        return response([], None, status="error", message=f"An error occurred: {str(e)}")