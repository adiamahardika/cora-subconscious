from flask import jsonify, request
from app.utils import response 

def index():
    try:
        page = request.args.get('page', default=1, type=int)
        page_size = request.args.get('page_size', default=10, type=int)

        return response([], None, status="success", message="Hello World")

    except Exception as e:
        # Handle any exceptions (e.g., database errors)
        return response([], None, status="error", message=f"An error occurred: {str(e)}")
