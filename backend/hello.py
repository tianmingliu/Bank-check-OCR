from flask_restful import Resource

class Hello(Resource):
    def get(self):
        return {"message": "Hello, Get World!"}
    
    def post(self):
        return {"message": "Hello, Post World!"}