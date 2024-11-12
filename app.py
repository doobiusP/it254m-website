from flask import Flask
from app.routes import app_routes

app = Flask(__name__, template_folder="templates", static_folder="static")
app.register_blueprint(app_routes)

if __name__ == "__main__":
    app.run(debug=True, port=5001)

