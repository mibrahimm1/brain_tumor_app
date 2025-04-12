from flask import Flask, render_template, request
from inference import process_image
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image = request.files["image"]
        path = os.path.join("static", image.filename)
        image.save(path)
        result_path = process_image(path)
        return render_template("result.html", result_image=result_path)
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
