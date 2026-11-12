"""
Cloud AI API - Smart Urban Drainage and Flood Alert System (Group 7)

The ESP32 sends sensor readings (JSON) with HTTP POST to /predict.
This API runs the trained model and sends back its decision (JSON).
The ESP32 then switches its LEDs / buzzer / screen / LINE alert from
this decision.

Run locally:   pip install -r requirements.txt
               API_KEY=mysecret python app.py
"""

import hmac
import json
import math
import os

from flask import Flask, jsonify, request

app = Flask(__name__)
app.json.ensure_ascii = False          # keep Thai text readable in JSON

API_KEY = os.environ.get("API_KEY", "change-me")
FLOOD_CM = 30.0
AI_ALERT_PROB = 0.70                   # same value as the ESP32 backup model

with open(os.path.join(os.path.dirname(__file__), "model.json")) as f:
    MODEL = json.load(f)               # written by train_ai.py


# ---------------------------------------------------------------- the AI
def flood_probability(level, slope, rain):
    """Chance (0..1) that the water reaches 30 cm within 10 minutes."""
    x = [level, slope, rain]
    z = MODEL["b"]
    for i in range(3):
        z += MODEL["w"][i] * (x[i] - MODEL["mean"][i]) / MODEL["std"][i]
    return 1.0 / (1.0 + math.exp(-z))


def eta_minutes(level, slope):
    """Minutes until 30 cm if the water keeps rising at this speed."""
    if slope > 0.05 and level < FLOOD_CM:
        return (FLOOD_CM - level) / slope
    return -1.0


def predictive_risk(prob, eta):
    """0 normal, 1 watch, 2 warning (flood expected soon).
    Level 3 (DANGER) is only set by the ESP32 when the road is really flooded."""
    if prob >= AI_ALERT_PROB and 0 < eta < 10:
        return 2
    if prob >= 0.5:
        return 1
    return 0


def advice(risk, eta):
    if risk >= 2:
        m = max(1, int(round(eta)))
        return (f"คาดว่าน้ำจะท่วมถนน แยก A ในอีกประมาณ {m} นาที "
                "กรุณาเลือกเส้นทางอื่นตั้งแต่ตอนนี้",
                f"Flood expected in about {m} minutes at Intersection A. "
                "Please choose another route now.")
    if risk == 1:
        return ("ระดับน้ำกำลังเพิ่มขึ้น แยก A โปรดระมัดระวัง",
                "Water is rising at Intersection A. Please stay alert.")
    return ("สถานการณ์ปกติ", "Normal conditions.")


# ---------------------------------------------------------------- routes
@app.get("/")
def health():
    return jsonify(status="ok", model=MODEL.get("version", "?"))


@app.post("/predict")
def predict():
    # simple password check so strangers cannot use the API
    key = request.headers.get("X-API-Key", "")
    if not hmac.compare_digest(key, API_KEY):
        return jsonify(error="unauthorized"), 401

    data = request.get_json(silent=True) or {}
    try:
        level = float(data["level"])          # cm
        slope = float(data["rise_rate"])      # cm/min
        rain = float(data["rain"])            # mm/h
    except (KeyError, TypeError, ValueError):
        return jsonify(error="send numbers: level, rise_rate, rain"), 400

    # reject impossible values
    if not (0 <= level <= 60 and -20 <= slope <= 40 and 0 <= rain <= 300):
        return jsonify(error="value out of range"), 400

    prob = flood_probability(level, slope, rain)
    eta = eta_minutes(level, slope)
    risk = predictive_risk(prob, eta)
    th, en = advice(risk, eta)

    return jsonify(
        flood_prob=round(prob, 3),
        eta_min=round(eta, 1),
        risk=risk,
        alert=risk >= 2,
        advice_th=th,
        advice_en=en,
        model=MODEL.get("version", "?"),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
