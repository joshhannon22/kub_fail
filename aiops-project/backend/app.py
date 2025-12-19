from flask import Flask, jsonify, request
import logging
from pythonjsonlogger import jsonlogger
import time
import random
import redis
import os
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Setup JSON logging
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Redis connection
redis_host = os.getenv('REDIS_HOST', 'redis-service')
redis_client = redis.Redis(host=redis_host, port=6379, decode_responses=True)

@app.route('/api/health')
def health():
    logger.info("Health check called")
    return jsonify({"status": "healthy"}), 200

@app.route('/api/data')
def get_data():
    logger.info("Data endpoint called")
    try:
        # Increment counter in Redis
        count = redis_client.incr('api_calls')
        data = {
            "message": "Hello from backend",
            "call_count": count,
            "timestamp": time.time()
        }
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error in /api/data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/heavy')
def heavy_computation():
    logger.info("Heavy computation started")
    start = time.time()
    
    # Simulate heavy CPU load
    result = 0
    for i in range(1000000):
        result += i ** 2
    
    duration = time.time() - start
    logger.info(f"Heavy computation completed in {duration:.2f}s")
    
    return jsonify({
        "result": result,
        "duration": duration,
        "message": "Heavy computation completed"
    }), 200

@app.route('/api/error')
def trigger_error():
    logger.error("Intentional error triggered")
    if random.random() > 0.5:
        raise Exception("Random error occurred!")
    return jsonify({"message": "No error this time"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)