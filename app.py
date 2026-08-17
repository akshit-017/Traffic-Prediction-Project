import os
from flask import Flask, render_template, request, jsonify

# ── Auto-train ML model if artefacts are missing ──────────────────────
# The models/ folder is gitignored, so on a fresh Render deploy the
# trained files won't exist.  This block runs main.py's training
# pipeline once so the app starts with a working ML model.
_model_path = os.path.join(os.path.dirname(__file__), 'models', 'random_forest.joblib')
if not os.path.exists(_model_path):
    print("🔧 ML model not found — auto-training before first request...")
    try:
        from main import main as train_main
        train_main()
        print("✅ Auto-training complete.")
    except Exception as e:
        print(f"⚠️ Auto-training failed ({e}). App will use heuristic fallback.")

from src.route_optimization import RouteOptimizer

# Ensure template and static directories exist
os.makedirs('templates', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)

app = Flask(__name__)
optimizer = RouteOptimizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/route', methods=['POST'])
def get_route():
    data = request.json
    source = data.get('source')
    destination = data.get('destination')
    
    if not source or not destination:
        return jsonify({'error': 'Source and destination required'}), 400

    # ── Dynamic predictions: ML model or heuristic fallback ──
    live_predictions = optimizer.generate_dynamic_predictions()
    optimizer.update_edge_weights(live_predictions)
    
    optimal_route, total_weight = optimizer.find_best_route(source, destination)
    graph_data = optimizer.get_graph_data()
    
    if optimal_route is None:
        return jsonify({'error': 'No valid route found between these points.'}), 404
    
    return jsonify({
        'route': optimal_route,
        'total_weight': total_weight,
        'graph': graph_data,
        'ml_model_loaded': optimizer.predictor.is_loaded,
    })

@app.route('/api/model-info')
def model_info():
    """Returns whether the ML model is loaded and powering predictions."""
    return jsonify({
        'ml_model_loaded': optimizer.predictor.is_loaded,
        'model_name': 'Random Forest (Log-Transformed)' if optimizer.predictor.is_loaded else None,
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
