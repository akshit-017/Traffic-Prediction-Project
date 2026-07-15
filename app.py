import os
from flask import Flask, render_template, request, jsonify
from src.route_optimization import RouteOptimizer

# Ensure template and static directories exist
os.makedirs('templates', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)

app = Flask(__name__)
optimizer = RouteOptimizer()

# Simulated predictions from the Random Forest Model for the next 30 mins
# In a real system, this would be periodically updated by the ML model.
rf_predictions = {
    ('A', 'B'): 2.5,
    ('A', 'C'): 1.0,
    ('B', 'C'): 5.0, # Heavy congestion
    ('B', 'D'): 0.5,
    ('C', 'E'): 1.5,
    ('D', 'E'): 3.0,
    ('D', 'F'): 0.2,
    ('E', 'F'): 4.0, # Heavy congestion
    ('E', 'G'): 1.0,
    ('F', 'G'): 0.5
}

# Apply the simulated ML predictions to the road network
optimizer.update_edge_weights(rf_predictions)

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
        
    optimal_route, total_weight = optimizer.find_best_route(source, destination)
    graph_data = optimizer.get_graph_data()
    
    if optimal_route is None:
        return jsonify({'error': 'No valid route found between these points.'}), 404
    
    return jsonify({
        'route': optimal_route,
        'total_weight': total_weight,
        'graph': graph_data
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
