// Initialize the Map centered on central Bengaluru
const map = L.map('map').setView([12.94, 77.62], 12);

// Add a dark theme tile layer (CartoDB Dark Matter)
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 20
}).addTo(map);

// Keep track of layers to remove them on new searches
let mapLayers = [];

function clearMap() {
    mapLayers.forEach(layer => map.removeLayer(layer));
    mapLayers = [];
}

function getCongestionColor(congestion) {
    if (congestion < 1.0) return '#2ecc71'; // Green (Free flow)
    if (congestion < 3.0) return '#f1c40f'; // Yellow (Moderate)
    return '#e74c3c'; // Red (Heavy)
}

document.getElementById('route-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const source = document.getElementById('source').value;
    const destination = document.getElementById('destination').value;
    
    if (source === destination) {
        alert("Source and Destination cannot be the same!");
        return;
    }
    
    try {
        const response = await fetch('/api/route', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source, destination })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            alert(data.error || "An error occurred");
            return;
        }
        
        // Render Map Graphics
        renderNetwork(data.graph, data.route);
        
        // Update Results UI
        document.getElementById('optimal-path-text').innerText = data.route.join(' → ');
        document.getElementById('travel-time-text').innerText = data.total_weight.toFixed(2) + ' units';
        document.getElementById('results').classList.remove('hidden');
        
    } catch (error) {
        console.error("Error fetching route:", error);
        alert("Failed to connect to the prediction server.");
    }
});

function renderNetwork(graph, optimalRoute) {
    clearMap();
    
    // Create a dictionary for quick node lookup by ID
    const nodeDict = {};
    graph.nodes.forEach(node => {
        nodeDict[node.id] = node;
        
        // Add Marker
        const marker = L.circleMarker([node.lat, node.lng], {
            radius: 6,
            fillColor: '#3b82f6',
            color: '#fff',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.8
        }).bindTooltip(`${node.name} (${node.id})`, {
            permanent: true,
            direction: 'top',
            className: 'node-label'
        }).addTo(map);
        
        mapLayers.push(marker);
    });
    
    // Draw edges
    graph.edges.forEach(edge => {
        const sourceNode = nodeDict[edge.source];
        const targetNode = nodeDict[edge.target];
        
        if (sourceNode && targetNode) {
            const latlngs = [
                [sourceNode.lat, sourceNode.lng],
                [targetNode.lat, targetNode.lng]
            ];
            
            // Check if this edge is part of the optimal route
            let isOptimal = false;
            for (let i = 0; i < optimalRoute.length - 1; i++) {
                if ((optimalRoute[i] === edge.source && optimalRoute[i+1] === edge.target) ||
                    (optimalRoute[i] === edge.target && optimalRoute[i+1] === edge.source)) {
                    isOptimal = true;
                    break;
                }
            }
            
            const color = getCongestionColor(edge.congestion);
            
            // Draw background line for congestion visualization
            const polyline = L.polyline(latlngs, {
                color: color,
                weight: isOptimal ? 8 : 4,
                opacity: isOptimal ? 0.9 : 0.4,
                dashArray: isOptimal ? null : '5, 10'
            }).addTo(map);
            
            // Add tooltip with weights
            polyline.bindTooltip(`Base: ${edge.base_weight}<br>Congestion: ${edge.congestion.toFixed(2)}`, {
                sticky: true
            });
            
            mapLayers.push(polyline);
            
            // Highlight optimal route with a glowing effect
            if (isOptimal) {
                const glowLine = L.polyline(latlngs, {
                    color: '#fff',
                    weight: 2,
                    opacity: 0.8
                }).addTo(map);
                mapLayers.push(glowLine);
            }
        }
    });
    
    // Fit bounds to show all nodes
    const bounds = L.latLngBounds(graph.nodes.map(n => [n.lat, n.lng]));
    map.fitBounds(bounds, { padding: [50, 50] });
}
