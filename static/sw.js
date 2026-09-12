// ================================================================
// Service Worker — Offline Tile Cache
// Caches OpenStreetMap tiles so the map remains visible when offline.
// Strategy: cache-first for tiles, network-first for everything else.
// ================================================================

const TILE_CACHE = 'map-tiles-v1';
const TILE_HOST  = 'tile.openstreetmap.org';

// Install — activate immediately
self.addEventListener('install', (event) => {
    self.skipWaiting();
});

// Activate — claim all clients immediately
self.addEventListener('activate', (event) => {
    event.waitUntil(self.clients.claim());
});

// Fetch — intercept tile requests
self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);

    // Only cache-first for OSM tile requests
    if (url.hostname.endsWith(TILE_HOST)) {
        event.respondWith(
            caches.open(TILE_CACHE).then((cache) => {
                return cache.match(event.request).then((cachedResponse) => {
                    if (cachedResponse) {
                        // Serve from cache, but also update cache in background
                        fetch(event.request)
                            .then((networkResponse) => {
                                if (networkResponse && networkResponse.ok) {
                                    cache.put(event.request, networkResponse);
                                }
                            })
                            .catch(() => { /* offline — ignore */ });
                        return cachedResponse;
                    }

                    // Not in cache — fetch from network and cache it
                    return fetch(event.request)
                        .then((networkResponse) => {
                            if (networkResponse && networkResponse.ok) {
                                cache.put(event.request, networkResponse.clone());
                            }
                            return networkResponse;
                        })
                        .catch(() => {
                            // Offline and not cached — return transparent pixel
                            return new Response(
                                Uint8Array.from(atob(
                                    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='
                                ), c => c.charCodeAt(0)),
                                { headers: { 'Content-Type': 'image/png' } }
                            );
                        });
                });
            })
        );
        return;
    }

    // All other requests — network first (normal behaviour)
    event.respondWith(
        fetch(event.request).catch(() => caches.match(event.request))
    );
});
