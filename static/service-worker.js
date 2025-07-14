const CACHE_NAME = 'ai-music-gen-v1';
const urlsToCache = [
  '/',
  '/static/style.css',
  '/static/manifest.json',
  // Add more static assets if needed
];

self.addEventListener('install', function(event) {
  event.waitUntil(
    caches.open(CACHE_NAME).then(function(cache) {
      return cache.addAll(urlsToCache);
    })
  );
});

self.addEventListener('fetch', function(event) {
  event.respondWith(
    caches.match(event.request).then(function(response) {
      return response || fetch(event.request);
    })
  );
}); 