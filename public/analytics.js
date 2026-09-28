// Vercel's HTML queue initializer, externalized for script-src 'self'.
// Pageviews only: the app never calls va or sends custom events.
window.va = window.va || function () {
  (window.vaq = window.vaq || []).push(arguments);
};
