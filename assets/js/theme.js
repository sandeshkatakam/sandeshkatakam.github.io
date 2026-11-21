/* Run synchronously before CSS so every page starts in the correct theme. */
(function () {
  'use strict';

  var key = 'site-theme';
  var root = document.documentElement;
  var media = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
  var preference = null;
  var buttons = [];

  function valid(value) {
    return value === 'light' || value === 'dark' ? value : null;
  }

  try {
    preference = valid(window.localStorage.getItem(key));
  } catch (error) {
    // Private browsing and storage restrictions must not disable the control.
  }

  function apply() {
    var theme = preference || (media && media.matches ? 'dark' : 'light');
    var dark = theme === 'dark';
    root.setAttribute('data-theme', theme);
    for (var i = 0; i < buttons.length; i++) {
      var label = 'Switch to ' + (dark ? 'light' : 'dark') + ' mode';
      buttons[i].setAttribute('aria-label', 'Dark mode');
      buttons[i].setAttribute('title', label);
      buttons[i].setAttribute('aria-pressed', String(dark));
    }
    var themeColor = document.querySelector('meta[name="theme-color"]');
    if (themeColor) themeColor.setAttribute('content', dark ? '#202020' : '#fafafa');
  }

  apply();

  function initialize() {
    buttons = document.querySelectorAll('[data-theme-toggle]');
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].hidden = false;
      buttons[i].addEventListener('click', function () {
        preference = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
        try {
          window.localStorage.setItem(key, preference);
        } catch (error) {
          // Keep this page's manual preference even if it cannot be saved.
        }
        apply();
      });
    }
    apply();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize);
  } else {
    initialize();
  }

  if (media) {
    if (media.addEventListener) media.addEventListener('change', apply);
    else if (media.addListener) media.addListener(apply);
  }

  window.addEventListener('storage', function (event) {
    if (event.key === key || event.key === null) {
      preference = valid(event.newValue);
      apply();
    }
  });
}());
