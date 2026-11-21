const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

// Simulate browser boundaries while executing the actual production script.
function browser({ saved = null, dark = false, blocked = false, media = true } = {}) {
  const attributes = {};
  const buttonAttributes = {};
  const handlers = {};
  const buttonHandlers = {};
  const storage = new Map(saved === null ? [] : [['site-theme', saved]]);
  const preference = {
    matches: dark,
    addEventListener(event, callback) { handlers[event] = callback; }
  };
  const button = {
    hidden: true,
    setAttribute(name, value) { buttonAttributes[name] = value; },
    addEventListener(event, callback) { buttonHandlers[event] = callback; }
  };
  const document = {
    documentElement: {
      setAttribute(name, value) { attributes[name] = value; },
      getAttribute(name) { return attributes[name]; }
    },
    readyState: 'loading',
    querySelectorAll() { return [button]; },
    querySelector() { return null; },
    addEventListener(event, callback) { handlers[event] = callback; }
  };
  const window = {
    localStorage: {
      getItem(key) { if (blocked) throw new Error('Storage blocked'); return storage.get(key) ?? null; },
      setItem(key, value) { if (blocked) throw new Error('Storage blocked'); storage.set(key, value); }
    },
    addEventListener(event, callback) { handlers[event] = callback; }
  };
  if (media) window.matchMedia = () => preference;
  // Empty script is the pre-feature baseline: assertions fail on absent behavior.
  const script = fs.existsSync('assets/js/theme.js') ? fs.readFileSync('assets/js/theme.js', 'utf8') : '';
  vm.runInNewContext(script, { window, document });
  return {
    attributes, buttonAttributes, button, storage,
    ready() { handlers.DOMContentLoaded?.(); },
    click() { buttonHandlers.click?.(); },
    system(value) { preference.matches = value; handlers.change?.({ matches: value }); },
    stored(value) { handlers.storage?.({ key: 'site-theme', newValue: value }); }
  };
}

for (const dark of [false, true]) {
  test(`applies system ${dark ? 'dark' : 'light'} before DOM readiness`, () => {
    const page = browser({ dark });
    assert.equal(page.attributes['data-theme'], dark ? 'dark' : 'light');
  });
}
for (const saved of ['light', 'dark']) {
  test(`saved ${saved} overrides the opposite system preference`, () => {
    const page = browser({ saved, dark: saved === 'light' });
    assert.equal(page.attributes['data-theme'], saved);
  });
}
test('invalid stored preference follows system', () => {
  assert.equal(browser({ saved: 'garbage', dark: true }).attributes['data-theme'], 'dark');
});
test('button switches, updates accessible state, persists, and restores on next page', () => {
  const page = browser();
  page.ready();
  assert.equal(page.button.hidden, false);
  assert.equal(page.buttonAttributes['aria-pressed'], 'false');
  assert.equal(page.buttonAttributes['aria-label'], 'Dark mode');
  assert.match(page.buttonAttributes.title, /dark/i);
  page.click();
  assert.equal(page.attributes['data-theme'], 'dark');
  assert.equal(page.buttonAttributes['aria-pressed'], 'true');
  assert.equal(page.buttonAttributes['aria-label'], 'Dark mode');
  assert.match(page.buttonAttributes.title, /light/i);
  assert.equal(page.storage.get('site-theme'), 'dark');
  assert.equal(browser({ saved: page.storage.get('site-theme') }).attributes['data-theme'], 'dark');
  page.click();
  assert.equal(page.attributes['data-theme'], 'light');
  assert.equal(page.storage.get('site-theme'), 'light');
});
test('blocked storage still allows toggling and retains the choice for this page', () => {
  const page = browser({ blocked: true });
  page.ready();
  page.click();
  assert.equal(page.attributes['data-theme'], 'dark');
  page.system(false);
  assert.equal(page.attributes['data-theme'], 'dark');
});
test('system changes follow automatically until the visitor makes a choice', () => {
  const page = browser();
  page.ready();
  page.system(true);
  assert.equal(page.attributes['data-theme'], 'dark');
  assert.equal(page.buttonAttributes['aria-pressed'], 'true');
  page.click();
  page.system(true);
  assert.equal(page.attributes['data-theme'], 'light');
});
test('falls back to light when matchMedia is unavailable', () => {
  assert.equal(browser({ media: false }).attributes['data-theme'], 'light');
});
test('another tab updates the theme; removing its override restores system preference', () => {
  const page = browser({ dark: true });
  page.ready();
  page.stored('light');
  assert.equal(page.attributes['data-theme'], 'light');
  page.stored(null);
  assert.equal(page.attributes['data-theme'], 'dark');
});
