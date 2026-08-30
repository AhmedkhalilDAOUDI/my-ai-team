(() => {
  if (!document.querySelector('link[href="/static/theme.css"]')) {
    const themeStyles = document.createElement('link'); themeStyles.rel = 'stylesheet'; themeStyles.href = '/static/theme.css?v=1'; document.head.appendChild(themeStyles);
  }
  const themeKey = 'my-ai-team:theme';
  const savedTheme = localStorage.getItem(themeKey) || 'system';
  const effectiveTheme = savedTheme === 'system' ? (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light') : savedTheme;
  document.documentElement.dataset.theme = effectiveTheme;

  const nav = document.querySelector('.nav');
  if (nav) {
    const links = nav.querySelector(':scope > div');
    const extraLinks = links ? [...links.querySelectorAll('a')].filter(link => ['/builder', '/studio'].includes(new URL(link.href).pathname)) : [];
    if (links && extraLinks.length) {
      const more = document.createElement('details');
      more.className = 'nav-more';
      more.innerHTML = '<summary>More</summary><div></div>';
      extraLinks.forEach(link => more.querySelector('div').appendChild(link));
      links.appendChild(more);
    }
    const theme = document.createElement('button');
    theme.type = 'button'; theme.className = 'theme-toggle';
    const updateThemeLabel = () => {
      const dark = document.documentElement.dataset.theme === 'dark';
      theme.textContent = dark ? '☀ Light' : '◐ Dark';
      theme.setAttribute('aria-label', `Switch to ${dark ? 'light' : 'dark'} mode`);
    };
    theme.onclick = () => {
      document.documentElement.dataset.theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
      localStorage.setItem(themeKey, document.documentElement.dataset.theme);
      updateThemeLabel();
    };
    updateThemeLabel(); nav.appendChild(theme);
  }

  const groupAdvanced = (elements, title, description) => {
    const visible = elements.filter(Boolean);
    if (!visible.length) return;
    const details = document.createElement('details'); details.className = 'advanced-panel';
    details.innerHTML = `<summary><span>${title}</span><small>${description}</small></summary><div class="advanced-content"></div>`;
    visible[0].before(details); visible.forEach(element => details.querySelector('.advanced-content').appendChild(element));
  };
  if (location.pathname === '/') groupAdvanced(
    [document.querySelector('.setup-grid'), document.querySelector('.source-importer'), document.querySelector('.jury-config')],
    'Advanced debate setup', 'Formats, evidence, saved debates, benchmarking, and jury controls'
  );
  if (location.pathname === '/workspace') groupAdvanced(
    [...document.querySelectorAll('.direct-model-picker')],
    'Advanced run options', 'Models, documents, history, and partial workflow controls'
  );
  if (location.pathname === '/chat') groupAdvanced(
    [document.querySelector('.model-bar'), document.querySelector('.usage-strip')],
    'Advanced chat options', 'Model selection, usage, and cost controls'
  );
  if (location.pathname === '/studio') {
    const tabs = document.querySelector('.studio-tabs');
    const advancedTabs = tabs ? [...tabs.querySelectorAll('[data-tab]')].filter(button => ['graph', 'integrations', 'settings', 'runs'].includes(button.dataset.tab)) : [];
    if (tabs && advancedTabs.length) {
      const moreTabs = document.createElement('details');
      moreTabs.className = 'studio-more-tabs'; moreTabs.innerHTML = '<summary>Advanced</summary><div></div>';
      advancedTabs.forEach(button => moreTabs.querySelector('div').appendChild(button)); tabs.appendChild(moreTabs);
    }
  }

  const key = `my-ai-team:page:${window.location.pathname}`;
  const outputsToKeep = ['#notice', '#synthesis', '#timeline', '#chat-feed', '#request-count', '#form-status'];
  const fieldKey = (field, index) => field.id || field.name || `field-${index}`;

  function savePageState() {
    const fields = {};
    document.querySelectorAll('input:not([type="file"]):not([type="password"]), textarea, select').forEach((field, index) => {
      fields[fieldKey(field, index)] = {value: field.value, checked: ['checkbox', 'radio'].includes(field.type) ? field.checked : undefined};
    });
    const outputs = {};
    outputsToKeep.forEach(selector => {
      const element = document.querySelector(selector);
      if (element) outputs[selector] = {html: element.innerHTML, className: element.className};
    });
    try { localStorage.setItem(key, JSON.stringify({fields, outputs})); } catch (_) {}
  }

  function restorePageState() {
    let state;
    try { state = JSON.parse(localStorage.getItem(key)); } catch (_) { return; }
    if (!state) return;
    document.querySelectorAll('input:not([type="file"]):not([type="password"]), textarea, select').forEach((field, index) => {
      const saved = state.fields?.[fieldKey(field, index)];
      if (!saved) return;
      field.value = saved.value ?? '';
      if (saved.checked !== undefined) field.checked = saved.checked;
    });
    Object.entries(state.outputs || {}).forEach(([selector, saved]) => {
      const element = document.querySelector(selector);
      if (element) { element.innerHTML = saved.html; element.className = saved.className; }
    });
  }

  window.savePageState = savePageState;
  window.restorePageState = restorePageState;
  window.clearPageState = () => localStorage.removeItem(key);
  restorePageState();
  window.addEventListener('pagehide', savePageState);
  window.addEventListener('beforeunload', savePageState);
})();
