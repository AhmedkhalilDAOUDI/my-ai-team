(() => {
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
