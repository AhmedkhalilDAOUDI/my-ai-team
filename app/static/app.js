const form = document.querySelector('#ask-form');
const button = form.querySelector('button[type="submit"]');
const notice = document.querySelector('#notice');
const synthesis = document.querySelector('#synthesis');
const workflowSelect = document.querySelector('#workflow-select');
const workflowDescription = document.querySelector('#workflow-description');
const pipelineLabel = document.querySelector('#pipeline-label');
const grid = document.querySelector('#provider-grid');
const openaiModel = document.querySelector('#direct-openai-model');
const deepseekModel = document.querySelector('#direct-deepseek-model');
const documentSelect = document.querySelector('#direct-documents');
const runSelect = document.querySelector('#direct-runs');
const stopRun = document.querySelector('#stop-run');
const runExports = document.querySelector('#run-exports');
const colors = ['blue', 'violet', 'gold', 'green', 'coral'];
let currentWorkflow = null;
let runController = null, activeRunId = null;

async function streamWorkflow(body) {
  const response = await fetch(`/api/workflows/${currentWorkflow.id}/stream`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body), signal:runController.signal});
  if (!response.ok) { const data=await response.json(); throw new Error(data.detail||`Request failed (${response.status})`); }
  const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='',complete=null;
  while(true){const {value,done}=await reader.read();buffer+=decoder.decode(value||new Uint8Array(),{stream:!done});const lines=buffer.split('\n');buffer=lines.pop();for(const line of lines){if(!line.trim())continue;const event=JSON.parse(line);if(event.type==='meta'){activeRunId=event.run_id;renderWorkflow(event.workflow)}else if(event.type==='step_start'){const answer=grid.querySelector(`[data-agent-id] .answer`);notice.textContent=`${event.agent_name} is responding…`;const card=[...grid.children][event.position-1];if(card){const box=card.querySelector('.answer');box.className='answer';box.textContent=''}}else if(event.type==='step_delta'){const card=[...grid.children][event.position-1];if(card)card.querySelector('.answer').textContent+=event.text}else if(event.type==='step_done'){const card=[...grid.children][event.turn.position-1];if(card)setAnswer(card.querySelector('.answer'),event.turn)}else if(event.type==='synthesis')setAnswer(synthesis,event.result);else if(event.type==='complete')complete={...event,run_id:activeRunId,workflow:currentWorkflow};else if(event.type==='error')throw new Error(event.error);else if(event.type==='cancelled')throw new DOMException('Run stopped','AbortError')}if(done)break}
  return complete;
}

function showExports(runId) {
  runExports.innerHTML = ['markdown', 'pdf', 'json'].map(format => `<a href="/api/runs/${runId}/export?format=${format}">Export ${format.toUpperCase()}</a>`).join('');
}

async function loadWorkspaceData() {
  const [documents, runs] = await Promise.all([fetch('/api/documents').then(r => r.json()), fetch('/api/runs?limit=50').then(r => r.json())]);
  documentSelect.innerHTML = '';
  documents.forEach(doc => documentSelect.add(new Option(`${doc.filename} · ${doc.characters.toLocaleString()} chars`, doc.id)));
  runSelect.innerHTML = '<option value="">Select a previous run</option>';
  runs.filter(run => run.interface === 'direct').forEach(run => runSelect.add(new Option(`Run ${run.id} · ${run.prompt.slice(0, 55)}`, run.id)));
}

runSelect.addEventListener('change', async () => {
  if (!runSelect.value) return;
  const run = await fetch(`/api/runs/${runSelect.value}`).then(r => r.json());
  const result = run.result || {};
  if (result.workflow) renderWorkflow(result.workflow, result.turns || []);
  if (result.synthesis) setAnswer(synthesis, result.synthesis);
  document.querySelector('#prompt').value = run.prompt;
  showExports(run.id);
});

async function loadModels() {
  const data = await fetch('/api/models').then(response => response.json());
  for (const [provider, select] of [['openai', openaiModel], ['deepseek', deepseekModel]]) {
    select.innerHTML = '';
    data.providers[provider].forEach(item => {
      const option = document.createElement('option');
      option.value = item.id;
      option.textContent = `${item.label} — ${item.description}`;
      select.appendChild(option);
    });
    select.value = data.defaults[provider];
  }
  window.restorePageState?.();
}

function setAnswer(element, result) {
  element.className = 'answer';
  if (result.status === 'ok') element.textContent = result.text;
  else if (result.status === 'error') { element.classList.add('error'); element.textContent = result.error; }
  else { element.classList.add('idle'); element.textContent = result.error || 'This agent is disabled.'; }
}

function agentCard(step, index, result = null) {
  const card = document.createElement('article');
  card.className = 'panel';
  card.dataset.agentId = step.agent_id;
  const head = document.createElement('div');
  head.className = 'panel-head';
  const dot = document.createElement('span');
  dot.className = `dot ${colors[index % colors.length]}`;
  const title = document.createElement('h2');
  title.textContent = step.agent_name;
  const detail = document.createElement('small');
  detail.textContent = `${step.mode} · ${step.model}`;
  const answer = document.createElement('div');
  answer.className = 'answer idle';
  answer.textContent = result ? '' : (step.enabled ? 'Waiting for a question.' : 'Agent is disabled.');
  head.append(dot, title, detail);
  card.append(head, answer);
  if (result) setAnswer(answer, result);
  return card;
}

function renderWorkflow(workflow, turns = null) {
  currentWorkflow = workflow;
  workflowDescription.textContent = workflow.description || 'Saved workflow';
  pipelineLabel.textContent = workflow.steps.map(step => step.agent_name).join(' → ') || 'No workflow steps';
  grid.innerHTML = '';
  workflow.steps.forEach((step, index) => grid.appendChild(agentCard(step, index, turns?.[index])));
  const start=document.querySelector('#start-step'),stop=document.querySelector('#stop-step');
  start.innerHTML='<option value="">First step</option>';stop.innerHTML='<option value="">Last step</option>';
  workflow.steps.forEach(step=>{start.add(new Option(`${step.position}. ${step.agent_name}`,step.position));stop.add(new Option(`${step.position}. ${step.agent_name}`,step.position))});
}

async function selectWorkflow() {
  if (!workflowSelect.value) return;
  try {
    const response = await fetch(`/api/workflows/${workflowSelect.value}`);
    if (!response.ok) throw new Error('Could not load workflow.');
    renderWorkflow(await response.json());
    window.savePageState?.();
  } catch (error) {
    notice.textContent = error.message;
  }
}

async function loadWorkflows() {
  try {
    const workflows = await fetch('/api/workflows').then(response => response.json());
    workflowSelect.innerHTML = '';
    workflows.forEach(workflow => {
      const option = document.createElement('option');
      option.value = workflow.id;
      option.textContent = workflow.name;
      workflowSelect.appendChild(option);
    });
    window.restorePageState?.();
    if (!workflowSelect.value && workflows.length) workflowSelect.value = workflows[0].id;
    if (workflows.length) await selectWorkflow();
    else {
      workflowSelect.innerHTML = '<option value="">No workflows available</option>';
      button.disabled = true;
    }
  } catch {
    notice.textContent = 'Could not load saved workflows.';
    button.disabled = true;
  }
}

async function loadStatus() {
  try {
    const data = await fetch('/api/health').then(response => response.json());
    const missing = [];
    if (!data.providers.openai) missing.push('OpenAI');
    if (!data.providers.deepseek) missing.push('DeepSeek');
    notice.textContent = missing.length ? `Not configured: ${missing.join(', ')}. Add keys in .env to enable them.` : 'All providers are configured.';
  } catch { notice.textContent = 'Could not check provider configuration.'; }
}

workflowSelect.addEventListener('change', selectWorkflow);
form.addEventListener('submit', async event => {
  event.preventDefault();
  if (!currentWorkflow) return;
  button.disabled = true;
  stopRun.hidden = false;
  runController = new AbortController();
  button.textContent = 'Team is thinking…';
  notice.textContent = `Running ${currentWorkflow.name}…`;
  grid.querySelectorAll('.answer').forEach(element => { element.className = 'answer idle'; element.textContent = 'Thinking…'; });
  synthesis.className = 'answer idle';
  synthesis.textContent = 'Preparing synthesis…';
  try {
    const data = await streamWorkflow({prompt: withUploadedFiles(document.querySelector('#prompt').value), models: {openai: openaiModel.value, deepseek: deepseekModel.value}, document_ids: [...documentSelect.selectedOptions].map(option => Number(option.value)), start_position:Number(document.querySelector('#start-step').value)||null, stop_after_position:Number(document.querySelector('#stop-step').value)||null, skip_positions:document.querySelector('#skip-steps').value.split(',').map(x=>Number(x.trim())).filter(Boolean)});
    if (!data) throw new Error('The stream ended before the workflow completed.');
    renderWorkflow(data.workflow, data.turns);
    setAnswer(synthesis, data.synthesis);
    notice.textContent = data.synthesis.status === 'ok' ? `Done · ${data.request_count} API requests.` : 'No final answer was generated. Review the agent messages.';
    showExports(data.run_id);
    await loadWorkspaceData();
    window.savePageState?.();
  } catch (error) {
    synthesis.className = 'answer error';
    synthesis.textContent = error.message;
    notice.textContent = 'The workflow could not be completed.';
  } finally {
    runController = null; activeRunId = null;
    stopRun.hidden = true;
    button.disabled = false;
    button.innerHTML = 'Ask the team <span>→</span>';
  }
});
stopRun.addEventListener('click', async () => { if(activeRunId)await fetch(`/api/runs/${activeRunId}/cancel`,{method:'POST'});runController?.abort();notice.textContent='Run stopped.'; });

openaiModel.addEventListener('change', () => window.savePageState?.());
deepseekModel.addEventListener('change', () => window.savePageState?.());
Promise.all([loadModels(), loadWorkflows(), loadStatus(), loadWorkspaceData()]);
