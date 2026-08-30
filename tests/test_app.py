import asyncio
import subprocess
import uuid
import pytest
from fastapi.testclient import TestClient
from app.main import app, store
from app.config import Settings
from app.orchestrator import ProviderResult, ThesisTeam

client=TestClient(app)

def test_pages_and_health():
    assert all(client.get(path).status_code==200 for path in ("/","/discussion","/chat","/builder","/studio"))
    assert set(client.get("/api/health").json()["providers"])=={"openai","deepseek","jury","auditor"}

def test_every_interface_keeps_its_browser_state():
    for path in ("/", "/discussion", "/chat", "/studio"):
        assert '/static/page-state.js' in client.get(path).text
    chat_script=client.get('/static/chat.js').text
    assert 'my-ai-team:chat-runtime' in chat_script
    assert 'Array.isArray(saved.history)' in chat_script
    assert 'conversationId' in chat_script and 'conversation-select' in client.get('/chat').text

def test_fixed_pipeline(monkeypatch):
    calls=[]
    async def fake(self,provider,prompt,system_prompt=None):
        calls.append((provider.name,prompt));return ProviderResult(status="ok",text=f"Reply from {provider.name}")
    monkeypatch.setattr(ThesisTeam,"_safe_ask",fake)
    body=client.post("/api/ask",json={"prompt":"Evaluate","route":"all"}).json()
    assert all(body["responses"][name]["status"]=="ok" for name in ("openai","deepseek","jury","auditor"))
    assert [x[0] for x in calls[:4]]==["ChatGPT / OpenAI","DeepSeek","Jury / OpenAI GPT-5.6 Sol","Completeness Auditor / DeepSeek V4 Pro"]
    assert "SUPERVISOR ANSWER" in calls[1][1] and "CRITIC REVIEW" in calls[2][1] and "JURY QUESTIONS" in calls[3][1]

def test_discussion_and_live_chat_context(monkeypatch):
    team=ThesisTeam(Settings(openai_api_key="x",deepseek_api_key="x"));captured={}
    async def fake(self,provider,prompt,system_prompt=None):captured[provider.name]=prompt;return ProviderResult(status="ok",text="reply")
    async def synth(self,prompt,answers):return ProviderResult(status="ok",text="combined")
    monkeypatch.setattr(ThesisTeam,"_safe_ask",fake);monkeypatch.setattr(ThesisTeam,"_synthesize",synth)
    turns,_,count=asyncio.run(team.discuss("task",1));assert len(turns)==6 and count==7
    asyncio.run(team.chat_reply("deepseek","review",[{"speaker":"openai","content":"proposal"}]))
    assert "ChatGPT: proposal" in captured["DeepSeek"]

def test_upload():
    r=client.post("/api/upload",files={"file":("notes.txt",b"evidence","text/plain")})
    assert r.status_code==200 and r.json()["text"]=="evidence"

def test_studio_persistence_and_default_workflow():
    agents=client.get("/api/agents").json()
    assert {"Supervisor","DeepSeek Critic","Jury","Completeness Auditor"}.issubset({a["name"] for a in agents})
    created=client.post("/api/agents",json={"name":"Test Analyst","provider":"deepseek","model":"deepseek-v4-pro","role":"Test role","instructions":"Be precise","max_sentences":4,"can_read_peers":True,"enabled":True})
    assert created.status_code==201
    agent=created.json();agent["role"]="Updated role"
    updated=client.put(f"/api/agents/{agent['id']}",json={key:agent[key] for key in ("name","provider","model","role","instructions","max_sentences","can_read_peers","enabled")})
    assert updated.status_code==200 and updated.json()["role"]=="Updated role"
    workflows=client.get("/api/workflows").json();detail=client.get(f"/api/workflows/{workflows[0]['id']}").json()
    assert len(detail["steps"])==4
    assert client.delete(f"/api/agents/{agent['id']}").status_code==204

def test_saved_workflow_drives_runtime(monkeypatch):
    team=ThesisTeam(Settings(openai_api_key="x",deepseek_api_key="x"));calls=[]
    workflow={"id":99,"name":"Custom","steps":[
        {"id":1,"position":1,"mode":"respond","agent_id":10,"agent_name":"Architect","provider":"openai","model":"custom-openai","role":"Design the system","instructions":"Choose one approach","max_sentences":3,"can_read_peers":1,"enabled":1},
        {"id":2,"position":2,"mode":"critique","agent_id":11,"agent_name":"Reviewer","provider":"deepseek","model":"custom-deepseek","role":"Find blocking flaws","instructions":"Be strict","max_sentences":2,"can_read_peers":1,"enabled":1},
    ]}
    async def fake(self,provider,prompt,system_prompt=None):
        calls.append((provider.name,provider.model,prompt,system_prompt));return ProviderResult(status="ok",text=f"answer {len(calls)}")
    async def synth(self,prompt,answers):return ProviderResult(status="ok",text="final")
    monkeypatch.setattr(ThesisTeam,"_safe_ask",fake);monkeypatch.setattr(ThesisTeam,"_synthesize",synth)
    turns,result,count=asyncio.run(team.run_workflow("build it",workflow))
    assert result.text=="final" and count==3 and len(turns)==2
    assert calls[0][1]=="custom-openai" and "Design the system" in calls[0][3] and "no more than 3" in calls[0][3]
    assert calls[1][1]=="custom-deepseek" and "Architect (respond):\nanswer 1" in calls[1][2]

def test_workflow_run_endpoint_and_validation(monkeypatch):
    async def fake(self,prompt,workflow):
        return ([{"agent_name":"A","status":"ok","text":"done"}],ProviderResult(status="ok",text="final"),2)
    monkeypatch.setattr(ThesisTeam,"run_workflow",fake)
    workflow_id=client.get('/api/workflows').json()[0]['id']
    response=client.post(f'/api/workflows/{workflow_id}/run',json={"prompt":"Execute"})
    assert response.status_code==200 and response.json()["synthesis"]["text"]=="final"
    store.delete_run(response.json()['run_id'])
    assert client.post('/api/workflows/999999/run',json={"prompt":"Execute"}).status_code==404
    invalid=client.put(f'/api/workflows/{workflow_id}/steps',json=[{"agent_id":999999,"mode":"respond"}])
    assert invalid.status_code==400

def test_named_conversation_lifecycle_and_server_context(monkeypatch):
    created=client.post('/api/conversations',json={"title":"Research chat","interface":"chat"})
    assert created.status_code==201
    conversation_id=created.json()['id']
    try:
        first=client.post(f'/api/conversations/{conversation_id}/messages',json={"speaker":"user","content":"Initial question"})
        second=client.post(f'/api/conversations/{conversation_id}/messages',json={"speaker":"openai","content":"Initial answer"})
        assert first.status_code==201 and second.status_code==201
        detail=client.get(f'/api/conversations/{conversation_id}').json()
        assert [message['content'] for message in detail['messages']]==["Initial question","Initial answer"]
        captured={}
        async def fake(self,provider_name,prompt,history):
            captured['history']=history;return ProviderResult(status="ok",text="continued")
        monkeypatch.setattr(ThesisTeam,'chat_reply',fake)
        reply=client.post('/api/chat/reply',json={"provider":"deepseek","prompt":"Continue","conversation_id":conversation_id})
        assert reply.status_code==200 and captured['history'][-1]['content']=="Initial answer"
        renamed=client.patch(f'/api/conversations/{conversation_id}',json={"title":"Renamed chat"})
        assert renamed.json()['title']=="Renamed chat"
        assert client.delete(f'/api/conversations/{conversation_id}/messages').status_code==204
        assert client.get(f'/api/conversations/{conversation_id}').json()['messages']==[]
    finally:
        assert client.delete(f'/api/conversations/{conversation_id}').status_code==204
    assert client.get(f'/api/conversations/{conversation_id}').status_code==404

def test_true_streaming_persists_reply_and_usage(monkeypatch):
    conversation=client.post('/api/conversations',json={"title":"Stream test","interface":"chat"}).json()
    conversation_id=conversation['id']
    client.post(f'/api/conversations/{conversation_id}/messages',json={"speaker":"user","content":"Stream this"})
    async def fake_stream(self,provider_name,prompt,history):
        yield {"type":"delta","text":"Live "}
        yield {"type":"delta","text":"answer"}
        yield {"type":"usage","input_tokens":20,"output_tokens":2}
    monkeypatch.setattr(ThesisTeam,'stream_chat_reply',fake_stream)
    monkeypatch.setattr(store,'record_usage',lambda conversation_id,provider,model,input_tokens,output_tokens,estimated_cost_usd,estimated=False:{"conversation_id":conversation_id,"provider":provider,"model":model,"input_tokens":input_tokens,"output_tokens":output_tokens,"estimated_cost_usd":estimated_cost_usd,"estimated":int(estimated)})
    try:
        response=client.post('/api/chat/stream',json={"provider":"openai","prompt":"Stream this","conversation_id":conversation_id})
        assert response.status_code==200
        events=[__import__('json').loads(line) for line in response.text.splitlines()]
        assert [event['text'] for event in events if event['type']=='delta']==['Live ','answer']
        assert next(event for event in events if event['type']=='usage')['output_tokens']==2
        messages=client.get(f'/api/conversations/{conversation_id}').json()['messages']
        assert messages[-1]['speaker']=='openai' and messages[-1]['content']=='Live answer'
    finally:
        client.delete(f'/api/conversations/{conversation_id}')

def test_usage_estimate_and_summary():
    estimate=client.post('/api/usage/estimate',json={"providers":["openai","deepseek"],"prompt":"Estimate this request"})
    assert estimate.status_code==200 and estimate.json()['maximum_cost_usd']>0
    summary=client.get('/api/usage/summary?days=30')
    assert summary.status_code==200 and 'daily_budget_usd' in summary.json()

def test_model_catalog_validation_and_model_aware_cost():
    catalog=client.get('/api/models')
    assert catalog.status_code==200
    assert {'gpt-5.6-sol','gpt-5.6-terra','gpt-5.6-luna'}.issubset({item['id'] for item in catalog.json()['providers']['openai']})
    assert {'deepseek-v4-pro','deepseek-v4-flash'}.issubset({item['id'] for item in catalog.json()['providers']['deepseek']})
    sol=client.post('/api/usage/estimate',json={'providers':['openai'],'prompt':'x','models':{'openai':'gpt-5.6-sol'}}).json()
    luna=client.post('/api/usage/estimate',json={'providers':['openai'],'prompt':'x','models':{'openai':'gpt-5.6-luna'}}).json()
    assert luna['maximum_cost_usd'] < sol['maximum_cost_usd']
    invalid=client.post('/api/usage/estimate',json={'providers':['openai'],'prompt':'x','models':{'openai':'deepseek-v4-pro'}})
    assert invalid.status_code==422
    invalid_agent=client.post('/api/agents',json={'name':'Invalid','provider':'openai','model':'deepseek-v4-pro','role':'Test','instructions':'','max_sentences':5,'can_read_peers':True,'enabled':True})
    assert invalid_agent.status_code==422

def test_conversation_model_selection_is_persistent():
    conversation=client.post('/api/conversations',json={'title':'Models','interface':'chat'}).json()
    try:
        response=client.patch(f"/api/conversations/{conversation['id']}/models",json={'openai_model':'gpt-5.6-terra','deepseek_model':'deepseek-v4-flash'})
        assert response.status_code==200
        detail=client.get(f"/api/conversations/{conversation['id']}").json()
        assert detail['openai_model']=='gpt-5.6-terra' and detail['deepseek_model']=='deepseek-v4-flash'
    finally:
        client.delete(f"/api/conversations/{conversation['id']}")

def test_direct_workspace_model_override(monkeypatch):
    captured={}
    async def fake(self,prompt,workflow):
        captured['models']=[step['model'] for step in workflow['steps']]
        return ([],ProviderResult(status='ok',text='final'),1)
    monkeypatch.setattr(ThesisTeam,'run_workflow',fake)
    workflow_id=client.get('/api/workflows').json()[0]['id']
    response=client.post(f'/api/workflows/{workflow_id}/run',json={'prompt':'Use selected models','models':{'openai':'gpt-5.6-luna','deepseek':'deepseek-v4-flash'}})
    assert response.status_code==200
    assert set(captured['models'])=={'gpt-5.6-luna','deepseek-v4-flash'}
    store.delete_run(response.json()['run_id'])

def test_document_library_settings_runs_and_exports(monkeypatch):
    document=client.post('/api/documents',files={'file':('shared.txt',b'shared evidence','text/plain')})
    assert document.status_code==201
    document_id=document.json()['id']
    old=client.get('/api/settings').json()['values']
    try:
        assert any(item['id']==document_id for item in client.get('/api/documents').json())
        changed={**old,'cost_warning_usd':0.07}
        assert client.put('/api/settings',json=changed).json()['values']['cost_warning_usd']==0.07
        run=store.create_run('direct','Export this')
        store.finish_run(run['id'],'completed',{'turns':[{'agent_name':'Supervisor','text':'Done'}],'synthesis':{'text':'Final'}})
        assert client.get(f"/api/runs/{run['id']}/export?format=markdown").status_code==200
        pdf=client.get(f"/api/runs/{run['id']}/export?format=pdf")
        assert pdf.status_code==200 and pdf.content.startswith(b'%PDF')
        assert client.get(f"/api/runs/{run['id']}/export?format=json").status_code==200
    finally:
        store.delete_run(run['id'])
        client.put('/api/settings',json=old)
        client.delete(f'/api/documents/{document_id}')

def test_workflow_crud_for_visual_builder():
    project_id=client.get('/api/projects').json()[0]['id']
    agent_id=client.get('/api/agents').json()[0]['id']
    workflow=client.post('/api/workflows',json={'project_id':project_id,'name':'Builder test','description':'Initial'}).json()
    try:
        updated=client.put(f"/api/workflows/{workflow['id']}",json={'name':'Builder updated','description':'Edited'})
        assert updated.status_code==200 and updated.json()['name']=='Builder updated'
        steps=client.put(f"/api/workflows/{workflow['id']}/steps",json=[{'agent_id':agent_id,'mode':'respond'}])
        assert steps.status_code==200 and len(steps.json()['steps'])==1
    finally:
        assert client.delete(f"/api/workflows/{workflow['id']}").status_code==204

def test_direct_and_discussion_stream_events(monkeypatch):
    async def workflow_stream(self,prompt,workflow):
        step=workflow['steps'][0]
        yield {'type':'step_start','position':step['position'],'agent_name':step['agent_name']}
        yield {'type':'step_delta','position':step['position'],'text':'Live'}
        turn={**step,'status':'ok','text':'Live','error':'','usage_provider':'openai','model':'gpt-5.6-sol','input_tokens':1,'output_tokens':1}
        yield {'type':'step_done','turn':turn}
        synthesis={'status':'ok','text':'Final','error':'','usage_provider':'','model':'','input_tokens':0,'output_tokens':0}
        yield {'type':'synthesis','result':synthesis}
        yield {'type':'complete','turns':[turn],'synthesis':synthesis,'request_count':1}
    async def discussion_stream(self,prompt,rounds):
        yield {'type':'turn_start','round':1,'provider':'openai'}
        yield {'type':'turn_delta','round':1,'provider':'openai','text':'Live'}
        turn={'round':1,'provider':'openai','status':'ok','text':'Live','error':'','usage_provider':'openai','model':'gpt-5.6-sol','input_tokens':1,'output_tokens':1}
        yield {'type':'turn_done','turn':turn}
        synthesis={'status':'ok','text':'Final','error':'','usage_provider':'','model':'','input_tokens':0,'output_tokens':0}
        yield {'type':'synthesis','result':synthesis}
        yield {'type':'complete','turns':[turn],'synthesis':synthesis,'request_count':1}
    monkeypatch.setattr(ThesisTeam,'stream_workflow',workflow_stream)
    monkeypatch.setattr(ThesisTeam,'stream_discussion',discussion_stream)
    workflow_id=client.get('/api/workflows').json()[0]['id']
    direct=client.post(f'/api/workflows/{workflow_id}/stream',json={'prompt':'Stream it'})
    discussion=client.post('/api/discuss/stream',json={'prompt':'Discuss it','rounds':1})
    assert direct.status_code==200 and 'step_delta' in direct.text and 'complete' in direct.text
    assert discussion.status_code==200 and 'turn_delta' in discussion.text and 'complete' in discussion.text
    direct_meta=next(__import__('json').loads(line) for line in direct.text.splitlines() if '"type": "meta"' in line)
    discussion_meta=next(__import__('json').loads(line) for line in discussion.text.splitlines() if '"type": "meta"' in line)
    store.delete_run(direct_meta['run_id']);store.delete_run(discussion_meta['run_id'])

def test_chunked_retrieval_citations_and_graph():
    document=client.post('/api/documents',files={'file':('graph.txt',b'Neo4j connects Knowledge Graph concepts. Knowledge Graph supports GraphRAG retrieval.','text/plain')}).json()
    try:
        search=client.post('/api/retrieval/search',json={'query':'Knowledge Graph GraphRAG','document_ids':[document['id']]}).json()
        assert search['passages'] and search['passages'][0]['citation'].startswith(f"D{document['id']}C")
        graph=client.get('/api/graph').json()
        assert any(edge['document_id']==document['id'] for edge in graph['relationships'])
    finally:
        client.delete(f"/api/documents/{document['id']}")

def test_prompt_versions_evaluations_plugins_and_backup():
    agent=client.post('/api/agents',json={'name':'Versioned','provider':'openai','model':'gpt-5.6-luna','role':'First role','instructions':'v1','max_sentences':5,'can_read_peers':True,'enabled':True}).json()
    project_id=client.get('/api/projects').json()[0]['id']
    suite=client.post('/api/evaluations',json={'project_id':project_id,'name':'Platform test','description':'test'}).json()
    try:
        payload={key:agent[key] for key in ('name','provider','model','role','instructions','max_sentences','can_read_peers','enabled')};payload['instructions']='v2'
        client.put(f"/api/agents/{agent['id']}",json=payload)
        assert len(client.get(f"/api/agents/{agent['id']}/versions").json())>=2
        case=client.post(f"/api/evaluations/{suite['id']}/cases",json={'question':'What is RAG?','expected':'retrieval generation','tags':'rag'})
        assert case.status_code==201 and client.get(f"/api/evaluations/{suite['id']}").json()['cases']
        assert client.post('/api/provider-plugins',json={'name':'Unsafe','adapter':'openai-compatible','base_url':'http://localhost','api_key_env':'X','enabled':False,'config':{}}).status_code==422
        plugin=client.post('/api/provider-plugins',json={'name':'Safe test','adapter':'openai-compatible','base_url':'https://example.invalid/v1','api_key_env':'SAFE_TEST_KEY','enabled':False,'config':{'models':['test-model']}})
        assert plugin.status_code==201
        backup=client.get('/api/backup')
        assert backup.status_code==200 and backup.content.startswith(b'PK')
    finally:
        with store.connect() as db:
            db.execute('DELETE FROM evaluation_suites WHERE id=?',(suite['id'],))
            db.execute("DELETE FROM provider_plugins WHERE name IN ('Unsafe','Safe test')")
        client.delete(f"/api/agents/{agent['id']}")

def test_durable_job_and_n8n_contract():
    workflow_id=client.get('/api/workflows').json()[0]['id']
    job=client.post(f'/api/webhooks/n8n/workflows/{workflow_id}',json={'prompt':'Queued test','project_id':1}).json()
    try:
        assert job['status']=='pending' and job['kind']=='workflow'
        assert any(item['id']==job['id'] for item in client.get('/api/jobs').json())
        cancelled=client.post(f"/api/jobs/{job['id']}/cancel")
        assert cancelled.status_code==200
    finally:
        with store.connect() as db:db.execute('DELETE FROM jobs WHERE id=?',(job['id'],))

def test_neo4j_sync_requires_configuration(monkeypatch):
    from app import main
    monkeypatch.setattr(main,'runtime_settings',lambda: Settings())
    response=client.post('/api/graph/neo4j-sync')
    assert response.status_code==409

def test_new_user_gets_seeded_isolated_workspace():
    from app.main import platform
    from app.storage import CURRENT_PROJECT_ID
    username=f'isolated-test-{uuid.uuid4().hex[:8]}'
    user=platform.create_user(username,'Isolated test workspace')
    token=CURRENT_PROJECT_ID.set(user['project_id'])
    try:
        agents=store.list_rows('agents')
        workflows=store.list_rows('workflows')
        assert len(agents)==4 and len(workflows)==1
        assert all(item['project_id']==user['project_id'] for item in agents+workflows)
        assert platform.authenticate(user['access_token'])['project_id']==user['project_id']
    finally:
        CURRENT_PROJECT_ID.reset(token)
        with store.connect() as db:
            db.execute('DELETE FROM users WHERE id=?',(user['id'],))
            db.execute('DELETE FROM projects WHERE id=?',(user['project_id'],))

def test_builder_workspace_isolated_branch_paths_and_merge(tmp_path):
    from app.builder import BuilderError, BuilderWorkspace
    from app.storage import Store
    repository=tmp_path/'repository';repository.mkdir()
    subprocess.run(['git','init','-b','main'],cwd=repository,check=True,capture_output=True)
    subprocess.run(['git','config','user.email','builder-test@example.invalid'],cwd=repository,check=True)
    subprocess.run(['git','config','user.name','Builder Test'],cwd=repository,check=True)
    (repository/'README.md').write_text('initial\n')
    subprocess.run(['git','add','.'],cwd=repository,check=True)
    subprocess.run(['git','commit','-m','initial'],cwd=repository,check=True,capture_output=True)
    isolated_store=Store(tmp_path/'builder.db');workspace=BuilderWorkspace(isolated_store,repository)
    session=workspace.create(1,'Update the readme','none')
    worktree=__import__('pathlib').Path(session['worktree'])
    assert worktree.exists() and session['branch'].startswith('builder/session-')
    with pytest.raises(BuilderError):workspace.apply_changes(worktree,[{'path':'../escape.txt','action':'write','content':'bad'}])
    workspace.apply_changes(worktree,[{'path':'README.md','action':'write','content':'updated\n'},{'path':'new.txt','action':'write','content':'new file\n'}])
    assert 'b/new.txt' in workspace.detail(session['id'],1)['diff']
    workspace.update(session['id'],status='ready')
    merged=workspace.merge(session['id'],1)
    assert merged['status']=='merged' and (repository/'README.md').read_text()=='updated\n'
