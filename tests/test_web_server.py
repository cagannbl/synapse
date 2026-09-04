import time
import urllib.request
import urllib.parse
import json
import pytest
from synapse.core.web import serve, SynapseServer, SynapseRequest, SynapseResponse
from synapse.core.memory import memory
from synapse.ai.agent_runtime import AgentRuntime


def test_synapse_request_response_classes():
    req = SynapseRequest("/test", "GET", {"q": "hello"}, {}, {"data": 123})
    assert req.path == "/test"
    assert req.method == "GET"
    assert req.query["q"] == "hello"

    resp = SynapseResponse({"status": "ok"}, status_code=201)
    assert resp.status_code == 201
    assert resp.body["status"] == "ok"


def test_synapse_web_server_live_endpoints():
    mem = memory()
    mem.remember("Synapse native AI engine.", metadata={"category": "tech"})

    agent = AgentRuntime(name="TestBot", instructions="Reply shortly.")

    def status_handler(req):
        return {"status": "ok", "mem_count": mem.count()}

    def chat_handler(req):
        p = req.body.get("prompt", "")
        reply = agent.run(p)
        return {"reply": reply}

    def search_handler(req):
        q = req.query.get("q", "")
        return {"results": mem.recall(q, top_k=2)}

    # Port 8099 üzerinde non-blocking test sunucusu
    server = serve(
        port=8099,
        routes={
            "/api/status": status_handler,
            "/api/chat": chat_handler,
            "/api/search": search_handler,
        },
        static_dir="public",
        blocking=False
    )

    try:
        # Sunucunun başlaması için kısa bekleme
        time.sleep(0.3)

        # 1. Statik Dosya Testi (GET /)
        with urllib.request.urlopen("http://127.0.0.1:8099/") as res:
            assert res.status == 200
            html = res.read().decode("utf-8")
            assert "Synapse" in html
            assert "Otonom Ajan" in html

        # 2. JSON API Endpoint Testi (GET /api/status)
        with urllib.request.urlopen("http://127.0.0.1:8099/api/status") as res:
            assert res.status == 200
            data = json.loads(res.read().decode("utf-8"))
            assert data["status"] == "ok"
            assert data["mem_count"] == 1

        # 3. POST JSON API Testi (/api/chat)
        req_data = json.dumps({"prompt": "Hello Synapse"}).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:8099/api/chat",
            data=req_data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as res:
            assert res.status == 200
            chat_data = json.loads(res.read().decode("utf-8"))
            assert "reply" in chat_data
            assert len(chat_data["reply"]) > 0

        # 4. Search API Testi (GET /api/search?q=engine)
        with urllib.request.urlopen("http://127.0.0.1:8099/api/search?q=engine") as res:
            assert res.status == 200
            search_data = json.loads(res.read().decode("utf-8"))
            assert len(search_data["results"]) >= 1
            assert "Synapse" in search_data["results"][0]["text"]

    finally:
        server.stop()
