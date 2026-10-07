/**
 * Synapse Playground worker.
 *
 * Sends the editor's program to the local playground server (POST /api/run),
 * which runs it with the real Synapse interpreter and returns stdout/stderr.
 * Start the server with: python playground/server.py
 */

const RUN_ENDPOINT = "/api/run";

function post(type, payload = {}) {
    self.postMessage({ type, ...payload });
}

function emitLines(type, text) {
    if (!text) return;
    for (const line of text.replace(/\n$/, "").split("\n")) {
        post(type, { text: line });
    }
}

async function runProgram(code) {
    let response;
    try {
        response = await fetch(RUN_ENDPOINT, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ code }),
        });
    } catch (err) {
        throw new Error(
            "Synapse çalışma zamanına ulaşılamadı. `python playground/server.py` ile sunucuyu başlatıp " +
            "yazdırdığı adresi açın (index.html dosyasını doğrudan diskten açmak kod çalıştıramaz)."
        );
    }

    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(result.error || `Sunucu HTTP ${response.status} döndürdü`);
    }
    return result;
}

self.onmessage = async function (e) {
    const data = e.data || {};

    if (data.type === "init" || data.type === "reset") {
        post("ready", { info: "Synapse çalışma zamanı hazır: programlar yerel sunucu üzerinden gerçek yorumlayıcıyla çalışır." });
        return;
    }

    if (data.type === "run") {
        try {
            const result = await runProgram(data.code || "");
            emitLines("stdout", result.stdout);
            emitLines("stderr", result.stderr);
            if (!result.stdout && !result.stderr) {
                post("info", { text: "(Program çıktı üretmeden tamamlandı.)" });
            }
            post("done", { duration: result.duration_ms });
        } catch (err) {
            post("error", { error: err.message });
        }
    }
};
