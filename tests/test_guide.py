"""Phase 7: hybrid tutor — deterministic knowledge base, action whitelist and
the Ollama adapter (exercised with mocked HTTP; no Ollama server is needed)."""
import json

import httpx
import pytest

from backend.core import config, guide

SAMPLES = [
    {"id": "phantom_normal", "name": "Phantom · Normal", "kind": "phantom", "view": "A4C", "subject_id": "phantom_normal", "ef_class": "normal (>=55 %)"},
    {"id": "phantom_hfref", "name": "Phantom · HFrEF", "kind": "phantom", "view": "A4C", "subject_id": "phantom_hfref", "ef_class": "reduced (<=40 %)"},
    {"id": "camus_patient0017_4ch", "name": "CAMUS · patient0017 · A4C", "kind": "camus", "view": "A4C", "subject_id": "camus_patient0017", "ef_class": "normal (>=55 %)"},
    {"id": "camus_patient0017_2ch", "name": "CAMUS · patient0017 · A2C", "kind": "camus", "view": "A2C", "subject_id": "camus_patient0017", "ef_class": "normal (>=55 %)"},
    {"id": "camus_patient0038_4ch", "name": "CAMUS · patient0038 · A4C", "kind": "camus", "view": "A4C", "subject_id": "camus_patient0038", "ef_class": "reduced (<=40 %)"},
]
CTX = {"view": "A4C", "tilt_deg": 0, "quiz_index": 0, "sample_id": "phantom_normal", "edv_ml": 200.0, "esv_ml": 136.0,
       "segmented": True, "calibrated": True}


def det(message, **kw):
    return guide.deterministic_respond(message, context=kw.pop("context", CTX), samples=kw.pop("samples", SAMPLES), **kw)


@pytest.fixture(autouse=True)
def no_real_ollama(monkeypatch):
    """Default for every test: Ollama enabled in config but nothing listening."""
    monkeypatch.setattr(config, "OLLAMA_ENABLED", True)
    guide.set_ollama(guide.OllamaClient(transport=httpx.MockTransport(lambda req: httpx.Response(503))))
    yield
    guide.set_ollama(None)


def mock_ollama(chat_handler, tags_status=200):
    calls = {"chat": 0, "bodies": []}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(tags_status, json={"models": []})
        assert request.url.path == "/api/chat"
        calls["chat"] += 1
        calls["bodies"].append(json.loads(request.content))
        return chat_handler(request)

    guide.set_ollama(guide.OllamaClient(transport=httpx.MockTransport(handler)))
    return calls


def llm_json(reply, actions):
    return httpx.Response(200, json={"message": {"role": "assistant", "content": json.dumps({"reply": reply, "actions": actions})}})


# ----------------------------- deterministic KB ----------------------------- #
@pytest.mark.parametrize("message,intent", [
    ("Show me the apical 2-chamber view", "set_view"),
    ("explain foreshortning", "explain_foreshortening"),           # typo is repaired
    ("demonstrate foreshortening at 25 degrees", "demo_foreshortening"),
    ("how do I avoid foreshortening?", "fix_foreshortening"),
    ("set the tilt to 12", "set_tilt"),
    ("What is the difference between single-plane and biplane Simpson?", "explain_biplane"),
    ("explain Simpson's method of disks", "explain_simpson"),
    ("Interpret the current EDV, ESV and LVEF", "interpret"),
    ("is my ejection fraction normal?", "interpret"),
    ("Tell me about the interventricular septum", "anatomy"),
    ("what is the left atrial appendage", "anatomy"),
    ("where is the IVC?", "anatomy"),
    ("The segmentation mask looks wrong, what can I do?", "prompting_help"),
    ("which way does the index marker point?", "probe_orientation"),
    ("hide the mask", "toggle_overlay"),
    ("load the HFrEF phantom", "select_sample"),
    ("is this a medical device?", "about"),
    ("quiz me", "quiz"),
    ("", "help"),
    ("what's the weather like", "fallback"),
])
def test_intent_matching(message, intent):
    assert det(message)["intent"] == intent


def test_deterministic_engine_is_pure():
    for msg in ("explain foreshortening", "interpret this LVEF", "quiz me on PLAX", "tell me about the aorta"):
        assert det(msg) == det(msg)


def test_anatomy_covers_every_structure_in_the_3d_model():
    assert set(guide.ANATOMY) == set(guide.STRUCTURE_NAMES)
    r = det("tell me about the pulmonary artery")
    assert {"type": "highlight", "chambers": ["PA"]} in r["actions"]
    assert {"type": "toggle_overlay", "overlay": "cutaway", "on": False} in r["actions"]


def test_interpretation_is_state_aware_and_calculated():
    r = det("interpret the current EDV, ESV and LVEF")
    assert "32.0 %" in r["verified"] and "reduced" in r["verified"] and "200.0" in r["verified"]
    assert "algorithmic estimates" in r["reply"] and "not a medical device" in r["reply"]
    uncal = det("interpret this LVEF", context={**CTX, "calibrated": False})
    assert "index" in uncal["verified"] and "unaffected" in uncal["reply"]
    empty = det("interpret this LVEF", context={"view": "A4C"})
    assert empty["verified"] is None and "Segment a clip first" in empty["reply"]


@pytest.mark.parametrize("lvef,band", [(25, "reduced"), (40, "reduced"), (45, "mildly reduced"), (52, "low-normal"), (60, "normal")])
def test_lvef_bands(lvef, band):
    assert band in guide.lvef_band(lvef)


def test_foreshortening_numbers_come_from_the_math_engine():
    from backend.core.simpsons_math import simulate_apical_foreshortening

    r = det("demonstrate foreshortening with a 20 degree tilt")
    expect = simulate_apical_foreshortening(200.0, 136.0, 20.0)
    assert f"{expect['foreshortened']['lvef_pct']:.1f}" in r["verified"]
    assert {"type": "set_tilt", "tilt_deg": 20.0} in r["actions"]
    assert "a′ = 1 / √" in det("explain foreshortening")["reply"]


@pytest.mark.parametrize("message,overlay,on", [
    ("hide the mask", "mask", False), ("show the sector fan", "fan", True), ("turn off the labels", "labels", False),
    ("disable the cut-away", "cutaway", False), ("show the 20 disks overlay", "chords", True), ("turn off the beating", "beat", False),
])
def test_overlay_commands(message, overlay, on):
    assert det(message)["actions"] == [{"type": "toggle_overlay", "overlay": overlay, "on": on}]


@pytest.mark.parametrize("message,sample_id", [
    ("load the HFrEF phantom", "phantom_hfref"), ("open the normal phantom", "phantom_normal"),
    ("load patient 38", "camus_patient0038_4ch"), ("switch to patient0017 a2c clip", "camus_patient0017_2ch"),
    ("next patient", "phantom_hfref"), ("load a reduced EF case", "camus_patient0038_4ch"),
])
def test_sample_selection(message, sample_id):
    r = det(message)
    assert r["intent"] == "select_sample" and r["actions"][0] == {"type": "select_sample", "sample_id": sample_id}


def test_unknown_sample_is_not_invented():
    r = det("load patient 9999")
    assert r["intent"] == "select_sample_unknown" and r["actions"] == []


def test_quiz_runs_through_a_view_and_keeps_score():
    q = det("quiz me on PLAX")["quiz"]
    assert q["id"] == "PLAX:0" and q["total"] == len(guide.QUIZ["PLAX"]) >= 4 and "answer" not in q
    asked = correct = 0
    while q:
        right = guide.QUIZ["PLAX"][int(q["id"].split(":")[1])]["answer"]
        r = guide.deterministic_respond(quiz_answer={"id": q["id"], "answer": right},
                                        context={**CTX, "quiz_asked": asked, "quiz_correct": correct}, samples=SAMPLES)
        asked += 1
        correct += 1
        assert r["quiz_result"]["correct"] and f"Score so far: {correct} / {asked}" in r["reply"]
        q = r["quiz"]
    assert asked == len(guide.QUIZ["PLAX"])
    wrong = guide.deterministic_respond(quiz_answer={"id": "A4C:0", "answer": 0}, context=CTX)
    assert wrong["quiz_result"]["correct"] is False and "Score so far: 0 / 1" in wrong["reply"]


# ------------------------------ action whitelist --------------------------- #
def test_action_whitelist_accepts_only_known_shapes():
    raw = [
        {"type": "set_view", "view": "PLAX"},
        {"type": "set_view", "view": "A5C"},                     # unknown view
        {"type": "set_tilt", "tilt_deg": 45},                    # out of range
        {"type": "set_tilt", "tilt_deg": 10},
        {"type": "toggle_overlay", "overlay": "mask", "on": False},
        {"type": "toggle_overlay", "overlay": "everything", "on": True},
        {"type": "select_sample", "sample_id": "phantom_hfref"},
        {"type": "select_sample", "sample_id": "../../etc/passwd"},
        {"type": "delete_all_files"},
        "set_view",
    ]
    valid, rejected = guide.validate_actions(raw[:8] + raw[8:], {s["id"] for s in SAMPLES})
    assert valid == [{"type": "set_view", "view": "PLAX"}, {"type": "set_tilt", "tilt_deg": 10.0},
                     {"type": "toggle_overlay", "overlay": "mask", "on": False},
                     {"type": "select_sample", "sample_id": "phantom_hfref"}]
    assert len(rejected) == 4                                    # only the first 8 items are even considered
    assert guide.validate_actions("not a list")[0] == []
    only_llm, rej = guide.validate_actions([{"type": "highlight", "chambers": ["LV"]}], allowed=guide.LLM_ALLOWED_ACTIONS)
    assert only_llm == [] and len(rej) == 1


def test_every_deterministic_action_passes_the_whitelist():
    ids = {s["id"] for s in SAMPLES}
    for msg in ("show me PLAX", "explain foreshortening", "demo a 20 degree tilt", "hide the mask", "load patient 38",
                "tell me about the aorta", "tell me about the SVC", "quiz me", "set the tilt to 5"):
        r = det(msg)
        valid, rejected = guide.validate_actions(r["actions"], ids)
        assert rejected == [] and valid == r["actions"], msg


# ------------------------------ Ollama adapter ------------------------------ #
def test_engine_is_deterministic_when_ollama_is_absent():
    st = guide.engine_status()
    assert st["id"] == "deterministic" and st["label"] == "Offline Deterministic Tutor" and st["reachable"] is False
    r = guide.respond("explain foreshortening", context=CTX, samples=SAMPLES)
    assert r["engine"] == "deterministic" and r["intent"] == "explain_foreshortening"


def test_ollama_can_be_disabled_by_configuration(monkeypatch):
    calls = mock_ollama(lambda req: llm_json("hi", []))
    monkeypatch.setattr(config, "OLLAMA_ENABLED", False)
    assert guide.engine_status()["id"] == "deterministic"
    assert guide.respond("tell me something about echo physics", context=CTX, samples=SAMPLES)["engine"] == "deterministic"
    assert calls["chat"] == 0


def test_valid_llm_json_is_used_and_actions_are_filtered():
    calls = mock_ollama(lambda req: llm_json(
        "Doppler measures blood velocity from the frequency shift of the returning echo.",
        [{"type": "set_view", "view": "PLAX"}, {"type": "toggle_overlay", "overlay": "fan", "on": False},
         {"type": "select_sample", "sample_id": "phantom_hfref"}, {"type": "select_sample", "sample_id": "made_up"},
         {"type": "run_shell", "cmd": "rm -rf /"}, {"type": "highlight", "chambers": ["LV"]}]))
    assert guide.engine_status() == {"id": "ollama", "label": "Ollama Local LLM", "model": config.OLLAMA_MODEL,
                                     "reachable": True, "ollama_enabled": True}
    r = guide.respond("tell me how doppler works", context=CTX, samples=SAMPLES)
    assert r["engine"] == "ollama" and r["intent"] == "llm" and r["reply"].startswith("Doppler measures")
    assert r["actions"] == [{"type": "set_view", "view": "PLAX"}, {"type": "toggle_overlay", "overlay": "fan", "on": False},
                            {"type": "select_sample", "sample_id": "phantom_hfref"}]
    assert len(r["rejected_actions"]) == 3
    body = calls["bodies"][0]
    assert body["model"] == config.OLLAMA_MODEL and body["stream"] is False and body["format"] == "json"
    assert body["messages"][0]["role"] == "system" and "not a medical device" in body["messages"][0]["content"]
    assert body["messages"][1] == {"role": "user", "content": "tell me how doppler works"}


def test_llm_start_quiz_action_yields_a_deterministic_question():
    mock_ollama(lambda req: llm_json("Let's test your PSAX knowledge.", [{"type": "start_quiz", "view": "PSAX"}]))
    r = guide.respond("could you examine my knowledge of something", context=CTX, samples=SAMPLES)
    assert r["engine"] == "ollama" and r["quiz"]["view"] == "PSAX" and r["quiz"]["question"] == guide.QUIZ["PSAX"][0]["question"]
    assert {"type": "set_view", "view": "PSAX"} in r["actions"]


def test_recognised_intent_keeps_deterministic_actions_and_numbers():
    mock_ollama(lambda req: llm_json("Foreshortening means the plane misses the apex.", [{"type": "set_view", "view": "PSAX"}]))
    r = guide.respond("demonstrate foreshortening with a 20 degree tilt", context=CTX, samples=SAMPLES)
    assert r["engine"] == "ollama" and r["intent"] == "demo_foreshortening"
    assert r["reply"] == "Foreshortening means the plane misses the apex."
    assert {"type": "set_tilt", "tilt_deg": 20.0} in r["actions"] and {"type": "set_view", "view": "PSAX"} not in r["actions"]
    assert "200" in r["verified"], "the verified calculation still comes from the math engine"


@pytest.mark.parametrize("content", [
    "Sure! Here is my answer: the heart has four chambers.",          # not JSON
    '{"reply": 42}',                                                   # wrong type
    '{"actions": []}',                                                 # reply missing
    '{"reply": "", "actions": []}',                                    # empty reply
    '{"reply": "ok", "actions": "set_view"}',                          # actions not a list
    '{"reply": "x", "actions": [] ',                                   # truncated JSON
])
def test_malformed_llm_output_falls_back_to_the_deterministic_tutor(content):
    mock_ollama(lambda req: httpx.Response(200, json={"message": {"content": content}}))
    r = guide.respond("tell me how doppler works", context=CTX, samples=SAMPLES)
    assert r["engine"] == "deterministic" and r["intent"] == "fallback" and r["llm_fallback"] is True
    assert r["actions"] == []


def test_fenced_json_is_accepted():
    parsed = guide.parse_llm_json('```json\n{"reply": "hello", "actions": []}\n```')
    assert parsed is not None and parsed.reply == "hello"


def test_timeout_falls_back_and_stops_retrying():
    def boom(request):
        raise httpx.ReadTimeout("timed out", request=request)

    calls = mock_ollama(boom)
    r = guide.respond("tell me how doppler works", context=CTX, samples=SAMPLES)
    assert r["engine"] == "deterministic" and r["llm_fallback"] is True and r["reply"]
    assert r["engine_status"]["id"] == "deterministic", "a failed server is treated as unreachable"
    guide.respond("tell me how doppler works", context=CTX, samples=SAMPLES)
    assert calls["chat"] == 1, "the next turn must not wait on the dead server again"


def test_http_error_falls_back():
    mock_ollama(lambda req: httpx.Response(500, text="model not found"))
    assert guide.respond("tell me how doppler works", context=CTX, samples=SAMPLES)["engine"] == "deterministic"


def test_quizzes_presets_and_ui_commands_never_reach_the_llm():
    calls = mock_ollama(lambda req: llm_json("should not be used", []))
    assert guide.respond("quiz me", context=CTX, samples=SAMPLES)["engine"] == "deterministic"
    assert guide.respond(quiz_answer={"id": "A4C:0", "answer": 1}, context=CTX, samples=SAMPLES)["quiz_result"]["correct"]
    assert guide.respond(preset="explain_simpson", context=CTX, samples=SAMPLES)["engine"] == "deterministic"
    assert guide.respond("hide the mask", context=CTX, samples=SAMPLES)["engine"] == "deterministic"
    assert guide.respond("load patient 38", context=CTX, samples=SAMPLES)["engine"] == "deterministic"
    assert calls["chat"] == 0


def test_adapter_defaults_follow_the_specification():
    client = guide.OllamaClient()
    assert client.base_url == "http://localhost:11434" and client.timeout == 4.0
    assert config.OLLAMA_MODEL.startswith("qwen2.5")
