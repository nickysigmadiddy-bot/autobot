#!/usr/bin/env python3
"""Wire learner.hpp into AutoBot v0.9.0 main.cpp.

Usage (from the project root, learner.hpp already copied to src/):
    python3 apply_learner_patch.py src/main.cpp

Every anchor must match exactly once or the script aborts without touching the file.
A backup is written to main.cpp.bak first.
"""
import sys, shutil

path = sys.argv[1] if len(sys.argv) > 1 else "src/main.cpp"
src = open(path, encoding="utf-8").read()
if "learner.hpp" in src:
    sys.exit("already patched")

def tab(block):
    """Blocks below are written with 4-space indents; main.cpp uses tabs."""
    out = []
    for line in block.split("\n"):
        n = len(line) - len(line.lstrip(" "))
        out.append("\t" * (n // 4) + line[n:])
    return "\n".join(out)

def replace_once(s, old, new, label):
    old, new = tab(old), tab(new)
    if s.count(old) != 1:
        sys.exit(f"ANCHOR NOT FOUND EXACTLY ONCE: {label} (found {s.count(old)})")
    return s.replace(old, new)

def replace_between(s, start, end, new, label):
    start, end, new = tab(start), tab(end), tab(new)
    if s.count(start) != 1 or s.count(end) != 1:
        sys.exit(f"MARKERS NOT FOUND EXACTLY ONCE: {label}")
    a, b = s.index(start), s.index(end)
    if a >= b:
        sys.exit(f"MARKERS OUT OF ORDER: {label}")
    return s[:a] + new + s[b:]

# 1) include ------------------------------------------------------------------------------
src = replace_once(src, "#include <vector>\n", '#include <vector>\n#include "learner.hpp"\n', "include")

# 2) Bot members --------------------------------------------------------------------------
src = replace_once(src,
"    // experiments\n    std::vector<Exp> exps;\n",
"    // experiments (legacy, no longer driven: learn::Learner is the single source of truth)\n"
"    std::vector<Exp> exps;\n"
"    learn::Learner L;\n"
"    bool learnReady = false;\n"
"    std::string learnKey;\n", "bot members")

# 3) every real ground click is reported to the learner -------------------------------------
src = replace_once(src,
"        G.pressAir = !p->m_isOnGround;\n",
"        G.pressAir = !p->m_isOnGround;\n"
"        if (C.learn && G.learnReady) G.L.noteClick(G.pressX, G.pressAir);\n", "setHold")

# 4) botStep: replace the old experiment block with the learner ----------------------------
src = replace_between(src,
"    // ---- trial-and-error experiments (timing learned from real deaths) ---------",
"    (void)fromPlan;",
"""    // ---- adaptive timing learner: the ONE source of truth for learned click timing -------
    if (C.learn && G.learnReady) {
        float F = std::max(1.f, G.speed / 60.f);                  // world units per game frame
        auto la = G.L.tick(pos.x, p->m_isOnGround, act && !G.holding, G.holding, F, F * h);
        if (la.suppress) act = false;                              // hold the planner's own click back
        if (la.force) { act = true; G.lastPredClickX = pos.x; ++G.expFires; }
        if (la.suppress || la.force) { G.havePred = false; G.reason = R_EXPERIMENT; }
        G.L.noteProgress(pos.x, p->m_isOnGround);
    }
""", "botStep block")

# 5) resetLevel ---------------------------------------------------------------------------
src = replace_once(src,
"        for (auto& ex : G.exps) ex.fired = false;\n",
"        if (G.learnReady) G.L.beginAttempt();\n", "resetLevel")

# 6) saveLearn helper + level key ----------------------------------------------------------
src = replace_once(src,
"static char const* reasonText(char* buf, size_t n) {",
"""static void saveLearn() {
    if (!G.learnReady || G.learnKey.empty() || !G.L.dirty) return;
    Mod::get()->setSavedValue<std::string>(G.learnKey, G.L.save());
    G.L.dirty = false;
}

static char const* reasonText(char* buf, size_t n) {""", "saveLearn")

# 7) setupHasCompleted: build + load ---------------------------------------------------------
src = replace_once(src,
"        G.debug = CCDrawNode::create();\n",
"""        {
            // Learned timings belong to one level, one layout, one game version and one physics config.
            std::vector<std::pair<float, float>> hz;
            float len = 1.f;
            for (auto const& o : G.objs) {
                if (o.k == K_HAZARD) hz.push_back({o.r.x0, o.r.x1});
                len = std::max(len, o.r.x1);
            }
            std::string lname = std::string(m_level->m_levelName);
            std::string lid = std::to_string((long long)m_level->m_levelID.value());
            uint64_t seed = learn::fnv(lid + "|" + lname + "|" + GEODE_GD_VERSION_STRING);
            char pc[160];
            std::snprintf(pc, sizeof(pc), "%.4f|%.4f|%.4f|%.4f|%.1f|%d", (double)BASE_SPEED, (double)NOM_G,
                (double)NOM_JUMP, (double)NOM_POS, (double)phys::HALF, (int)G.objs.size());
            seed = learn::fnv(pc, seed);
            G.L = learn::Learner();
            G.L.build(hz, len, seed);
            G.learnKey = "learn-" + lid + "-" + std::to_string(learn::fnv(lname) & 0xffffffull);
            G.learnReady = true;
            std::string saved = Mod::get()->getSavedValue<std::string>(G.learnKey, "");
            bool loaded = !saved.empty() && G.L.load(saved);
            log::info("[AutoBot] learner: {} obstacle sites, saved timings {}", G.L.cl.size(),
                saved.empty() ? "none yet" : (loaded ? "loaded" : "REJECTED (other level layout / game version / physics)"));
        }

        G.debug = CCDrawNode::create();
""", "setupHasCompleted")

# 8) onQuit + levelComplete ------------------------------------------------------------------
src = replace_once(src,
"    void onQuit() {\n        releaseBtn(this, true);\n",
"""    void levelComplete() {
        if (G.learnReady) { G.L.noteProgress(1e9f, true); saveLearn(); }
        PlayLayer::levelComplete();
    }

    void onQuit() {
        saveLearn();
        G.learnReady = false;
        releaseBtn(this, true);
""", "onQuit")

# 9) onDeath: replace the old Exp bookkeeping --------------------------------------------------
src = replace_between(src,
"        if (!C.learn) return;\n",
"    void postUpdate(float dt) {",
"""        if (!C.learn || !G.learnReady) return;

        // Map the death classifier onto what it says about TIMING.
        int lk;
        switch (cls) {
            case F_TOO_EARLY:                                      lk = learn::K_EARLY; break;
            case F_TOO_LATE: case F_LOW_HEIGHT:                    lk = learn::K_LATE;  break;
            case F_JUMP_COLLISION: case F_UNSAFE_LANDING: case F_DEAD_END: lk = learn::K_AMBIG; break;
            case F_NOT_REGISTERED:                                 lk = learn::K_REG;   break;
            default:                                               lk = learn::K_PHYS;  break;
        }
        float F = std::max(1.f, G.speed / 60.f);
        auto rep = G.L.onDeath(pos.x, lastGr, vLast, lk, F, F * G.lastH);
        log::info("[AutoBot] learner: {}", rep.text);
        saveLearn();
    }

""", "onDeath")

# 10) HUD ---------------------------------------------------------------------------------
src = replace_once(src,
"        std::string s = buf;\n        (void)n;\n",
"""        std::string s = buf;
        (void)n;
        if (C.learn && G.learnReady) { s += "\\n"; s += G.L.hud(m_player1->getPositionX()); }
""", "hud")

shutil.copy(path, path + ".bak")
open(path, "w", encoding="utf-8").write(src)
print("patched", path, "(backup:", path + ".bak)")
