#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_set>
#include <vector>

using namespace geode::prelude;

// ===========================================================================
// AutoBot v0.6.0 - cube only, step-aligned, exact search, committed plans.
// v0.6.0: measured cube physics (jump 11.18, gravity 0.864/frame^2, position
// scale 0.9), safe floor learning, anti-cheat/phantom death filter, log rate
// limiting, cached settings, time-throttled HUD and overlay.
// ===========================================================================

constexpr float BASE_SPEED = 311.58f;
constexpr double SEARCH_MS_LIMIT = 2.0;
constexpr float DIV_TOL = 0.8f;
constexpr int DIV_COOLDOWN = 20;
namespace phys {
	constexpr float HALF = 15.f;
	constexpr float GRAVITY = 0.864f;     // velocity loss per frame (1/60 s), measured
	constexpr float JUMP = 11.18f;        // measured
	constexpr float POS_SCALE = 0.9f;     // y += vy * h * POS_SCALE, measured
	constexpr float FALL_MAX = 15.f;      // unverified
	constexpr float SPEEDS[5] = {251.16f, 311.58f, 387.42f, 468.0f, 576.0f};
	constexpr float PSPEEDS[5] = {0.7f, 0.9f, 1.1f, 1.3f, 1.6f};
}

static bool isCube(PlayerObject* p) {
	return !(p->m_isShip || p->m_isBall || p->m_isBird || p->m_isDart ||
	         p->m_isRobot || p->m_isSpider || p->m_isSwing);
}
static char const* modeName(PlayerObject* p) {
	if (p->m_isShip) return "ship";
	if (p->m_isBall) return "ball";
	if (p->m_isBird) return "ufo";
	if (p->m_isDart) return "wave";
	if (p->m_isRobot) return "robot";
	if (p->m_isSpider) return "spider";
	if (p->m_isSwing) return "swing";
	return "cube";
}

// ---------------------------------------------------------------------------
// Settings cache (refreshed every 60 steps instead of every step)
// ---------------------------------------------------------------------------
struct Cfg {
	bool enabled = true, practiceOnly = true, useOrbs = true, debugDraw = true, learn = true;
	int horizon = 60;
	long budget = 150000;
	float margin = 2.f;
	int age = 1000000;
};
static Cfg C;
static void refreshCfg() {
	auto m = Mod::get();
	C.enabled = m->getSettingValue<bool>("enabled");
	C.practiceOnly = m->getSettingValue<bool>("practice-only");
	C.useOrbs = m->getSettingValue<bool>("use-orbs");
	C.debugDraw = m->getSettingValue<bool>("debug-draw");
	C.learn = m->getSettingValue<bool>("learn-from-deaths");
	C.horizon = std::clamp<int>(static_cast<int>(m->getSettingValue<int64_t>("horizon")), 10, 62);
	C.budget = (long)m->getSettingValue<int64_t>("beam-width") * 1500L;
	C.margin = static_cast<float>(m->getSettingValue<double>("hazard-margin"));
	C.age = 0;
}

// ---------------------------------------------------------------------------
// Geometry / objects
// ---------------------------------------------------------------------------
struct Rc { float x0, y0, x1, y1; };
static Rc toRc(CCRect const& r) { return {r.getMinX(), r.getMinY(), r.getMaxX(), r.getMaxY()}; }
static inline Rc mir(Rc const& r, float sgn) {
	return sgn > 0.f ? r : Rc{r.x0, -r.y1, r.x1, -r.y0};
}
static inline bool hit(Rc const& a, Rc const& b) {
	return a.x0 < b.x1 && a.x1 > b.x0 && a.y0 < b.y1 && a.y1 > b.y0;
}

enum Kind : uint8_t { K_SOLID, K_HAZARD, K_ORB, K_PAD, K_PORTAL };

struct Obj {
	Rc r;
	Kind k;
	int id = 0;
	int ix = -1;
	bool used = false;
};

static bool isOrbId(int id) { return id == 36 || id == 141 || id == 1333; }
static bool isPadId(int id) { return id == 35 || id == 140 || id == 1332; }
static bool isPortalId(int id) {
	switch (id) {
		case 10: case 11: case 12:
		case 200: case 201: case 202: case 203: case 1334:
			return true;
	}
	return false;
}
static bool isUnsupportedId(int id) {
	switch (id) {
		case 13:
		case 47: case 111: case 660: case 745:
		case 1331: case 2926:
		case 286: case 287:
		case 101: case 99:
		case 45: case 46: case 747:
		case 67: case 84: case 1022: case 1330:
		case 1704: case 1751: case 1594:
			return true;
	}
	return false;
}
static float padVel(int id) { return id == 35 ? 16.f : (id == 140 ? 10.4f : 20.f); }
static float orbVel(int id) { return id == 36 ? 11.18f : (id == 141 ? 7.2f : 14.5f); }

static std::vector<Obj>::iterator lowerBoundX(std::vector<Obj>& v, float x0) {
	return std::lower_bound(v.begin(), v.end(), x0,
		[](Obj const& o, float x) { return o.r.x0 < x; });
}

// ---------------------------------------------------------------------------
// Simulator
// ---------------------------------------------------------------------------
struct Sim {
	float x = 0.f;
	float y = 0.f;
	float vy = 0.f;
	float sgn = 1.f;
	float speed = BASE_SPEED;
	bool ground = false;
	bool held = false;
	uint64_t mask = 0;
};

struct Band { float x, y; };

struct Env {
	std::vector<Obj> loc;
	float maxW = 30.f;
	float floorY = 90.f;
	float ceilY = 390.f;
	float margin = 2.f;
	float h = 0.25f;
	float limitX = 1e9f;
	bool useOrbs = true;
	std::vector<Band> const* forbid = nullptr;
};

static inline Rc box(Sim const& s, float h) {
	return {s.x - h, s.y - h, s.x + h, s.y + h};
}

static size_t firstIdx(Env const& e, float x) {
	float key = x - 25.f - e.maxW - e.margin;
	size_t lo = 0, hi = e.loc.size();
	while (lo < hi) {
		size_t mid = (lo + hi) / 2;
		if (e.loc[mid].r.x0 < key) lo = mid + 1; else hi = mid;
	}
	return lo;
}

static void flipGravity(Sim& s) { s.sgn = -s.sgn; s.y = -s.y; s.vy = -s.vy; }

static void applyPortal(Sim& s, int id) {
	switch (id) {
		case 10: if (s.sgn < 0.f) flipGravity(s); break;
		case 11: if (s.sgn > 0.f) flipGravity(s); break;
		case 200: s.speed = phys::SPEEDS[0]; break;
		case 201: s.speed = phys::SPEEDS[1]; break;
		case 202: s.speed = phys::SPEEDS[2]; break;
		case 203: s.speed = phys::SPEEDS[3]; break;
		case 1334: s.speed = phys::SPEEDS[4]; break;
	}
}

static bool stepFrame(Sim& s, bool press, Env const& e) {
	using namespace phys;
	bool edge = press && !s.held;
	s.held = press;
	float h = e.h;

	s.x += s.speed * (h / 60.f);
	float py = s.y;
	bool canJump = s.ground;
	s.ground = false;
	if (canJump && press) s.vy = JUMP;
	s.vy = std::max(s.vy - GRAVITY * h, -FALL_MAX);
	s.y += s.vy * h * POS_SCALE;

	Rc pr = box(s, HALF);
	float reach = HALF + e.margin + 1.f;
	for (size_t i = firstIdx(e, s.x); i < e.loc.size(); ++i) {
		auto const& o = e.loc[i];
		if (o.r.x0 > s.x + reach) break;
		if (o.r.x1 < s.x - reach) continue;
		Rc r = mir(o.r, s.sgn);

		if (o.k == K_HAZARD) {
			float m = e.margin;
			Rc hz{r.x0 - m, r.y0 - m, r.x1 + m, r.y1 + m};
			if (hit(pr, hz)) return false;
		} else if (o.k == K_SOLID) {
			if (!hit(pr, r)) continue;
			if (py - HALF >= r.y1 - 3.f && s.vy <= 0.f) {
				s.y = r.y1 + HALF; s.vy = 0.f; s.ground = true;
			} else if (py + HALF <= r.y0 + 3.f && s.vy >= 0.f) {
				s.y = r.y0 - HALF; s.vy = 0.f;
			} else {
				return false;
			}
			pr = box(s, HALF);
		} else {
			if (o.ix < 0 || ((s.mask >> o.ix) & 1ull) || !hit(pr, r)) continue;
			uint64_t bit = 1ull << o.ix;
			if (o.k == K_PAD) {
				s.vy = padVel(o.id); s.ground = false; s.mask |= bit;
			} else if (o.k == K_ORB) {
				if (edge && e.useOrbs) {
					s.vy = orbVel(o.id); s.ground = false; s.mask |= bit; edge = false;
				}
			} else {
				float before = s.sgn;
				applyPortal(s, o.id);
				s.mask |= bit;
				if (before != s.sgn) { py = -py; pr = box(s, HALF); }
			}
		}
	}

	float fl = s.sgn > 0.f ? e.floorY : -e.ceilY;
	float ce = s.sgn > 0.f ? e.ceilY : -e.floorY;
	if (s.y - HALF <= fl) {
		s.y = fl + HALF;
		if (s.vy < 0.f) s.vy = 0.f;
		s.ground = true;
	}
	if (s.y + HALF >= ce) {
		s.y = ce - HALF;
		if (s.vy > 0.f) s.vy = 0.f;
	}
	return true;
}

static bool touchingOrb(Sim const& s, Env const& e) {
	Rc pr = box(s, phys::HALF);
	for (size_t i = firstIdx(e, s.x); i < e.loc.size(); ++i) {
		auto const& o = e.loc[i];
		if (o.r.x0 > s.x + 20.f) break;
		if (o.r.x1 < s.x - 20.f) continue;
		if (o.k != K_ORB || o.ix < 0 || ((s.mask >> o.ix) & 1ull)) continue;
		if (hit(pr, mir(o.r, s.sgn))) return true;
	}
	return false;
}

static bool canPress(Sim const& s, Env const& e) {
	if (!e.forbid) return true;
	float wy = s.y * s.sgn;
	for (auto const& b : *e.forbid) {
		if (std::abs(s.x - b.x) < 3.5f && std::abs(wy - b.y) < 3.f) return false;
	}
	return true;
}

// ---------------------------------------------------------------------------
// Exact search
// ---------------------------------------------------------------------------
using Clock = std::chrono::steady_clock;

struct Searcher {
	Env const* e = nullptr;
	int H = 0;
	long budget = 0;
	long nodes = 0;
	bool timeUp = false;
	Clock::time_point t0;
	std::unordered_set<uint64_t> dead;
	std::vector<uint8_t> path;
	int bestT = -1;
	std::vector<uint8_t> bestPath;

	static uint64_t keyOf(Sim const& s, int t) {
		uint64_t h = 1469598103934665603ull;
		auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ull; };
		mix((uint32_t)t);
		mix((uint32_t)(int32_t)std::lround(s.y * 2.f));
		mix((uint32_t)(int32_t)std::lround(s.vy * 10.f));
		mix((s.ground ? 1u : 0u) | (s.sgn > 0.f ? 2u : 0u));
		mix(s.mask);
		mix((uint32_t)(int32_t)std::lround(s.speed));
		return h;
	}

	bool stop() {
		if (timeUp) return true;
		if (++nodes > budget) { timeUp = true; return true; }
		if ((nodes & 127) == 0) {
			double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
			if (ms > SEARCH_MS_LIMIT) { timeUp = true; return true; }
		}
		return false;
	}

	bool dfs(Sim s, int t) {
		size_t base = path.size();
		for (;;) {
			if (t >= H || s.x >= e->limitX) return true;
			if (t > bestT) { bestT = t; bestPath = path; }
			if (stop()) { path.resize(base); return false; }

			bool choice = s.ground || touchingOrb(s, *e);
			if (!choice) {
				if (!stepFrame(s, false, *e)) { path.resize(base); return false; }
				path.push_back(0);
				++t;
				continue;
			}

			uint64_t k = keyOf(s, t);
			if (dead.count(k)) { path.resize(base); return false; }

			Sim w = s;
			if (stepFrame(w, false, *e)) {
				path.push_back(0);
				if (dfs(w, t + 1)) return true;
				path.pop_back();
			}
			if (!timeUp && canPress(s, *e)) {
				Sim p = s;
				if (stepFrame(p, true, *e)) {
					path.push_back(1);
					if (dfs(p, t + 1)) return true;
					path.pop_back();
				}
			}
			if (!timeUp) dead.insert(k);
			path.resize(base);
			return false;
		}
	}
};

struct Plan {
	std::vector<uint8_t> acts;
	std::vector<float> px, py;
	size_t idx = 0;
	bool valid = false;
	bool full = false;
	bool clipped = false;
};

static void makePlan(Sim const& s0, Env const& e, int H, long budget, Plan& out) {
	static Searcher S;
	S.e = &e; S.H = H; S.budget = budget; S.nodes = 0; S.timeUp = false;
	S.t0 = Clock::now();
	S.dead.clear(); S.path.clear(); S.bestT = -1; S.bestPath.clear();

	bool ok = S.dfs(s0, 0);
	std::vector<uint8_t> acts = ok ? S.path : S.bestPath;

	if (ok && !S.timeUp) {
		auto it = std::find(acts.begin(), acts.end(), (uint8_t)1);
		if (it != acts.end()) {
			int p0 = (int)(it - acts.begin());
			std::vector<Sim> pre;
			Sim s = s0;
			for (int i = 0; i < p0; ++i) { pre.push_back(s); stepFrame(s, false, e); }
			int back = std::max(1, (int)(8.f / e.h));
			int cmin = p0;
			for (int c = p0 - 1; c >= 0 && c >= p0 - back; --c) {
				Sim const& st = pre[c];
				if (!(st.ground || touchingOrb(st, e)) || !canPress(st, e)) break;
				Sim q = st;
				if (!stepFrame(q, true, e)) break;
				S.path.assign(c, 0);
				S.path.push_back(1);
				if (S.dfs(q, c + 1)) cmin = c; else break;
				if (S.timeUp) break;
			}
			int c = (cmin + p0) / 2;
			if (cmin < p0 && c != p0 && !S.timeUp) {
				Sim q = pre[c];
				if (stepFrame(q, true, e)) {
					S.path.assign(c, 0);
					S.path.push_back(1);
					if (S.dfs(q, c + 1)) acts = S.path;
				}
			}
		}
	}

	out.acts = acts;
	out.px.clear(); out.py.clear();
	Sim s = s0;
	for (auto a : acts) {
		if (!stepFrame(s, a == 1, e)) break;
		out.px.push_back(s.x);
		out.py.push_back(s.y * s.sgn);
	}
	out.acts.resize(out.px.size());
	out.idx = 0;
	out.valid = true;
	out.full = ok;
	out.clipped = ok && (int)acts.size() < H;
}

// ===========================================================================
// Bot state
// ===========================================================================
struct Rec { float x, y, vN, predY, err; bool gr, act; };

struct Bot {
	std::vector<Obj> objs;
	std::vector<std::pair<float, int>> unsup;
	Plan plan;
	Env dbgEnv;

	bool holding = false, pressedLast = false;
	int clicks = 0, deaths = 0, phantom = 0, replans = 0;

	float prevX = -1.f, prevY = 0.f, prevH = 0.25f;
	float speed = BASE_SPEED;

	bool havePred = false;
	float predX = 0.f, predY = 0.f;
	float err = 0.f, maxErr = 0.f;
	bool diverged = false;
	bool forceReplan = false;
	int cool = 0;

	float floorY = 90.f, ceilY = 390.f;
	bool ceilLearned = false;

	Rec ring[48];
	int ringN = 0;
	int sincePress = 1000000;
	float pressX = 0.f, pressY = 0.f;
	std::vector<Band> forbid;

	double lastMs = 0.0, maxMs = 0.0;
	int aheadId = -1, aheadDist = 0;

	CCDrawNode* debug = nullptr;
	CCLabelBMFont* hud = nullptr;
	bool active = false;
	char const* modeStr = "cube";

	int drawVer = 0, drawnVer = -1;
	Clock::time_point lastHudT{}, lastDrawT{}, lastDeathLogT{};
	std::string lastHud;
};
static Bot G;

static void releaseBtn(GJBaseGameLayer* gl) {
	if (!G.holding) return;
	G.holding = false;
	gl->handleButton(false, 1, true);
}
static void setHold(GJBaseGameLayer* gl, bool want, PlayerObject* p) {
	if (want == G.holding) return;
	G.holding = want;
	if (want) {
		++G.clicks;
		G.pressedLast = true;
		G.sincePress = 0;
		G.pressX = p->getPositionX();
		G.pressY = p->getPositionY();
	}
	gl->handleButton(want, 1, true);
}

// ---------------------------------------------------------------------------
// One bot decision per physics step
// ---------------------------------------------------------------------------
static void botStep(GJBaseGameLayer* gl, PlayLayer* pl, float dt) {
	if (++C.age >= 60) refreshCfg();
	auto p = pl->m_player1;

	bool on = C.enabled;
	if (C.practiceOnly && !pl->m_isPracticeMode) on = false;
	G.modeStr = modeName(p);
	if (!on || p->m_isDead || dt <= 1e-5f || dt > 0.05f) {
		G.active = false;
		releaseBtn(gl);
		G.prevX = -1.f; G.havePred = false; G.plan.valid = false;
		return;
	}
	G.active = true;

	if (!isCube(p)) {
		releaseBtn(gl);
		G.prevX = -1.f; G.havePred = false; G.plan.valid = false;
		return;
	}

	auto pos = p->getPosition();
	float sgn = p->m_isUpsideDown ? -1.f : 1.f;
	float h = dt * 60.f;
	bool fresh = G.prevX < 0.f;
	if (G.cool > 0) --G.cool;

	// ---- telemetry --------------------------------------------------------
	float v = static_cast<float>(p->m_yVelocity);
	if (!fresh) {
		float dy = pos.y - G.prevY;
		if (std::abs(dy) > 0.02f && std::abs(v) > 0.3f && dy * v < 0.f) v = -v;
	}
	float vN = v * sgn;

	if (fresh) {
		int bi = 1; float bd = 1e9f;
		for (int i = 0; i < 5; ++i) {
			float d = std::abs((float)p->m_playerSpeed - phys::PSPEEDS[i]);
			if (d < bd) { bd = d; bi = i; }
		}
		G.speed = phys::SPEEDS[bi];
	} else {
		float sp = (pos.x - G.prevX) / (G.prevH / 60.f);
		for (float t : phys::SPEEDS) {
			if (std::abs(sp - t) < t * 0.06f) { G.speed = t; break; }
		}
	}

	// ---- prediction audit ---------------------------------------------------
	float stepErr = 0.f;
	if (G.havePred && !fresh) {
		float ex = pos.x - G.predX, ey = pos.y - G.predY;
		stepErr = std::max(std::abs(ex), std::abs(ey));
		G.err = ey;
		G.maxErr = std::max(G.maxErr, std::abs(ey));
		if (stepErr > DIV_TOL) {
			if (G.cool == 0) { G.forceReplan = true; G.cool = DIV_COOLDOWN; }
			if (!G.diverged) {
				G.diverged = true;
				int nearId = -1;
				for (auto it = lowerBoundX(G.objs, pos.x - 100.f); it != G.objs.end() && it->r.x0 < pos.x + 60.f; ++it) {
					Rc pr{pos.x - 25.f, pos.y - 25.f, pos.x + 25.f, pos.y + 25.f};
					if (hit(pr, it->r)) { nearId = it->id; break; }
				}
				log::info("[AutoBot] FIRST DIVERGENCE x={} y={} dx={} dy={} vRaw={} vN={} gnd={} sgn={} speed={} h={} nearObj={}",
					(double)pos.x, (double)pos.y, (double)ex, (double)ey, (double)p->m_yVelocity,
					(double)vN, p->m_isOnGround, (double)sgn, (double)G.speed, (double)h, nearId);
			}
		}
	}
	G.havePred = false;

	// ---- learn floor / ceiling (only plausible values, never from blocks) ----
	auto solidIn = [&](Rc q) {
		for (auto it = lowerBoundX(G.objs, pos.x - 200.f); it != G.objs.end() && it->r.x0 < pos.x + 40.f; ++it) {
			if (it->k == K_SOLID && hit(q, it->r)) return true;
		}
		return false;
	};
	if (p->m_isOnGround) {
		if (sgn > 0.f) {
			float cand = pos.y - phys::HALF;
			Rc below{pos.x - 14.f, pos.y - 19.f, pos.x + 14.f, pos.y - 14.5f};
			if (cand > 80.f && cand < 100.f && !solidIn(below)) G.floorY = cand;
		} else {
			float cand = pos.y + phys::HALF;
			Rc above{pos.x - 14.f, pos.y + 14.5f, pos.x + 14.f, pos.y + 19.f};
			if (cand > G.floorY + 240.f && cand < G.floorY + 420.f && !solidIn(above)) {
				G.ceilY = cand; G.ceilLearned = true;
			}
		}
	}

	// ---- mark consumed pads/portals/orbs -------------------------------------
	{
		Rc pr{pos.x - 15.f, pos.y - 15.f, pos.x + 15.f, pos.y + 15.f};
		for (auto it = lowerBoundX(G.objs, pos.x - 200.f); it != G.objs.end() && it->r.x0 < pos.x + 40.f; ++it) {
			if (it->used || !hit(pr, it->r)) continue;
			if (it->k == K_PAD || it->k == K_PORTAL) it->used = true;
			else if (it->k == K_ORB && G.pressedLast) it->used = true;
		}
	}
	G.pressedLast = false;
	G.prevX = pos.x; G.prevY = pos.y; G.prevH = h;
	if (G.sincePress < 1000000) ++G.sincePress;

	// ---- unsupported-mechanic detection (binary search, no string work) -------
	float limitX = 1e9f;
	G.aheadId = -1;
	{
		auto it = std::lower_bound(G.unsup.begin(), G.unsup.end(), std::pair<float, int>(pos.x - 15.f, INT_MIN));
		if (it != G.unsup.end()) {
			limitX = it->first - 22.f;
			G.aheadId = it->second;
			G.aheadDist = (int)(it->first - pos.x);
		}
	}

	// ---- decide whether to (re)plan -------------------------------------------
	int H = C.horizon;
	int Hs = std::max(8, (int)std::ceil((float)H / h));
	size_t remaining = G.plan.valid ? (G.plan.acts.size() > G.plan.idx ? G.plan.acts.size() - G.plan.idx : 0) : 0;
	bool need = !G.plan.valid || G.forceReplan || fresh ||
	            (G.plan.full ? remaining <= (size_t)(Hs / 2) : remaining == 0);
	if (G.plan.valid && G.plan.clipped && !G.forceReplan && !fresh && remaining > 0) need = false;

	if (need) {
		Env env;
		env.floorY = G.floorY;
		env.ceilY = G.ceilLearned ? G.ceilY : G.floorY + 300.f;
		env.margin = C.margin;
		env.useOrbs = C.useOrbs;
		env.h = h;
		env.limitX = limitX;
		env.forbid = &G.forbid;

		Sim s0;
		s0.x = pos.x; s0.sgn = sgn; s0.y = pos.y * sgn; s0.vy = vN;
		s0.speed = G.speed; s0.ground = p->m_isOnGround; s0.held = G.holding;

		float reachX = G.speed * (H / 60.f) + 90.f;
		int slot = 0;
		for (auto it = lowerBoundX(G.objs, pos.x - 200.f); it != G.objs.end() && it->r.x0 <= pos.x + reachX; ++it) {
			if (it->r.x1 < pos.x - 60.f) continue;
			Obj o = *it;
			if (o.k == K_ORB || o.k == K_PAD || o.k == K_PORTAL) {
				if (slot < 64) {
					o.ix = slot++;
					if (o.used) s0.mask |= 1ull << o.ix;
				}
			}
			env.maxW = std::max(env.maxW, o.r.x1 - o.r.x0);
			env.loc.push_back(o);
		}

		auto t0 = Clock::now();
		makePlan(s0, env, Hs, C.budget, G.plan);
		G.lastMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
		G.maxMs = std::max(G.maxMs, G.lastMs);
		G.forceReplan = false;
		++G.replans;
		G.dbgEnv = env;
		++G.drawVer;
	}

	// ---- execute the committed plan --------------------------------------------
	bool act = false;
	float predNY = pos.y;
	if (G.plan.valid && G.plan.idx < G.plan.acts.size()) {
		size_t i = G.plan.idx++;
		act = G.plan.acts[i] == 1;
		G.predX = G.plan.px[i];
		G.predY = G.plan.py[i];
		predNY = G.predY;
		G.havePred = true;
	}
	setHold(gl, act, p);

	Rec& r = G.ring[G.ringN % 48];
	r = {pos.x, pos.y, vN, predNY, stepErr, p->m_isOnGround, act};
	++G.ringN;
}

// ===========================================================================
// Physics-step hook (GD 2.2081: dt, isHalfTick, isLastTick)
// ===========================================================================
class $modify(BotGame, GJBaseGameLayer) {
	void processCommands(float dt, bool isHalfTick, bool isLastTick) {
		auto pl = PlayLayer::get();
		if (pl && static_cast<GJBaseGameLayer*>(pl) == this) {
			botStep(this, pl, dt);
		}
		GJBaseGameLayer::processCommands(dt, isHalfTick, isLastTick);
	}
};

// ===========================================================================
// PlayLayer hook
// ===========================================================================
static bool nearKnownObject(float x, float y) {
	Rc q{x - 22.f, y - 22.f, x + 22.f, y + 22.f};
	for (auto it = lowerBoundX(G.objs, x - 100.f); it != G.objs.end() && it->r.x0 < x + 60.f; ++it) {
		if (hit(q, it->r)) return true;
	}
	for (auto const& u : G.unsup) {
		if (std::abs(u.first - x) < 60.f) return true;
	}
	return false;
}

class $modify(BotPlay, PlayLayer) {
	void setupHasCompleted() {
		PlayLayer::setupHasCompleted();
		refreshCfg();

		G.objs.clear();
		G.unsup.clear();
		G.forbid.clear();
		G.plan = Plan();
		G.floorY = 90.f; G.ceilY = 390.f; G.ceilLearned = false;

		if (m_objects) {
			for (auto obj : CCArrayExt<GameObject*>(m_objects)) {
				int id = obj->m_objectID;
				Obj o;
				o.id = id;
				o.r = toRc(obj->getObjectRect());
				if (isPortalId(id))                                   o.k = K_PORTAL;
				else if (isOrbId(id))                                 o.k = K_ORB;
				else if (isPadId(id))                                 o.k = K_PAD;
				else if (isUnsupportedId(id) || obj->m_objectType == GameObjectType::Slope) {
					G.unsup.push_back({o.r.x0, id});
					continue;
				}
				else if (obj->m_objectType == GameObjectType::Hazard) o.k = K_HAZARD;
				else if (obj->m_objectType == GameObjectType::Solid)  o.k = K_SOLID;
				else continue;
				G.objs.push_back(o);
			}
		}
		std::sort(G.objs.begin(), G.objs.end(), [](Obj const& a, Obj const& b) { return a.r.x0 < b.r.x0; });
		std::sort(G.unsup.begin(), G.unsup.end());

		G.debug = CCDrawNode::create();
		m_objectLayer->addChild(G.debug, 1000);
		G.drawVer = 0; G.drawnVer = -1;

		G.hud = CCLabelBMFont::create("AutoBot", "bigFont.fnt");
		G.hud->setScale(0.35f);
		G.hud->setOpacity(190);
		G.hud->setAnchorPoint({0.f, 1.f});
		auto win = CCDirector::get()->getWinSize();
		G.hud->setPosition({6.f, win.height - 6.f});
		this->addChild(G.hud, 1000);
		G.lastHud.clear();

		log::info("[AutoBot] v0.6.0 ready: {} objects, {} unsupported. Physics: jump=11.18 gravity=0.864/frame posScale=0.9",
			G.objs.size(), G.unsup.size());
	}

	void onQuit() {
		G.debug = nullptr;
		G.hud = nullptr;
		G.plan = Plan();
		G.prevX = -1.f;
		PlayLayer::onQuit();
	}

	void resetLevel() {
		PlayLayer::resetLevel();
		for (auto& o : G.objs) o.used = false;
		G.holding = false;
		G.pressedLast = false;
		G.prevX = -1.f;
		G.havePred = false;
		G.plan.valid = false;
		G.diverged = false;
		G.err = 0.f;
		G.cool = 0;
		G.sincePress = 1000000;
		++G.drawVer;
	}

	void destroyPlayer(PlayerObject* player, GameObject* object) {
		if (player == m_player1 && !m_player1->m_isDead && G.active && C.enabled) {
			onDeath(object);
		}
		PlayLayer::destroyPlayer(player, object);
	}

	void onDeath(GameObject* killer) {
		auto pos = m_player1->getPosition();

		// Deaths with no level object near the player are not spikes (GD's anti-cheat
		// spike or another mod). Ignore them quietly and rarely log them.
		if (!nearKnownObject(pos.x, pos.y)) {
			++G.phantom;
			if (G.phantom <= 3 || G.phantom % 50 == 0) {
				log::info("[AutoBot] ignored death at x={} y={} killerId={}: no level object near the player (anti-cheat or another mod). {} so far",
					(double)pos.x, (double)pos.y, killer ? killer->m_objectID : -1, G.phantom);
			}
			return;
		}

		++G.deaths;
		char const* cls;
		bool addBand = false;
		if (G.diverged)                           { cls = "MODEL ERROR (sim diverged before death)"; addBand = true; }
		else if (G.plan.valid && !G.plan.full)    { cls = "PLANNER DEAD END (no survivable path found)"; }
		else                                      { cls = "UNSEEN (plan was clean: unmodelled object/hitbox)"; addBand = true; }

		auto now = Clock::now();
		bool dump = std::chrono::duration<double>(now - G.lastDeathLogT).count() > 3.0;
		log::info("[AutoBot] death #{} x={} y={} killerId={} speed={} class={} replans={} maxErr={} lastSearch={}ms",
			G.deaths, (double)pos.x, (double)pos.y, killer ? killer->m_objectID : -1,
			(double)G.speed, cls, G.replans, (double)G.maxErr, G.lastMs);

		if (dump) {
			G.lastDeathLogT = now;
			int n = std::min(G.ringN, 12);
			for (int i = n; i >= 1; --i) {
				Rec const& r = G.ring[(G.ringN - i) % 48];
				log::info("[AutoBot]   t-{} x={} y={} vN={} gnd={} act={} predY={} err={}",
					i, (double)r.x, (double)r.y, (double)r.vN, r.gr, r.act, (double)r.predY, (double)r.err);
			}
		}

		if (!C.learn) return;
		if (addBand && G.sincePress < 160) {
			if (G.forbid.size() >= 64) G.forbid.erase(G.forbid.begin());
			G.forbid.push_back({G.pressX, G.pressY});
			log::info("[AutoBot] forbidden click band at x={} y={} ({} total)",
				(double)G.pressX, (double)G.pressY, G.forbid.size());
		}
	}

	void postUpdate(float dt) {
		PlayLayer::postUpdate(dt);
		updateHud();
		drawDebug();
	}

	// HUD: at most 4x per second, only if the text changed.
	void updateHud() {
		if (!G.hud) return;
		auto now = Clock::now();
		if (std::chrono::duration<double>(now - G.lastHudT).count() < 0.25) return;
		G.lastHudT = now;

		char buf[200];
		std::snprintf(buf, sizeof(buf), "AutoBot %s | %s%s | clk %d dth %d | plan %d%s | err %.2f/%.2f | %.1fms",
			G.active ? "ON" : "OFF", G.modeStr, (G.active && !isCube(m_player1)) ? " (cube only)" : "",
			G.clicks, G.deaths,
			(int)(G.plan.valid ? G.plan.acts.size() - std::min(G.plan.idx, G.plan.acts.size()) : 0),
			(G.plan.valid && !G.plan.full) ? " DEADEND" : "",
			(double)G.err, (double)G.maxErr, G.lastMs);
		std::string s = buf;
		if (G.aheadId >= 0 && G.aheadDist < 600) {
			char b2[64];
			std::snprintf(b2, sizeof(b2), " | unsupported id %d in %d u", G.aheadId, G.aheadDist);
			s += b2;
		}
		if (s == G.lastHud) return;
		G.lastHud = s;
		G.hud->setString(s.c_str());
	}

	// Debug overlay: rebuilt only after a new plan, at most ~6x per second, nearby objects only.
	void drawDebug() {
		auto d = G.debug;
		if (!d) return;
		if (!G.active || !C.debugDraw) {
			if (G.drawnVer != -2) { d->clear(); G.drawnVer = -2; }
			return;
		}
		if (G.drawnVer == G.drawVer) return;
		auto now = Clock::now();
		if (std::chrono::duration<double>(now - G.lastDrawT).count() < 0.15) return;
		G.lastDrawT = now;
		G.drawnVer = G.drawVer;
		d->clear();

		auto drawRect = [&](Rc const& r, ccColor4F col) {
			CCPoint v[4] = {{r.x0, r.y0}, {r.x1, r.y0}, {r.x1, r.y1}, {r.x0, r.y1}};
			d->drawPolygon(v, 4, {0, 0, 0, 0}, 1.f, col);
		};
		float px = m_player1->getPositionX();
		int drawn = 0;
		for (auto const& o : G.dbgEnv.loc) {
			if (o.r.x1 < px - 20.f || o.r.x0 > px + 500.f) continue;
			if (++drawn > 120) break;
			ccColor4F col = o.k == K_HAZARD ? ccColor4F{1.f, 0.2f, 0.2f, 1.f}
			              : o.k == K_SOLID  ? ccColor4F{0.3f, 0.6f, 1.f, 0.8f}
			              : o.k == K_PORTAL ? ccColor4F{0.8f, 0.4f, 1.f, 1.f}
			                                : ccColor4F{1.f, 0.9f, 0.2f, 1.f};
			drawRect(o.r, col);
		}
		for (auto const& u : G.unsup) {
			if (u.first < px || u.first > px + 600.f) continue;
			drawRect({u.first, 0.f, u.first + 30.f, 900.f}, {1.f, 0.5f, 0.f, 0.35f});
		}
		if (G.plan.valid && !G.plan.px.empty()) {
			CCPoint prev = {G.plan.px[0], G.plan.py[0]};
			for (size_t i = 1; i < G.plan.px.size(); ++i) {
				bool click = G.plan.acts[i] != 0;
				if (!click && (i % 6) != 0 && i + 1 != G.plan.px.size()) continue;
				CCPoint cur = {G.plan.px[i], G.plan.py[i]};
				d->drawSegment(prev, cur, 1.2f, click ? ccColor4F{1.f, 1.f, 1.f, 1.f} : ccColor4F{0.2f, 1.f, 0.4f, 1.f});
				prev = cur;
			}
		}
	}
};
