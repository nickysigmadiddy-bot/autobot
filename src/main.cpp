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
#include <utility>
#include <vector>

using namespace geode::prelude;

// ===========================================================================
// AutoBot v0.8.0 - cube only.
//  * hitbox relaxation ladder + "doom" fallback (never just walk into death)
//  * trial-and-error timing learned from REAL deaths (the game is the judge)
//  * every click placed in the middle of its safe window, click latency measured
// ===========================================================================

constexpr float BASE_SPEED = 311.58f;
constexpr double SEARCH_MS_FULL = 3.5;
constexpr double SEARCH_MS_DEADEND = 1.0;
constexpr float DIV_TOL = 0.8f;
constexpr int DIV_COOLDOWN = 24;
constexpr float MIN_PLAN_H = 0.25f;     // 240 steps per second at most
constexpr float WALL_SLACK = 90.f;
constexpr int SETTLE_STEPS = 60;
constexpr int LEAD_OPEN = 12;
constexpr size_t MAX_EXPS = 24;
namespace phys {
	constexpr float HALF = 15.f;
	constexpr float GRAVITY = 0.864f;
	constexpr float JUMP = 11.18f;
	constexpr float POS_SCALE = 0.9f;
	constexpr float FALL_MAX = 15.f;
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
// Settings cache
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
	bool held = false;     // effective (already applied) button state
	uint8_t q = 0;         // last decided actions, bit0 = newest (for click latency)
	uint64_t mask = 0;
};

struct Env {
	std::vector<Obj> loc;
	float maxW = 30.f;
	float floorY = 90.f;
	float ceilY = 390.f;
	float margin = 2.f;
	float hzHalf = 15.f;   // half size of the player box used against hazards
	float h = 0.25f;
	float limitX = 1e9f;
	bool useOrbs = true;
	int lat = 0;           // click latency in planning steps (0..3)
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

// 'press' is the DECIDED action this step. It becomes effective e.lat steps later.
static bool stepFrame(Sim& s, bool press, Env const& e) {
	using namespace phys;
	s.q = (uint8_t)(((s.q << 1) | (press ? 1u : 0u)) & 0xFu);
	bool eff = ((s.q >> e.lat) & 1u) != 0u;
	bool edge = eff && !s.held;
	s.held = eff;
	float h = e.h;

	s.x += s.speed * (h / 60.f);
	float py = s.y;
	bool canJump = s.ground;
	s.ground = false;
	if (canJump && eff) s.vy = JUMP;
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
			Rc hb = box(s, e.hzHalf);
			if (hit(hb, hz)) return false;
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

// A click can only matter if some non-portal object is close enough ahead.
static bool pressRelevant(Sim const& s, Env const& e) {
	for (size_t i = firstIdx(e, s.x); i < e.loc.size(); ++i) {
		auto const& o = e.loc[i];
		if (o.r.x0 > s.x + 360.f) break;
		if (o.k != K_PORTAL && o.r.x1 > s.x - 40.f) return true;
	}
	return false;
}

// ---------------------------------------------------------------------------
// Fixed-size dead-state table (no allocation after construction)
// ---------------------------------------------------------------------------
struct DeadSet {
	static constexpr size_t N = 1u << 15;
	static constexpr size_t MASK = N - 1;
	std::vector<uint64_t> keys;
	std::vector<uint32_t> stamp;
	uint32_t cur = 1;
	size_t cnt = 0;
	DeadSet() : keys(N, 0), stamp(N, 0) {}
	void clear() {
		if (++cur == 0) { std::fill(stamp.begin(), stamp.end(), 0u); cur = 1; }
		cnt = 0;
	}
	bool has(uint64_t k) const {
		size_t i = (size_t)(k ^ (k >> 29)) & MASK;
		while (stamp[i] == cur) {
			if (keys[i] == k) return true;
			i = (i + 1) & MASK;
		}
		return false;
	}
	void add(uint64_t k) {
		if (cnt * 10 > N * 6) return;
		size_t i = (size_t)(k ^ (k >> 29)) & MASK;
		while (stamp[i] == cur) {
			if (keys[i] == k) return;
			i = (i + 1) & MASK;
		}
		stamp[i] = cur; keys[i] = k; ++cnt;
	}
};

// ---------------------------------------------------------------------------
// Exact search
// ---------------------------------------------------------------------------
using Clock = std::chrono::steady_clock;

struct Searcher {
	Env const* e = nullptr;
	int H = 0;
	long budget = 0;
	long nodes = 0;
	double msLimit = SEARCH_MS_FULL;
	bool timeUp = false;
	Clock::time_point t0;
	DeadSet dead;
	std::vector<uint8_t> path;
	int bestT = -1;
	std::vector<uint8_t> bestPath;

	uint64_t keyOf(Sim const& s, int t) const {
		uint64_t h = 1469598103934665603ull;
		auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ull; };
		mix((uint32_t)t);
		mix((uint32_t)(int32_t)std::lround(s.y * 2.f));
		mix((uint32_t)(int32_t)std::lround(s.vy * 10.f));
		mix((s.ground ? 1u : 0u) | (s.sgn > 0.f ? 2u : 0u) | (s.held ? 4u : 0u));
		mix(s.mask);
		mix((uint32_t)(int32_t)std::lround(s.speed));
		mix((uint32_t)(s.q & ((1u << e->lat) - 1u)));
		return h;
	}

	bool stop() {
		if (timeUp) return true;
		if (++nodes > budget) { timeUp = true; return true; }
		if ((nodes & 63) == 0) {
			double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
			if (ms > msLimit) { timeUp = true; return true; }
		}
		return false;
	}

	bool dfs(Sim s, int t) {
		size_t base = path.size();
		for (;;) {
			// A plan only counts as safe once the cube has landed (or after a generous settle window).
			if (t >= H && (s.ground || t >= H + SETTLE_STEPS)) return true;
			if (s.x >= e->limitX && (s.ground || s.x >= e->limitX + WALL_SLACK)) return true;
			if (t > bestT) { bestT = t; bestPath = path; }
			if (stop()) { path.resize(base); return false; }

			bool orb = touchingOrb(s, *e);
			bool choice = s.ground || orb;
			if (!choice) {
				if (!stepFrame(s, false, *e)) { path.resize(base); return false; }
				path.push_back(0);
				++t;
				continue;
			}

			uint64_t k = keyOf(s, t);
			if (dead.has(k)) { path.resize(base); return false; }

			Sim w = s;
			if (stepFrame(w, false, *e)) {
				path.push_back(0);
				if (dfs(w, t + 1)) return true;
				path.pop_back();
			}
			if (!timeUp && (orb || pressRelevant(s, *e))) {
				Sim p = s;
				if (stepFrame(p, true, *e)) {
					path.push_back(1);
					if (dfs(p, t + 1)) return true;
					path.pop_back();
				}
			}
			if (!timeUp) dead.add(k);
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
	bool timedOut = false;
	bool doom = false;
	long nodes = 0;
	int winMin = -1, winMax = -1, winPick = -1;
	bool winBounded = false;
};

// The sim sees no survivable path. Instead of walking blindly into it, pick the single click
// time that survives the longest (middle of the best range). Waiting is kept if it is as good.
static bool doomFallback(Sim const& s0, Env const& e, int H, std::vector<uint8_t>& out) {
	int maxT = H + SETTLE_STEPS;
	auto depth = [&](int c) {
		Sim s = s0;
		for (int t = 0; t < maxT; ++t) {
			if (!stepFrame(s, t == c, e)) return t;
		}
		return maxT;
	};
	int best = depth(-1);
	std::vector<int> bestC;
	auto t0 = Clock::now();
	for (int c = 0; c < 160; c += 2) {
		int d = depth(c);
		if (d > best) { best = d; bestC.clear(); bestC.push_back(c); }
		else if (d == best && !bestC.empty()) bestC.push_back(c);
		if ((c & 15) == 0 && std::chrono::duration<double, std::milli>(Clock::now() - t0).count() > 3.0) break;
	}
	if (bestC.empty()) return false;
	int c = bestC[bestC.size() / 2];
	out.assign((size_t)c, 0);
	out.push_back(1);
	for (int t = c + 1; t < best; ++t) out.push_back(0);
	return true;
}

// The DFS finds the LATEST click that survives (it always waits first), which is the edge of the
// safe window. Here every click (up to 3 per plan) is moved to the middle of its own window.
static void makePlan(Sim const& s0, Env const& e, int H, long budget, double msLimit, int bias, Plan& out, bool allowDoom) {
	static Searcher S;
	S.e = &e; S.H = H; S.budget = budget; S.nodes = 0; S.timeUp = false;
	S.msLimit = msLimit;
	S.t0 = Clock::now();
	S.dead.clear(); S.path.clear(); S.bestT = -1; S.bestPath.clear();

	out.winMin = out.winMax = out.winPick = -1;
	out.winBounded = false;
	out.doom = false;

	bool ok = S.dfs(s0, 0);
	std::vector<uint8_t> cur = ok ? S.path : S.bestPath;

	if (!ok && !S.timeUp && allowDoom) {
		std::vector<uint8_t> d;
		if (doomFallback(s0, e, H, d)) { cur = d; out.doom = true; }
	}

	if (ok && !S.timeUp) {
		auto isStart = [&](size_t i) {
			uint8_t prev = (i == 0) ? (uint8_t)(s0.q & 1u) : cur[i - 1];
			return cur[i] == 1 && prev == 0;
		};
		size_t from = 0;
		for (int iter = 0; iter < 3 && !S.timeUp; ++iter) {
			int p0 = -1;
			for (size_t i = from; i < cur.size(); ++i) {
				if (isStart(i)) { p0 = (int)i; break; }
			}
			if (p0 < 0) break;

			std::vector<Sim> pre;
			pre.reserve((size_t)p0);
			Sim s = s0;
			for (int i = 0; i < p0; ++i) { pre.push_back(s); stepFrame(s, cur[i] == 1, e); }

			int back = std::max(1, (int)(10.f / e.h));
			int lo = std::max((int)from, p0 - back);
			int cmin = p0;
			bool bounded = false;
			for (int c = p0 - 1; c >= lo; --c) {
				Sim const& st = pre[c];
				if (!(st.ground || touchingOrb(st, e))) { bounded = true; break; }
				Sim q = st;
				if (!stepFrame(q, true, e)) { bounded = true; break; }
				S.path.assign(cur.begin(), cur.begin() + c);
				S.path.push_back(1);
				if (S.dfs(q, c + 1)) {
					cmin = c;
				} else {
					if (!S.timeUp) bounded = true;
					break;
				}
			}
			if (S.timeUp) break;

			int width = p0 - cmin;
			int off = std::clamp(bias, -(width / 4), width / 4);
			int target;
			if (width == 0)    target = p0;
			else if (bounded)  target = (cmin + p0) / 2 - off;
			else               target = p0 - LEAD_OPEN - off;
			target = std::clamp(target, cmin, p0);

			if (iter == 0) {
				out.winMin = cmin; out.winMax = p0; out.winPick = target; out.winBounded = bounded;
			}

			if (target != p0) {
				Sim q = pre[target];
				if (stepFrame(q, true, e)) {
					S.path.assign(cur.begin(), cur.begin() + target);
					S.path.push_back(1);
					if (S.dfs(q, target + 1)) cur = S.path;
				}
			}
			from = (size_t)target + 1;
		}
	}

	std::vector<uint8_t> acts = cur;
	out.acts = acts;
	out.px.clear(); out.py.clear();
	out.px.reserve(acts.size()); out.py.reserve(acts.size());
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
	out.timedOut = S.timeUp;
	out.nodes = S.nodes;
}

// ===========================================================================
// Trial-and-error learning. The game itself is the judge: after a REAL death the bot blames a click
// (or a missing click) and tries a different click position on the next attempt, one frame at a
// time, until the obstacle is passed. Positions repeat exactly because the physics is deterministic.
// ===========================================================================
struct Exp {
	float base = 0.f;      // reference x (the culprit click, or the place the missing click belongs before)
	float F = 5.19f;       // units per frame when created
	int n = 0;             // failed attempts so far
	int dir = -1;          // shift experiments: first direction (-1 earlier, +1 later)
	bool shift = true;     // true: move an existing click; false: add a click before 'base'
	bool air = false;      // the culprit click was an air click (orb)
	bool fired = false;    // already handled this attempt
	bool dead = false;     // gave up
	float xt = 0.f;        // x where the click is forced in the current attempt
};

static void expTarget(Exp& e) {
	if (e.shift) {
		int k = e.n / 2 + 1;
		int sgn = (e.n % 2 == 0) ? e.dir : -e.dir;
		e.xt = e.base + (float)(sgn * k) * e.F;
	} else {
		e.xt = e.base - (float)(e.n + 1) * e.F;
	}
}
static bool expExhausted(Exp const& e) {
	return e.shift ? (e.n / 2 + 1 > 10) : (e.n >= 26);
}

// ===========================================================================
// Bot state
// ===========================================================================
struct Rec { float x, y, vN, predY, err; bool gr, act; };

struct Perf {
	double stepUsSum = 0.0, stepUsMax = 0.0;
	double gameUsSum = 0.0, gameUsMax = 0.0;
	long steps = 0;
	double planMsSum = 0.0, planMsMax = 0.0;
	long plans = 0;
	double uiUsSum = 0.0, uiUsMax = 0.0;
	long frames = 0;
	Clock::time_point winStart{};
};

struct Bot {
	std::vector<Obj> objs;
	std::vector<std::pair<float, int>> unsup;
	Plan plan;
	Env dbgEnv;          // also used as the planner's working environment

	bool holding = false, pressedLast = false;
	int clicks = 0, deaths = 0, phantom = 0, replans = 0;

	float prevX = -1.f, prevY = 0.f;
	float sinceTick = 0.f;
	float planH = 0.25f;
	float speed = BASE_SPEED;

	bool havePred = false;
	float predX = 0.f, predY = 0.f;
	float err = 0.f, maxErr = 0.f;
	bool diverged = false;
	int divLogged = 0;
	int sinceDiv = 1000000;
	bool forceReplan = false;
	int cool = 0;
	int noReplan = 0;
	int rung = 0;

	// click timing
	int lat = 0;
	int latCnt[4] = {0, 0, 0, 0};
	int latWait = 0, latAge = 0;
	uint8_t qHist = 0;

	// experiments
	std::vector<Exp> exps;
	float lastGroundX = 0.f;
	bool pressAir = false;
	int expFires = 0;

	float floorY = 90.f, ceilY = 390.f;
	bool ceilLearned = false;

	Rec ring[48];
	int ringN = 0;
	int sincePress = 1000000;
	float pressX = 0.f, pressY = 0.f;

	double lastMs = 0.0, maxMs = 0.0;
	int aheadId = -1, aheadDist = 0;

	CCDrawNode* debug = nullptr;
	CCDrawNode* pathNode = nullptr;
	CCLabelBMFont* hud = nullptr;
	bool active = false;
	char const* modeStr = "cube";

	int drawVer = 0, drawnVer = -1, pathVer = -1, lastPathIdx = -1;
	Clock::time_point lastHudT{}, lastDrawT{}, lastPathT{}, lastDeathLogT{}, lastDeadLogT{};
	std::string lastHud;
	Perf perf;
	double stepAvgUs = 0.0;
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
		G.pressAir = !p->m_isOnGround;
	}
	gl->handleButton(want, 1, true);
}

static void resetTracking() {
	G.prevX = -1.f; G.havePred = false; G.plan.valid = false; G.sinceTick = 0.f;
	G.qHist = 0; G.latWait = 0;
}

static void perfTick(Clock::time_point now) {
	auto& P = G.perf;
	if (P.winStart.time_since_epoch().count() == 0) { P.winStart = now; return; }
	double el = std::chrono::duration<double>(now - P.winStart).count();
	if (el < 5.0) return;
	if (G.active && P.steps > 0 && P.frames > 0) {
		G.stepAvgUs = P.stepUsSum / (double)P.steps;
		log::info("[AutoBot] perf: {:.0f} render fps | {:.0f} steps/s | bot {:.1f}us avg {:.0f}us max | game step {:.1f}us avg {:.0f}us max | {} plans {:.2f}ms avg {:.2f}ms max | hud+overlay {:.1f}us avg {:.0f}us max",
			(double)P.frames / el, (double)P.steps / el,
			G.stepAvgUs, P.stepUsMax,
			P.gameUsSum / (double)P.steps, P.gameUsMax,
			P.plans, P.plans ? P.planMsSum / (double)P.plans : 0.0, P.planMsMax,
			P.uiUsSum / (double)P.frames, P.uiUsMax);
	}
	P = Perf();
	P.winStart = now;
}

// ---------------------------------------------------------------------------
// One bot decision per planning tick (never faster than 240 per second)
// ---------------------------------------------------------------------------
static void botStep(GJBaseGameLayer* gl, PlayLayer* pl, float dt) {
	if (++C.age >= 240) refreshCfg();
	auto p = pl->m_player1;

	bool on = C.enabled;
	if (C.practiceOnly && !pl->m_isPracticeMode) on = false;
	G.modeStr = modeName(p);
	if (!on || p->m_isDead || dt <= 1e-6f || dt > 0.05f) {
		G.active = false;
		releaseBtn(gl);
		resetTracking();
		return;
	}
	G.active = true;

	if (!isCube(p)) {
		releaseBtn(gl);
		resetTracking();
		return;
	}

	// ---- tick gating: plan at max 240 steps/s no matter the render FPS -------
	float hRaw = dt * 60.f;
	G.sinceTick += hRaw;
	bool fresh = G.prevX < 0.f;
	float h = std::max(MIN_PLAN_H, std::round(hRaw * 4.f) / 4.f);
	if (!fresh && G.sinceTick < h - 0.02f) return;
	float elapsed = fresh ? h : G.sinceTick;
	G.sinceTick = 0.f;

	auto pos = p->getPosition();
	float sgn = p->m_isUpsideDown ? -1.f : 1.f;
	if (G.cool > 0) --G.cool;
	if (G.noReplan > 0) --G.noReplan;
	if (G.plan.valid && std::abs(G.planH - h) > 0.01f) G.forceReplan = true;
	if (p->m_isOnGround) G.lastGroundX = pos.x;

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
		float sp = (pos.x - G.prevX) / (elapsed / 60.f);
		for (float t : phys::SPEEDS) {
			if (std::abs(sp - t) < t * 0.06f) { G.speed = t; break; }
		}
	}

	// ---- prediction audit ---------------------------------------------------
	if (G.sinceDiv < 1000000) ++G.sinceDiv;
	float stepErr = 0.f;
	if (G.havePred && !fresh) {
		float ex = pos.x - G.predX, ey = pos.y - G.predY;
		stepErr = std::max(std::abs(ex), std::abs(ey));
		G.err = ey;
		G.maxErr = std::max(G.maxErr, std::abs(ey));
		if (stepErr > DIV_TOL) {
			G.sinceDiv = 0;
			G.diverged = true;
			if (G.cool == 0) { G.forceReplan = true; G.cool = DIV_COOLDOWN; }
			if (G.divLogged < 5) {
				++G.divLogged;
				int nearId = -1;
				for (auto it = lowerBoundX(G.objs, pos.x - 100.f); it != G.objs.end() && it->r.x0 < pos.x + 60.f; ++it) {
					Rc pr{pos.x - 25.f, pos.y - 25.f, pos.x + 25.f, pos.y + 25.f};
					if (hit(pr, it->r)) { nearId = it->id; break; }
				}
				log::info("[AutoBot] DIVERGENCE #{} x={} y={} dx={} dy={} vRaw={} vN={} gnd={} sgn={} speed={} h={} nearObj={}",
					G.divLogged, (double)pos.x, (double)pos.y, (double)ex, (double)ey, (double)p->m_yVelocity,
					(double)vN, p->m_isOnGround, (double)sgn, (double)G.speed, (double)h, nearId);
			}
		}
	}
	G.havePred = false;

	// ---- measure click latency: decision tick -> visible jump velocity ---------
	if (G.latWait) {
		++G.latAge;
		if (vN > 5.f) {
			int meas = G.latAge - 1;
			G.latWait = 0;
			if (meas >= 0 && meas <= 3) {
				++G.latCnt[meas];
				int best = 0;
				for (int i = 1; i < 4; ++i) {
					if (G.latCnt[i] > G.latCnt[best]) best = i;
				}
				if (best != G.lat) {
					G.lat = best;
					G.forceReplan = true;
					log::info("[AutoBot] click latency is now {} step(s) (samples: {} / {} / {} / {})",
						G.lat, G.latCnt[0], G.latCnt[1], G.latCnt[2], G.latCnt[3]);
				}
			}
		} else if (G.latAge >= 6) {
			G.latWait = 0;
		}
	}

	// ---- learn floor / ceiling ----------------------------------------------
	auto solidIn = [&](Rc q) {
		for (auto it = lowerBoundX(G.objs, pos.x - 200.f); it != G.objs.end() && it->r.x0 < pos.x + 40.f; ++it) {
			if (it->k == K_SOLID && hit(q, it->r)) return true;
		}
		return false;
	};
	auto padOrOrbNear = [&]() {
		Rc q{pos.x - 40.f, pos.y - 40.f, pos.x + 40.f, pos.y + 40.f};
		for (auto it = lowerBoundX(G.objs, pos.x - 100.f); it != G.objs.end() && it->r.x0 < pos.x + 60.f; ++it) {
			if ((it->k == K_PAD || it->k == K_ORB) && hit(q, it->r)) return true;
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
	G.prevX = pos.x; G.prevY = pos.y;
	if (G.sincePress < 1000000) ++G.sincePress;

	// ---- unsupported-mechanic detection ---------------------------------------
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
	if (G.noReplan > 0 && G.plan.valid && !G.forceReplan && !fresh) need = false;

	if (need) {
		bool wasFull = G.plan.valid && G.plan.full;
		bool keepable = wasFull && remaining > 0 && !G.forceReplan && !fresh;
		Plan backup;
		if (keepable) backup = G.plan;

		Env& env = G.dbgEnv;
		env.loc.clear();
		env.maxW = 30.f;
		env.floorY = G.floorY;
		env.ceilY = G.ceilLearned ? G.ceilY : G.floorY + 300.f;
		env.useOrbs = C.useOrbs;
		env.h = h;
		env.limitX = limitX;
		env.lat = G.lat;

		Sim s0;
		s0.x = pos.x; s0.sgn = sgn; s0.y = pos.y * sgn; s0.vy = vN;
		s0.speed = G.speed; s0.ground = p->m_isOnGround;
		s0.q = G.qHist;
		s0.held = ((G.qHist >> G.lat) & 1u) != 0u;

		float reachX = G.speed * (H / 60.f) + 90.f + (SETTLE_STEPS * h / 60.f) * G.speed;
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

		double lim = (fresh || wasFull || !G.plan.valid) ? SEARCH_MS_FULL : SEARCH_MS_DEADEND;
		auto t0 = Clock::now();

		// Hitbox ladder: normal hitboxes first. Only if no survivable path exists, relax the player's
		// hazard box step by step. The last rung also allows the "best single click" fallback.
		struct Rung { float margin, hz; };
		Rung rungs[4] = {{C.margin, 15.f}, {0.f, 15.f}, {0.f, 12.f}, {0.f, 9.f}};
		int used = 0;
		for (int ri = 0; ri < 4; ++ri) {
			env.margin = rungs[ri].margin;
			env.hzHalf = rungs[ri].hz;
			makePlan(s0, env, Hs, C.budget, ri == 0 ? lim : SEARCH_MS_DEADEND, 0, G.plan, ri == 3);
			used = ri;
			if (G.plan.full || G.plan.timedOut) break;
		}
		G.rung = used;
		if (used > 0 && G.plan.full) {
			static int rungLogs = 0;
			if (rungLogs < 8) {
				++rungLogs;
				log::info("[AutoBot] no safe path with normal hitboxes at x={}; found one with relaxed rung {} (player hazard box {})",
					(double)pos.x, used, (double)rungs[used].hz);
			}
		}

		// The search ran out of time and found nothing: the old plan is still better than guessing.
		if (keepable && !G.plan.full && G.plan.timedOut) {
			G.plan = std::move(backup);
			G.noReplan = 12;
		} else if (!G.plan.full) {
			G.noReplan = 3;
		}

		G.lastMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
		G.maxMs = std::max(G.maxMs, G.lastMs);
		G.perf.planMsSum += G.lastMs;
		G.perf.planMsMax = std::max(G.perf.planMsMax, G.lastMs);
		++G.perf.plans;
		G.forceReplan = false;
		G.planH = h;
		++G.replans;
		++G.drawVer;

		if (!G.plan.full) {
			auto now = Clock::now();
			if (std::chrono::duration<double>(now - G.lastDeadLogT).count() > 1.0) {
				G.lastDeadLogT = now;
				log::info("[AutoBot] dead-end plan at x={} y={} vN={} gnd={} | sim survives {} steps, dies near x={} | doomFallback={} timedOut={} nodes={} wallId={} wallDist={}",
					(double)pos.x, (double)pos.y, (double)vN, p->m_isOnGround, G.plan.acts.size(),
					(double)(G.plan.px.empty() ? pos.x : G.plan.px.back()),
					G.plan.doom, G.plan.timedOut, G.plan.nodes, G.aheadId, G.aheadDist);
			}
		}
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

	// ---- trial-and-error experiments (timing learned from real deaths) ---------
	if (C.learn && !G.exps.empty()) {
		bool overridden = false;
		for (auto& ex : G.exps) {
			if (ex.dead || ex.fired) continue;
			float lo = ex.shift ? std::min(ex.base - 45.f, ex.xt - 5.f) : ex.xt - 1.f;
			if (pos.x < lo) continue;
			if (pos.x < ex.xt) {
				if (ex.shift) { act = false; overridden = true; }   // hold the original click back
				continue;
			}
			ex.fired = true;                                        // reached the target x: handle once
			G.forceReplan = true;
			if (p->m_isOnGround || ex.air) {
				act = true;
				overridden = true;
				++G.expFires;
			}
		}
		if (overridden) G.havePred = false;
	}

	bool wasHolding = G.holding;
	setHold(gl, act, p);
	G.qHist = (uint8_t)(((G.qHist << 1) | (act ? 1u : 0u)) & 0xFu);

	// Arm a latency measurement on clean ground clicks (not near pads/orbs).
	if (act && !wasHolding && p->m_isOnGround && std::abs(vN) < 0.5f && !G.latWait && !padOrOrbNear()) {
		G.latWait = 1;
		G.latAge = 0;
	}

	Rec& r = G.ring[G.ringN % 48];
	r = {pos.x, pos.y, vN, predNY, stepErr, p->m_isOnGround, act};
	++G.ringN;
}

// ===========================================================================
// Physics-step hook
// ===========================================================================
class $modify(BotGame, GJBaseGameLayer) {
	void processCommands(float dt, bool isHalfTick, bool isLastTick) {
		auto pl = PlayLayer::get();
		if (pl && static_cast<GJBaseGameLayer*>(pl) == this) {
			auto t0 = Clock::now();
			botStep(this, pl, dt);
			auto t1 = Clock::now();
			GJBaseGameLayer::processCommands(dt, isHalfTick, isLastTick);
			auto t2 = Clock::now();
			double botUs = std::chrono::duration<double, std::micro>(t1 - t0).count();
			double gameUs = std::chrono::duration<double, std::micro>(t2 - t1).count();
			G.perf.stepUsSum += botUs;
			G.perf.stepUsMax = std::max(G.perf.stepUsMax, botUs);
			G.perf.gameUsSum += gameUs;
			G.perf.gameUsMax = std::max(G.perf.gameUsMax, gameUs);
			++G.perf.steps;
			return;
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
		G.exps.clear();
		G.plan = Plan();
		G.floorY = 90.f; G.ceilY = 390.f; G.ceilLearned = false;
		G.sinceTick = 0.f;
		G.sinceDiv = 1000000; G.divLogged = 0;
		G.qHist = 0; G.latWait = 0; G.noReplan = 0; G.rung = 0;
		G.perf = Perf();
		G.dbgEnv.loc.reserve(512);

		int skippedStart = 0;
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
				else if (obj->m_objectType == GameObjectType::Hazard) {
					// A hazard overlapping the spawn point is GD's own start/anti-cheat spike, not level geometry.
					if (o.r.x0 < 30.f) { ++skippedStart; continue; }
					o.k = K_HAZARD;
				}
				else if (obj->m_objectType == GameObjectType::Solid)  o.k = K_SOLID;
				else continue;
				G.objs.push_back(o);
			}
		}
		std::sort(G.objs.begin(), G.objs.end(), [](Obj const& a, Obj const& b) { return a.r.x0 < b.r.x0; });
		std::sort(G.unsup.begin(), G.unsup.end());

		G.debug = CCDrawNode::create();
		m_objectLayer->addChild(G.debug, 1000);
		G.pathNode = CCDrawNode::create();
		m_objectLayer->addChild(G.pathNode, 1001);
		G.drawVer = 0; G.drawnVer = -1; G.pathVer = -1; G.lastPathIdx = -1;

		G.hud = CCLabelBMFont::create("AutoBot", "bigFont.fnt");
		G.hud->setScale(0.35f);
		G.hud->setOpacity(190);
		G.hud->setAnchorPoint({0.f, 1.f});
		auto win = CCDirector::get()->getWinSize();
		G.hud->setPosition({6.f, win.height - 6.f});
		this->addChild(G.hud, 1000);
		G.lastHud.clear();

		log::info("[AutoBot] v0.8.0 ready: {} objects, {} unsupported, {} start hazards ignored. Planning at <=240 steps/s.",
			G.objs.size(), G.unsup.size(), skippedStart);
	}

	void onQuit() {
		G.debug = nullptr;
		G.pathNode = nullptr;
		G.hud = nullptr;
		G.plan = Plan();
		G.prevX = -1.f;
		PlayLayer::onQuit();
	}

	void resetLevel() {
		PlayLayer::resetLevel();
		for (auto& o : G.objs) o.used = false;
		for (auto& ex : G.exps) ex.fired = false;
		G.holding = false;
		G.pressedLast = false;
		G.prevX = -1.f;
		G.havePred = false;
		G.plan.valid = false;
		G.diverged = false;
		G.divLogged = 0;
		G.sinceDiv = 1000000;
		G.err = 0.f;
		G.cool = 0;
		G.noReplan = 0;
		G.sinceTick = 0.f;
		G.sincePress = 1000000;
		G.qHist = 0;
		G.latWait = 0;
		++G.drawVer;
	}

	void destroyPlayer(PlayerObject* player, GameObject* object) {
		bool wasDead = m_player1 ? m_player1->m_isDead : true;
		PlayLayer::destroyPlayer(player, object);
		// Only count it if the player really died (GD calls this for harmless start-spike touches too).
		if (player == m_player1 && !wasDead && m_player1->m_isDead && G.active && C.enabled) {
			onDeath(object);
		}
	}

	void onDeath(GameObject* killer) {
		auto pos = m_player1->getPosition();

		if (!nearKnownObject(pos.x, pos.y)) {
			++G.phantom;
			if (G.phantom <= 3 || G.phantom % 50 == 0) {
				log::info("[AutoBot] ignored death at x={} y={} killerId={}: no level object near the player (anti-cheat or another mod). {} so far",
					(double)pos.x, (double)pos.y, killer ? killer->m_objectID : -1, G.phantom);
			}
			return;
		}

		++G.deaths;
		bool simSawDoom = G.plan.valid && !G.plan.full;
		char const* cls;
		if (simSawDoom)                    cls = "PLANNER DEAD END (sim saw no survivable path)";
		else if (G.sinceDiv < 120)         cls = "MODEL ERROR (sim diverged shortly before death)";
		else                               cls = "UNSEEN (plan was clean: timing, hitbox or unmodelled object)";

		int kid = killer ? killer->m_objectID : -1;
		Rc kr{0.f, 0.f, 0.f, 0.f};
		int knownKind = -1;   // -1 = sim does not know this object; 0 solid, 1 hazard, 2 orb, 3 pad, 4 portal
		if (killer) {
			kr = toRc(killer->getObjectRect());
			for (auto it = lowerBoundX(G.objs, kr.x0 - 40.f); it != G.objs.end() && it->r.x0 < kr.x1 + 40.f; ++it) {
				if (it->id == kid && std::abs(it->r.x0 - kr.x0) < 1.f && std::abs(it->r.y0 - kr.y0) < 1.f) {
					knownKind = (int)it->k;
					break;
				}
			}
		}

		float vLast = G.ringN > 0 ? G.ring[(G.ringN - 1) % 48].vN : 0.f;
		bool lastGr = G.ringN > 0 ? G.ring[(G.ringN - 1) % 48].gr : true;

		auto now = Clock::now();
		bool dump = std::chrono::duration<double>(now - G.lastDeathLogT).count() > 3.0;
		log::info("[AutoBot] death #{} x={} y={} killerId={} speed={} class={} replans={} maxErr={} lastSearch={}ms",
			G.deaths, (double)pos.x, (double)pos.y, kid,
			(double)G.speed, cls, G.replans, (double)G.maxErr, G.lastMs);
		log::info("[AutoBot]   killer rect=[{}, {}, {}, {}] simKnowsIt={} wallId={} wallDist={} planSteps={} planFull={} rung={}",
			(double)kr.x0, (double)kr.y0, (double)kr.x1, (double)kr.y1, knownKind,
			G.aheadId, G.aheadDist, G.plan.acts.size(), G.plan.full, G.rung);
		log::info("[AutoBot]   last click window=[{}..{}] bounded={} picked={} | ticksSinceClick={} lastClickX={} vyAtDeath={} onGround={} | latency={}",
			G.plan.winMin, G.plan.winMax, G.plan.winBounded, G.plan.winPick,
			G.sincePress, (double)G.pressX, (double)vLast, lastGr, G.lat);

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

		// 1) Was the click that killed us one that an experiment forced? Then that timing failed: try the next.
		Exp* resp = nullptr;
		for (auto& ex : G.exps) {
			if (ex.dead || !ex.fired) continue;
			if (G.sincePress < 280 && std::abs(G.pressX - ex.xt) < 8.f) { resp = &ex; break; }
		}
		if (resp) {
			++resp->n;
			if (expExhausted(*resp)) {
				resp->dead = true;
				log::info("[AutoBot] experiment at x={} gave up after {} tries (planner alone from here)",
					(double)resp->base, resp->n);
			} else {
				expTarget(*resp);
				log::info("[AutoBot] experiment at x={}: that timing died, next attempt clicks at x={} (try {})",
					(double)resp->base, (double)resp->xt, resp->n + 1);
			}
			return;
		}

		// 2) New culprit. Died in the air soon after a click: move that click. Otherwise a click is missing.
		if (G.exps.size() >= MAX_EXPS) return;
		Exp ex;
		ex.F = std::max(1.f, G.speed / 60.f);
		if (!lastGr && G.sincePress < 400) {
			ex.shift = true;
			ex.base = G.pressX;
			ex.air = G.pressAir;
			ex.dir = (vLast < -1.f) ? 1 : -1;   // falling = clicked too early -> later; rising/unknown -> earlier
		} else {
			ex.shift = false;
			ex.base = lastGr ? pos.x - 5.f : G.lastGroundX;
		}
		for (auto const& o : G.exps) {
			if (o.shift == ex.shift && std::abs(o.base - ex.base) < 30.f) {
				log::info("[AutoBot] already experimented near x={} (dead={}): not repeating", (double)ex.base, o.dead);
				return;
			}
		}
		expTarget(ex);
		G.exps.push_back(ex);
		if (ex.shift) {
			log::info("[AutoBot] experiment: click at x={} killed us in the air. Next attempt clicks {} ({} frame) at x={}",
				(double)ex.base, ex.dir < 0 ? "earlier" : "later", 1, (double)ex.xt);
		} else {
			log::info("[AutoBot] experiment: never clicked before dying near x={}. Next attempt clicks at x={}, then earlier each try",
				(double)pos.x, (double)ex.xt);
		}
	}

	void postUpdate(float dt) {
		PlayLayer::postUpdate(dt);
		auto t0 = Clock::now();
		++G.perf.frames;
		updateHud(t0);
		drawDebug(t0);
		perfTick(t0);
		double us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
		G.perf.uiUsSum += us;
		G.perf.uiUsMax = std::max(G.perf.uiUsMax, us);
	}

	void updateHud(Clock::time_point now) {
		if (!G.hud) return;
		if (std::chrono::duration<double>(now - G.lastHudT).count() < 0.1) return;
		G.lastHudT = now;

		int activeExp = 0;
		for (auto const& e : G.exps) if (!e.dead) ++activeExp;

		char buf[256];
		std::snprintf(buf, sizeof(buf), "AutoBot %s | %s%s | clk %d dth %d | plan %d%s | err %.2f/%.2f | %.1fms | lat %d rung %d exp %d | %.0fus/step",
			G.active ? "ON" : "OFF", G.modeStr, (G.active && !isCube(m_player1)) ? " (cube only)" : "",
			G.clicks, G.deaths,
			(int)(G.plan.valid ? G.plan.acts.size() - std::min(G.plan.idx, G.plan.acts.size()) : 0),
			(G.plan.valid && !G.plan.full) ? " DEADEND" : "",
			(double)G.err, (double)G.maxErr, G.lastMs, G.lat, G.rung, activeExp, G.stepAvgUs);
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

	void drawDebug(Clock::time_point now) {
		auto d = G.debug;
		auto pn = G.pathNode;
		if (!d || !pn) return;
		if (!G.active || !C.debugDraw) {
			if (G.drawnVer != -2) { d->clear(); pn->clear(); G.drawnVer = -2; G.lastPathIdx = -1; G.pathVer = -1; }
			return;
		}

		float px = m_player1->getPositionX();

		// Static objects: rebuilt only after a new plan, at most ~7x per second, nearby only.
		if (G.drawnVer != G.drawVer &&
		    std::chrono::duration<double>(now - G.lastDrawT).count() >= 0.15) {
			G.lastDrawT = now;
			G.drawnVer = G.drawVer;
			d->clear();

			auto drawRect = [&](Rc const& r, ccColor4F col) {
				CCPoint v[4] = {{r.x0, r.y0}, {r.x1, r.y0}, {r.x1, r.y1}, {r.x0, r.y1}};
				d->drawPolygon(v, 4, {0, 0, 0, 0}, 1.f, col);
			};
			int drawn = 0;
			for (auto const& o : G.dbgEnv.loc) {
				if (o.r.x1 < px - 20.f || o.r.x0 > px + 360.f) continue;
				if (++drawn > 80) break;
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
		}

		// Remaining planned path, starting at the player: refreshed up to 60x per second.
		int idx = (int)G.plan.idx;
		if ((idx != G.lastPathIdx || G.pathVer != G.drawVer) &&
		    std::chrono::duration<double>(now - G.lastPathT).count() >= (1.0 / 60.0)) {
			G.lastPathT = now;
			G.lastPathIdx = idx;
			G.pathVer = G.drawVer;
			pn->clear();
			if (G.plan.valid && G.plan.idx < G.plan.px.size()) {
				size_t i0 = G.plan.idx;
				size_t n = G.plan.px.size();
				CCPoint prev = {px, m_player1->getPositionY()};
				for (size_t i = i0; i < n; ++i) {
					bool click = G.plan.acts[i] != 0;
					if (!click && ((i - i0) % 12) != 11 && i + 1 != n) continue;
					CCPoint cur = {G.plan.px[i], G.plan.py[i]};
					pn->drawSegment(prev, cur, 1.2f, click ? ccColor4F{1.f, 1.f, 1.f, 1.f} : ccColor4F{0.2f, 1.f, 0.4f, 1.f});
					prev = cur;
				}
			}
		}
	}
};
