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
// AutoBot v0.9.0 - cube only. Predictive, multi-rate, uncertainty-aware.
//
//  Rates (kept strictly separate):
//   * ENGINE TICK  - one processCommands() call. The only moment the game can register an input
//                    and the only moment the real state changes. One bot decision per tick, at
//                    whatever tick rate the game runs (60/240/480/...). The sim step h follows the
//                    real tick delta (no 240/s cap, no artificial sub-steps).
//   * PLAN RATE    - a plan is a schedule of future ticks. It is (re)built only when needed, with a
//                    compute budget that grows with the danger level.
//   * VALIDATION   - when danger >= WARNING the committed plan is re-simulated from the REAL state
//                    every tick, so a drifting/invalid plan is dropped immediately.
//   * RENDER FRAME - HUD + overlay only.
//
//  Input: Requested -> Sent -> (game processes) -> Confirmed/Failed by telemetry.
//  Hitboxes are never shrunk to find a path. If no path exists the bot says so (DEADEND/BEST-EFFORT).
// ===========================================================================

constexpr float BASE_SPEED = 311.58f;
constexpr float MIN_STEP_H = 1e-4f;            // preserve the actual engine delta; no artificial tick quantisation
constexpr float MAX_STEP_H = 3.f;                 // dt > 0.05 s is rejected by botStep below
constexpr float DIV_TOL = 0.8f;
constexpr float DIV_COOLDOWN_FRAMES = 6.f;
constexpr float WALL_SLACK = 90.f;
constexpr float SETTLE_FRAMES = 15.f;
constexpr float LAND_GRACE_FRAMES = 12.f;        // plan must survive this long AFTER landing
constexpr float LEAD_OPEN_FRAMES = 3.f;
constexpr size_t MAX_EXPS = 24;
constexpr int LAT_MAX = 3;
constexpr double MS_DEADEND = 1.0;

constexpr float NOM_G = 0.864f;
constexpr float NOM_JUMP = 11.18f;
constexpr float NOM_POS = 0.9f;
namespace phys {
	constexpr float HALF = 15.f;
	constexpr float FALL_MAX = 15.f;
	constexpr float SPEEDS[5] = {251.16f, 311.58f, 387.42f, 468.0f, 576.0f};
	constexpr float PSPEEDS[5] = {0.7f, 0.9f, 1.1f, 1.3f, 1.6f};
}
struct Phys { float g = NOM_G, jump = NOM_JUMP, posScale = NOM_POS; };   // calibrated at runtime

enum Danger : int { D_SAFE, D_CAUTION, D_WARNING, D_CRITICAL, D_EMERGENCY };
static char const* dangerName(int d) {
	static char const* n[] = {"SAFE", "CAUTION", "WARNING", "CRITICAL", "EMERGENCY"};
	return n[std::clamp(d, 0, 4)];
}
static double dangerBudgetMs(int d) {
	static double t[] = {2.0, 3.0, 3.5, 5.0, 6.0};
	return t[std::clamp(d, 0, 4)];
}

enum JState : uint8_t { J_IDLE, J_SENT, J_CONFIRMED, J_FAILED };
static char const* jName(int s) {
	static char const* n[] = {"idle", "sent", "confirmed", "FAILED"};
	return n[std::clamp(s, 0, 3)];
}

enum Fail : int {
	F_TOO_EARLY, F_TOO_LATE, F_NOT_REGISTERED, F_JUMP_COLLISION, F_LOW_HEIGHT, F_UNSAFE_LANDING,
	F_PHYSICS, F_GEOMETRY, F_UNSUPPORTED, F_DEAD_END, F_UNEXPECTED
};
static char const* failName(int f) {
	static char const* n[] = {
		"jump too early", "jump too late / missing", "jump failed to register", "jump caused collision",
		"insufficient jump height", "unsafe landing", "incorrect physics prediction",
		"incorrect collision geometry", "unsupported level mechanic", "planner dead end", "unexpected game behavior"};
	return n[std::clamp(f, 0, 10)];
}

enum Reason : int {
	R_NONE, R_RUN, R_WAIT, R_CLICK, R_EXPERIMENT, R_DEADEND, R_BEST_EFFORT, R_JUMPFAIL, R_INVALID
};

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
// Settings cache (same setting keys as v0.8.0)
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
	bool bonk = false;     // touched a ceiling / underside this step
	uint8_t q = 0;         // last decided actions, bit0 = newest (for click latency)
	uint64_t mask = 0;
};

struct Env {
	std::vector<Obj> loc;
	float maxW = 30.f;
	float floorY = 90.f;
	float ceilY = 390.f;
	float margin = 2.f;
	float hzHalf = 15.f;   // player box vs hazards. Never shrunk.
	float h = 0.25f;
	float limitX = 1e9f;
	bool useOrbs = true;
	int lat = 0;           // click latency in ticks (0..LAT_MAX)
	int settle = 60;       // steps allowed to land after the horizon
	Phys ph;
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
	s.bonk = false;
	float h = e.h;

	s.x += s.speed * (h / 60.f);
	float py = s.y;
	bool canJump = s.ground;
	s.ground = false;
	if (canJump && eff) s.vy = e.ph.jump;
	s.vy = std::max(s.vy - e.ph.g * h, -FALL_MAX);
	s.y += s.vy * h * e.ph.posScale;

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
				s.y = r.y0 - HALF; s.vy = 0.f; s.bonk = true;
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
		s.bonk = true;
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
// Exact search (all action sequences, deduplicated, branch order: wait first)
// ---------------------------------------------------------------------------
using Clock = std::chrono::steady_clock;

struct Searcher {
	Env const* e = nullptr;
	int H = 0;
	long budget = 0;
	long nodes = 0;
	double msLimit = 3.5;
	bool timeUp = false;
	Clock::time_point t0;
	DeadSet dead;
	std::vector<uint8_t> path, bestPath, cur, tmp;   // reusable buffers
	std::vector<Sim> pre;
	int bestT = -1;

	uint64_t keyOf(Sim const& s, int t) const {
		uint64_t h = 1469598103934665603ull;
		auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ull; };
		mix((uint32_t)t);
		// X is not always implied by t: different portal histories can produce different
		// horizontal positions even when the current speed and vertical state match.
		mix((uint32_t)(int32_t)std::lround(s.x * 2.f));
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
			if (t >= H && (s.ground || t >= H + e->settle)) return true;
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

struct WinPt { float x = 0.f, y = 0.f; };

struct Plan {
	std::vector<uint8_t> acts;
	std::vector<float> px, py;
	size_t idx = 0;
	bool valid = false;
	bool full = false;
	bool clipped = false;
	bool timedOut = false;
	bool doom = false;      // best-effort only: the sim found NO survivable path
	long nodes = 0;
	int winMin = -1, winMax = -1, winPick = -1;   // first click window, in ticks from plan start
	bool winBounded = false;
	int tolTicks = -1;      // timing tolerance of the first click (min distance to either window edge)
	int clicks = 0;
	float x0 = 0.f;
	WinPt wMin, wMax, wPick;
};

// The sim sees no survivable path. Instead of walking blindly into it, pick the single click
// time that survives the longest. Conservative hitboxes are kept: this is flagged as lethal.
static bool bestEffort(Sim const& s0, Env const& e, int H, std::vector<uint8_t>& out) {
	int maxT = H + e.settle;
	auto depth = [&](int c) -> int {
		Sim s = s0;
		for (int t = 0; t < maxT; ++t) {
			if (!stepFrame(s, t == c, e)) return t;
		}
		return maxT;
	};
	int best = depth(-1);
	static std::vector<int> bestC;
	bestC.clear();
	auto t0 = Clock::now();
	int cMax = std::max(8, (int)(40.f / e.h));
	for (int c = 0; c < cMax; c += 2) {
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
// safe window. Every click (up to 3 per plan) is searched backwards to find the earliest safe tick,
// so the real window [earliest, latest] is known, then moved inside it (bias shifts the choice).
static void makePlan(Sim const& s0, Env const& e, int H, long budget, double msLimit, int bias, Plan& out, bool allowBestEffort) {
	static Searcher S;
	S.e = &e; S.H = H; S.budget = budget; S.nodes = 0; S.timeUp = false;
	S.msLimit = msLimit;
	S.t0 = Clock::now();
	S.dead.clear(); S.path.clear(); S.bestT = -1; S.bestPath.clear();

	out.winMin = out.winMax = out.winPick = -1;
	out.winBounded = false;
	out.doom = false;
	out.tolTicks = -1;
	out.wMin = out.wMax = out.wPick = WinPt();
	out.x0 = s0.x;

	bool ok = S.dfs(s0, 0);
	std::vector<uint8_t>& cur = S.cur;
	cur = ok ? S.path : S.bestPath;

	if (!ok && !S.timeUp && allowBestEffort) {
		if (bestEffort(s0, e, H, S.tmp)) { cur = S.tmp; out.doom = true; }
	}

	if (ok && !S.timeUp) {
		size_t from = 0;
		int leadOpen = std::max(1, (int)(LEAD_OPEN_FRAMES / e.h));
		for (int iter = 0; iter < 3 && !S.timeUp; ++iter) {
			int p0 = -1;
			for (size_t i = from; i < cur.size(); ++i) {
				uint8_t prev = (i == 0) ? (uint8_t)(s0.q & 1u) : cur[i - 1];
				if (cur[i] == 1 && prev == 0) { p0 = (int)i; break; }
			}
			if (p0 < 0) break;

			S.pre.clear();
			Sim s = s0;
			for (int i = 0; i < p0; ++i) { S.pre.push_back(s); stepFrame(s, cur[i] == 1, e); }

			int back = std::max(1, (int)(10.f / e.h));
			int lo = std::max((int)from, p0 - back);
			int cmin = p0;
			bool bounded = false;
			for (int c = p0 - 1; c >= lo; --c) {
				Sim const& st = S.pre[(size_t)c];
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
			else               target = p0 - leadOpen - off;
			target = std::clamp(target, cmin, p0);

			if (iter == 0) {
				out.winMin = cmin; out.winMax = p0; out.winPick = target; out.winBounded = bounded;
				out.tolTicks = std::min(target - cmin, p0 - target);
				Sim const& a = (cmin < p0) ? S.pre[(size_t)cmin] : s;
				out.wMin = {a.x, a.y * a.sgn};
				out.wMax = {s.x, s.y * s.sgn};
				Sim const& b = (target < p0) ? S.pre[(size_t)target] : s;
				out.wPick = {b.x, b.y * b.sgn};
			}

			if (target != p0) {
				Sim q = S.pre[(size_t)target];
				if (stepFrame(q, true, e)) {
					S.path.assign(cur.begin(), cur.begin() + target);
					S.path.push_back(1);
					if (S.dfs(q, target + 1)) cur = S.path;
				}
			}
			from = (size_t)target + 1;
		}
	}

	out.acts = cur;
	out.px.clear(); out.py.clear();
	out.px.reserve(out.acts.size()); out.py.reserve(out.acts.size());
	Sim s = s0;
	for (auto a : out.acts) {
		if (!stepFrame(s, a == 1, e)) break;
		out.px.push_back(s.x);
		out.py.push_back(s.y * s.sgn);
	}
	out.acts.resize(out.px.size());
	out.clicks = 0;
	for (size_t i = 0; i < out.acts.size(); ++i) {
		uint8_t prev = i == 0 ? (uint8_t)(s0.q & 1u) : out.acts[i - 1];
		if (out.acts[i] == 1 && prev == 0) ++out.clicks;
	}
	out.idx = 0;
	out.valid = true;
	out.full = ok;
	out.clipped = ok && (int)out.acts.size() < H;
	out.timedOut = S.timeUp;
	out.nodes = S.nodes;
}

// Re-simulate the remaining part of the committed plan from the REAL current state.
static bool planStillSafe(Sim s, Env const& e, Plan const& p) {
	for (size_t i = p.idx; i < p.acts.size(); ++i) {
		if (!stepFrame(s, p.acts[i] == 1, e)) return false;
	}
	return true;
}

// Counterfactual probe: "what if I never click?" -> time to collision / leaving platform / landing / ceiling.
struct Probe { int ttc = -1, fall = -1, land = -1, ceil = -1; };
static void probeNoPress(Sim s, Env const& e, int Hs, Probe& pr) {
	pr = Probe();
	bool wasG = s.ground;
	for (int t = 0; t < Hs; ++t) {
		if (!stepFrame(s, false, e)) { pr.ttc = t; break; }
		if (wasG && !s.ground && pr.fall < 0) pr.fall = t;
		if (!wasG && s.ground && pr.land < 0) pr.land = t;
		if (s.bonk && pr.ceil < 0) pr.ceil = t;
		wasG = s.ground;
	}
}

// Risk score of a plan. A plan is replayed open-loop under plausible variations of the things the
// bot is unsure about (latency +-1 tick, hazards inflated). Open-loop is pessimistic (the real bot
// replans every tick), so this is a ranking signal, never a reason to accept a lethal plan.
struct Score { int ok = 0, total = 0; long value = 0; };
static Score scorePlan(Sim const& s0, Env& e, Plan const& p) {
	Score r;
	int dl[5] = {0, 1, -1, 0, 0};
	float dm[5] = {0.f, 0.f, 0.f, 1.5f, 3.f};
	float m0 = e.margin;
	int l0 = e.lat;
	for (int k = 0; k < 5; ++k) {
		int l = l0 + dl[k];
		if (l < 0 || l > LAT_MAX) continue;
		e.lat = l;
		e.margin = m0 + dm[k];
		Sim s = s0;
		s.held = ((s0.q >> l) & 1u) != 0u;
		bool alive = true;
		for (size_t i = 0; i < p.acts.size() && alive; ++i) alive = stepFrame(s, p.acts[i] == 1, e);
		++r.total;
		if (alive) ++r.ok;
	}
	e.lat = l0; e.margin = m0;
	long tol = p.tolTicks < 0 ? 12 : std::min(p.tolTicks, 12);
	r.value = (long)r.ok * 3000 + tol * 60 - (long)p.clicks * 350;
	return r;
}

static int ticksToClick(Plan const& p) {
	for (size_t i = p.idx; i < p.acts.size(); ++i) {
		uint8_t prev = i == 0 ? (uint8_t)0 : p.acts[i - 1];
		if (p.acts[i] == 1 && prev == 0) return (int)(i - p.idx);
	}
	return -1;
}

// SAFE .. EMERGENCY from the time left before the click that can still save us.
static int classifyDanger(Plan const& plan, float h, int lat) {
	if (!plan.valid || !plan.full || plan.doom) return D_EMERGENCY;
	int c = ticksToClick(plan);
	if (c < 0) return D_SAFE;
	int latest = c;
	bool inFirst = plan.winMax >= 0 && (int)plan.idx <= plan.winMax;
	if (inFirst) latest = std::max(0, plan.winMax - (int)plan.idx);
	float tickMs = h * (1000.f / 60.f);
	float ms = (float)latest * tickMs;
	float minReact = (float)(lat + 2) * tickMs;
	int lvl;
	if (ms > 500.f) lvl = D_CAUTION;
	else if (ms > 200.f) lvl = D_WARNING;
	else if (ms > minReact + tickMs) lvl = D_CRITICAL;
	else lvl = D_EMERGENCY;
	if (inFirst && plan.tolTicks >= 0 && plan.tolTicks <= 2 && ms < 300.f && lvl < D_CRITICAL) lvl = D_CRITICAL;
	return lvl;
}

// ===========================================================================
// Trial-and-error learning. The game itself is the judge: after a REAL death the bot blames a click
// (or a missing click) and tries a different click position on the next attempt, never repeating a
// timing that already failed. Positions repeat exactly because the physics is deterministic.
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
	bool solved = false;   // the player got well past this click alive: timing is preserved
	float xt = 0.f;        // x where the click is forced in the current attempt
	int cause = 0;         // Fail enum that created it
	int nTried = 0;
	float tried[32]{};     // click positions that already killed us
};

static float rawTarget(Exp const& e, int n) {
	if (e.shift) {
		int k = n / 2 + 1;
		int sgn = (n % 2 == 0) ? e.dir : -e.dir;
		return e.base + (float)(sgn * k) * e.F;
	}
	return e.base - (float)(n + 1) * e.F;
}
static bool expExhausted(Exp const& e) {
	return e.shift ? (e.n / 2 + 1 > 10) : (e.n >= 26);
}
// Next untried target position.
static void expTarget(Exp& e) {
	for (int guard = 0; guard < 64 && !expExhausted(e); ++guard) {
		float xt = rawTarget(e, e.n);
		bool seen = false;
		for (int i = 0; i < e.nTried; ++i) {
			if (std::abs(e.tried[i] - xt) < 0.5f * e.F) { seen = true; break; }
		}
		e.xt = xt;
		if (!seen) return;
		++e.n;
	}
}

// ===========================================================================
// Bot state
// ===========================================================================
struct Rec { float x, y, vN, predY, err; bool gr, act; };

struct DeathRec {
	int attempt = 0, cls = 0, killerId = -1, danger = 0;
	float x = 0.f, y = 0.f, vy = 0.f, lastClickX = 0.f, predClickX = 0.f, err = 0.f;
	bool grounded = false, registered = false;
};

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

// Input state controller: Requested -> Sent -> (game processes it) -> Confirmed / Failed by telemetry.
struct InCtl {
	JState st = J_IDLE;
	long tick = 0;
	long sentTick = -1, confirmTick = -1, requestTick = -1;
	float sentX = 0.f, sentY = 0.f, vBefore = 0.f, lastConfirmedX = 0.f;
	bool sentAir = false, sentClean = false;
	int confirms = 0, fails = 0, consecFails = 0;
};

struct Bot {
	std::vector<Obj> objs;
	std::vector<std::pair<float, int>> unsup;
	Plan plan, alt;
	Env env;               // planner working environment (also drawn)
	Phys ph;               // calibrated physics

	bool holding = false, pressedLast = false;
	int clicks = 0, deaths = 0, phantom = 0, replans = 0, attempt = 0;
	long ticks = 0;

	float prevX = -1.f, prevY = 0.f, prevVN = 0.f;
	bool prevAir = false;
	float planH = 0.25f, lastH = 0.25f;
	float speed = BASE_SPEED;

	bool havePred = false;
	float predX = 0.f, predY = 0.f;
	float err = 0.f, maxErr = 0.f, calibErr = 0.f;
	bool diverged = false;
	int divLogged = 0;
	int sinceDiv = 1000000;
	bool forceReplan = false;
	int cool = 0;
	int noReplan = 0;
	int rung = 0;
	float usedMargin = 2.f;
	int invalidations = 0, altWins = 0;

	// input
	InCtl in;
	int lat = 0;
	float latCnt[LAT_MAX + 1] = {0.f, 0.f, 0.f, 0.f};
	uint8_t qHist = 0;
	float lastPredClickX = 0.f;

	// danger / compute allocation
	int danger = D_SAFE;
	Probe probe;
	long probeTick = -1000;
	Score planScore;
	int nextObjId = -1;
	float nextObjDist = 0.f;
	float depthScale = 1.f;
	double loadMsAcc = 0.0, loadFrac = 0.0;
	Clock::time_point loadT0{};
	int reason = R_NONE, reasonA = 0, reasonB = 0;

	// experiments
	std::vector<Exp> exps;
	float lastGroundX = 0.f;
	bool pressAir = false;
	int expFires = 0;

	float floorY = 90.f, ceilY = 390.f;
	bool ceilLearned = false;

	Rec ring[48];
	int ringN = 0;
	CCPoint trail[96];
	int trailN = 0;
	DeathRec deathHist[16];
	int deathN = 0;
	int sincePress = 1000000;
	float pressX = 0.f, pressY = 0.f;

	double lastMs = 0.0, maxMs = 0.0;
	int aheadId = -1, aheadDist = 0;

	CCDrawNode* debug = nullptr;
	CCDrawNode* pathNode = nullptr;
	CCDrawNode* trailNode = nullptr;
	CCLabelBMFont* hud = nullptr;
	bool active = false;
	char const* modeStr = "cube";

	int drawVer = 0, drawnVer = -1, pathVer = -1, lastPathIdx = -1;
	Clock::time_point lastHudT{}, lastDrawT{}, lastPathT{}, lastTrailT{}, lastDeathLogT{}, lastDeadLogT{};
	std::string lastHud;
	Perf perf;
	double stepAvgUs = 0.0;
};
static Bot G;

static float latConf() {
	float tot = 0.f, best = 0.f;
	for (float c : G.latCnt) { tot += c; best = std::max(best, c); }
	return tot < 3.f ? 0.5f : best / tot;
}
// Extra hazard margin from what the bot does not know: shaky latency estimate, physics prediction error.
static float uncertainty() {
	float u = latConf() < 0.75f ? 1.f : 0.f;
	u += std::min(2.f, G.calibErr * 1.5f);
	if (G.in.consecFails > 0) u += 1.f;
	return std::min(u, 3.5f);
}

static void releaseBtn(GJBaseGameLayer* gl, bool force = false) {
	// On reset/death/quit, send an explicit release even if our local flag is already
	// false. This repairs a possible local/game input-state desynchronisation.
	if (!G.holding && !force) return;
	G.holding = false;
	gl->handleButton(false, 1, true);
}

// 'want' is the decided button state for this tick. A press records everything needed to later
// confirm from telemetry that the game really jumped.
static void setHold(GJBaseGameLayer* gl, bool want, PlayerObject* p, float vN, bool clean) {
	if (want == G.holding) return;
	G.holding = want;
	if (want) {
		++G.clicks;
		G.pressedLast = true;
		G.sincePress = 0;
		G.pressX = p->getPositionX();
		G.pressY = p->getPositionY();
		G.pressAir = !p->m_isOnGround;
		auto& in = G.in;
		in.st = J_SENT;
		in.sentTick = in.tick;
		in.requestTick = in.tick;
		in.sentX = G.pressX; in.sentY = G.pressY;
		in.vBefore = vN;
		in.sentAir = G.pressAir;
		in.sentClean = clean;
	}
	gl->handleButton(want, 1, true);
}

static void resetTracking() {
	G.prevX = -1.f; G.havePred = false; G.plan.valid = false;
	G.qHist = 0;
	G.in.st = J_IDLE;
}

// Called at the start of every tick: did the last press really produce a jump?
static void inputConfirm(GJBaseGameLayer* gl, PlayerObject* p, float vN, float h) {
	auto& in = G.in;
	if (in.st != J_SENT) return;
	long age = in.tick - in.sentTick;   // 1 = first tick after the tick that sent the press
	// A ground press is only confirmed when the player is actually airborne and
	// moving upward. This avoids counting a positive velocity while still grounded.
	// The upper bound also helps distinguish an ordinary jump from a nearby strong pad.
	bool jumped = in.sentAir
		? (vN > in.vBefore + 2.5f)
		: (!p->m_isOnGround && vN > 4.f && vN < G.ph.jump + 2.f);
	if (jumped) {
		in.st = J_CONFIRMED;
		in.confirmTick = in.tick;
		in.lastConfirmedX = in.sentX;
		in.consecFails = 0;
		++in.confirms;
		int meas = (int)age - 1;
		if (in.sentClean && meas >= 0 && meas <= LAT_MAX) {
			for (auto& c : G.latCnt) c *= 0.93f;
			G.latCnt[meas] += 1.f;
			int best = 0;
			for (int i = 1; i <= LAT_MAX; ++i) if (G.latCnt[i] > G.latCnt[best]) best = i;
			if (best != G.lat && G.latCnt[best] > G.latCnt[G.lat] * 1.25f + 0.4f) {
				G.lat = best;
				G.forceReplan = true;
				log::info("[AutoBot] click latency is now {} tick(s) (samples: {:.1f} / {:.1f} / {:.1f} / {:.1f}, confidence {:.0f}%)",
					G.lat, G.latCnt[0], G.latCnt[1], G.latCnt[2], G.latCnt[3], (double)(latConf() * 100.f));
			}
			if (meas == G.lat) {
				// first visible velocity after a clean ground jump = jump - one gravity application
				float est = vN + G.ph.g * h;
				G.ph.jump = std::clamp(G.ph.jump * 0.85f + est * 0.15f, NOM_JUMP * 0.95f, NOM_JUMP * 1.05f);
			}
		}
	} else if (age > (long)G.lat + 3) {
		in.st = J_FAILED;
		++in.fails; ++in.consecFails;
		G.forceReplan = true;
		G.havePred = false;       // the prediction assumed a jump that did not happen
		G.plan.valid = false;
		G.reason = R_JUMPFAIL;
		releaseBtn(gl);           // allow a fresh press edge
		static int failLogs = 0;
		if (failLogs < 12) {
			++failLogs;
			log::info("[AutoBot] JUMP NOT REGISTERED: press sent at x={} ({}) {} ticks ago, vN={} (before {}). Reassessing from the real state. fails={} consecutive={}",
				(double)in.sentX, in.sentAir ? "air" : "ground", age, (double)vN, (double)in.vBefore, in.fails, in.consecFails);
		}
	}
}

static bool objNear(float x, float y, float r) {
	Rc q{x - r, y - r, x + r, y + r};
	for (auto it = lowerBoundX(G.objs, x - r - 60.f); it != G.objs.end() && it->r.x0 < x + r; ++it) {
		if (hit(q, it->r)) return true;
	}
	return false;
}

static void perfTick(Clock::time_point now) {
	auto& P = G.perf;
	if (P.winStart.time_since_epoch().count() == 0) { P.winStart = now; return; }
	double el = std::chrono::duration<double>(now - P.winStart).count();
	if (el < 5.0) return;
	if (G.active && P.steps > 0 && P.frames > 0) {
		G.stepAvgUs = P.stepUsSum / (double)P.steps;
		log::info("[AutoBot] perf: {:.0f} render fps | {:.0f} decisions/s (1 per engine tick, h={}) | bot {:.1f}us avg {:.0f}us max | game step {:.1f}us avg {:.0f}us max | {} plans {:.2f}ms avg {:.2f}ms max | depth x{:.2f} | hud+overlay {:.1f}us avg {:.0f}us max",
			(double)P.frames / el, (double)P.steps / el, (double)G.lastH,
			G.stepAvgUs, P.stepUsMax,
			P.gameUsSum / (double)P.steps, P.gameUsMax,
			P.plans, P.plans ? P.planMsSum / (double)P.plans : 0.0, P.planMsMax,
			(double)G.depthScale,
			P.uiUsSum / (double)P.frames, P.uiUsMax);
	}
	P = Perf();
	P.winStart = now;
}

// Strict compute governor: if planning eats more than ~20% of wall time the search depth shrinks.
static void loadUpdate(double ms, Clock::time_point now) {
	if (G.loadT0.time_since_epoch().count() == 0) { G.loadT0 = now; return; }
	G.loadMsAcc += ms;
	double el = std::chrono::duration<double>(now - G.loadT0).count();
	if (el < 1.0) return;
	G.loadFrac = G.loadMsAcc / (el * 1000.0);
	G.depthScale = G.loadFrac > 0.20 ? std::max(0.5f, G.depthScale * 0.9f) : std::min(1.f, G.depthScale * 1.05f);
	G.loadMsAcc = 0.0;
	G.loadT0 = now;
}

static void buildEnv(Env& env, Sim& s0, float px, float h, int Hs, float limitX) {
	env.loc.clear();
	env.maxW = 30.f;
	env.floorY = G.floorY;
	env.ceilY = G.ceilLearned ? G.ceilY : G.floorY + 300.f;
	env.useOrbs = C.useOrbs;
	env.h = h;
	env.limitX = limitX;
	env.lat = G.lat;
	env.ph = G.ph;
	env.hzHalf = 15.f;
	env.settle = std::max(8, (int)std::ceil(SETTLE_FRAMES / h));
	float reachX = G.speed * ((float)Hs * h / 60.f) + 90.f + G.speed * ((float)env.settle * h / 60.f);
	s0.mask = 0;
	int slot = 0;
	for (auto it = lowerBoundX(G.objs, px - 200.f); it != G.objs.end() && it->r.x0 <= px + reachX; ++it) {
		if (it->r.x1 < px - 60.f) continue;
		Obj o = *it;
		if (o.k == K_ORB || o.k == K_PAD || o.k == K_PORTAL) {
			if (slot < 64) {
				o.ix = slot++;
				if (o.used) s0.mask |= 1ull << o.ix;
			} else {
				// The simulator's consumed-object mask is 64 bits. Never silently ignore
				// additional interactive objects: treat overflow as a conservative hazard.
				o.k = K_HAZARD;
				o.ix = -1;
			}
		}
		env.maxW = std::max(env.maxW, o.r.x1 - o.r.x0);
		env.loc.push_back(o);
	}
}

// ---------------------------------------------------------------------------
// One bot decision per ENGINE TICK (the only rate at which input can matter)
// ---------------------------------------------------------------------------
static void botStep(GJBaseGameLayer* gl, PlayLayer* pl, float dt) {
	if (++C.age >= 240) refreshCfg();
	auto p = pl->m_player1;
	if (!p) {
		G.active = false;
		releaseBtn(gl, true);
		resetTracking();
		return;
	}

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

	auto tickStart = Clock::now();
	++G.in.tick;
	++G.ticks;
	float hRaw = dt * 60.f;
	float h = std::clamp(hRaw, MIN_STEP_H, MAX_STEP_H);
	G.lastH = h;
	bool fresh = G.prevX < 0.f;

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
		float sp = (pos.x - G.prevX) / (hRaw / 60.f);
		for (float t : phys::SPEEDS) {
			if (std::abs(sp - t) < t * 0.06f) { G.speed = t; break; }
		}
	}

	// ---- did the last press really register? ----------------------------------
	G.reason = R_NONE;
	if (!fresh) inputConfirm(gl, p, vN, h);

	// ---- prediction audit + divergence classification ---------------------------
	if (G.sinceDiv < 1000000) ++G.sinceDiv;
	float stepErr = 0.f;
	if (G.havePred && !fresh) {
		float ex = pos.x - G.predX, ey = pos.y - G.predY;
		stepErr = std::max(std::abs(ex), std::abs(ey));
		G.err = ey;
		G.maxErr = std::max(G.maxErr, std::abs(ey));
		G.calibErr = G.calibErr * 0.98f + std::min(stepErr, 5.f) * 0.02f;
		if (stepErr > DIV_TOL) {
			G.sinceDiv = 0;
			G.diverged = true;
			if (G.cool == 0) { G.forceReplan = true; G.cool = std::max(2, (int)(DIV_COOLDOWN_FRAMES / h)); }
			if (G.divLogged < 5) {
				++G.divLogged;
				int nearId = -1;
				for (auto it = lowerBoundX(G.objs, pos.x - 100.f); it != G.objs.end() && it->r.x0 < pos.x + 60.f; ++it) {
					Rc pr{pos.x - 25.f, pos.y - 25.f, pos.x + 25.f, pos.y + 25.f};
					if (hit(pr, it->r)) { nearId = it->id; break; }
				}
				char const* cause;
				if (G.sincePress <= G.lat + 3)            cause = "input latency / jump registration";
				else if (std::abs(ex) > DIV_TOL)          cause = "horizontal speed / speed portal";
				else if (!p->m_isOnGround)                cause = "gravity / jump velocity / position scale";
				else                                      cause = "ground / landing / collision geometry";
				log::info("[AutoBot] DIVERGENCE #{} x={} y={} dx={} dy={} vRaw={} vN={} gnd={} sgn={} speed={} h={} nearObj={} likelyCause={} | calib g={:.4f} jump={:.3f} posScale={:.3f}",
					G.divLogged, (double)pos.x, (double)pos.y, (double)ex, (double)ey, (double)p->m_yVelocity,
					(double)vN, p->m_isOnGround, (double)sgn, (double)G.speed, (double)h, nearId, cause,
					(double)G.ph.g, (double)G.ph.jump, (double)G.ph.posScale);
			}
		}
	}
	G.havePred = false;

	// ---- passive physics calibration from clean free-fall -----------------------
	if (!fresh && !p->m_isOnGround && G.prevAir && G.sincePress > G.lat + 3 &&
	    std::abs(vN) < 14.f && std::abs(G.prevVN) < 14.f && !objNear(pos.x, pos.y, 45.f)) {
		float dv = G.prevVN - vN;
		if (dv > 0.02f) {
			float gEst = dv / h;
			G.ph.g = std::clamp(G.ph.g * 0.97f + gEst * 0.03f, NOM_G * 0.9f, NOM_G * 1.1f);
		}
		float dyN = (pos.y - G.prevY) * sgn;
		if (std::abs(vN) > 2.f) {
			float ps = dyN / (vN * h);
			if (ps > 0.5f && ps < 1.5f)
				G.ph.posScale = std::clamp(G.ph.posScale * 0.97f + ps * 0.03f, NOM_POS * 0.85f, NOM_POS * 1.15f);
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
	G.prevX = pos.x; G.prevY = pos.y; G.prevVN = vN; G.prevAir = !p->m_isOnGround;
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

	// ---- horizon / current state ------------------------------------------------
	float Hfr = (float)C.horizon + LAND_GRACE_FRAMES;
	if (G.danger >= D_CRITICAL) Hfr = std::min(Hfr, 34.f);   // short-horizon emergency planning
	else Hfr *= G.depthScale;                                 // adaptive depth under load
	int Hs = std::max(8, (int)std::ceil(Hfr / h));

	Sim s0;
	s0.x = pos.x; s0.sgn = sgn; s0.y = pos.y * sgn; s0.vy = vN;
	s0.speed = G.speed; s0.ground = p->m_isOnGround;
	s0.q = G.qHist;
	s0.held = ((G.qHist >> G.lat) & 1u) != 0u;

	size_t remaining = G.plan.valid ? (G.plan.acts.size() > G.plan.idx ? G.plan.acts.size() - G.plan.idx : 0) : 0;
	bool need = !G.plan.valid || G.forceReplan || fresh ||
	            (G.plan.full ? remaining <= (size_t)(Hs / 2) : remaining == 0);
	if (G.plan.valid && G.plan.clipped && !G.forceReplan && !fresh && remaining > 0) need = false;
	if (G.noReplan > 0 && G.plan.valid && !G.forceReplan && !fresh) need = false;

	bool envFresh = false, probeNow = false;
	double planMs = 0.0;

	// ---- emergency recovery: near danger the plan is re-validated from the REAL state every tick ----
	if (!need && G.plan.valid && G.plan.full && G.danger >= D_WARNING) {
		buildEnv(G.env, s0, pos.x, h, Hs, limitX);
		G.env.margin = G.usedMargin;
		envFresh = true;
		if (!planStillSafe(s0, G.env, G.plan)) {
			need = true;
			G.forceReplan = true;
			++G.invalidations;
			if (G.reason == R_NONE) G.reason = R_INVALID;
		}
	}

	// ---- (re)plan ----------------------------------------------------------------
	if (need) {
		bool wasFull = G.plan.valid && G.plan.full;
		bool keepable = wasFull && remaining > 0 && !G.forceReplan && !fresh;
		static Plan backup;
		if (keepable) backup = G.plan;

		buildEnv(G.env, s0, pos.x, h, Hs, limitX);
		envFresh = true;
		Env& env = G.env;

		double lim = (fresh || wasFull || !G.plan.valid) ? dangerBudgetMs(G.danger) : MS_DEADEND;
		auto t0 = Clock::now();

		// Margin ladder is CONSERVATIVE only: with prediction uncertainty it first looks for a path
		// with extra margin, then falls back to the configured margin. It never goes below it and
		// the player's hazard box is never shrunk.
		float unc = uncertainty();
		float rm[3];
		int nr = 0;
		if (unc > 0.05f) { rm[nr++] = C.margin + unc + 1.f; rm[nr++] = C.margin + unc; }
		rm[nr++] = C.margin;
		int used = 0;
		for (int ri = 0; ri < nr; ++ri) {
			env.margin = rm[ri];
			makePlan(s0, env, Hs, C.budget, ri == 0 ? lim : MS_DEADEND, 0, G.plan, ri == nr - 1);
			used = ri;
			if (G.plan.full) break;
		}
		G.rung = used;
		G.usedMargin = env.margin;

		// Compare futures: score the plan under latency/margin variations and try plans whose clicks
		// sit at other places inside their windows. A lethal plan can never win on score.
		if (G.plan.full && !G.plan.timedOut) {
			G.planScore = scorePlan(s0, env, G.plan);
			int w = G.plan.winMax - G.plan.winMin;
			auto elapsed = [&]() { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); };
			if (w >= 4 && elapsed() < lim * 0.6) {
				int biases[2] = {-(w / 4), (w / 4)};
				for (int bi = 0; bi < 2; ++bi) {
					if (biases[bi] == 0 || elapsed() > lim) continue;
					makePlan(s0, env, Hs, C.budget, MS_DEADEND, biases[bi], G.alt, false);
					if (!G.alt.full || G.alt.timedOut) continue;
					Score sc = scorePlan(s0, env, G.alt);
					if (sc.value > G.planScore.value) {
						G.planScore = sc;
						std::swap(G.plan, G.alt);
						++G.altWins;
					}
				}
			}
		} else {
			G.planScore = Score();
		}

		// The search ran out of time and found nothing: the old plan is still better than guessing.
		if (keepable && !G.plan.full && G.plan.timedOut) {
			G.plan = std::move(backup);
			G.noReplan = 12;
		} else if (!G.plan.full) {
			G.noReplan = 3;
		}

		planMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
		G.lastMs = planMs;
		G.maxMs = std::max(G.maxMs, G.lastMs);
		G.perf.planMsSum += G.lastMs;
		G.perf.planMsMax = std::max(G.perf.planMsMax, G.lastMs);
		++G.perf.plans;
		G.forceReplan = false;
		G.planH = h;
		++G.replans;
		++G.drawVer;
		probeNow = true;

		if (!G.plan.full) {
			auto now = Clock::now();
			if (std::chrono::duration<double>(now - G.lastDeadLogT).count() > 1.0) {
				G.lastDeadLogT = now;
				log::info("[AutoBot] NO SURVIVABLE PATH at x={} y={} vN={} gnd={} | conservative hitboxes, margin {} | sim survives {} steps, dies near x={} | bestEffort={} timedOut={} nodes={} wallId={} wallDist={}",
					(double)pos.x, (double)pos.y, (double)vN, p->m_isOnGround, (double)G.usedMargin, G.plan.acts.size(),
					(double)(G.plan.px.empty() ? pos.x : G.plan.px.back()),
					G.plan.doom, G.plan.timedOut, G.plan.nodes, G.aheadId, G.aheadDist);
			}
		}
	}

	// ---- counterfactual probe + danger level (cheap; refreshed every few ticks) -------------
	if (!envFresh && G.plan.valid && (G.in.tick - G.probeTick) >= 6) {
		buildEnv(G.env, s0, pos.x, h, Hs, limitX);
		G.env.margin = G.usedMargin;
		envFresh = true;
		probeNow = true;
	}
	if (envFresh && probeNow && G.plan.valid) {
		probeNoPress(s0, G.env, Hs, G.probe);
		G.probeTick = G.in.tick;
		G.nextObjId = -1;
		G.nextObjDist = 0.f;
		for (auto const& o : G.env.loc) {
			if ((o.k == K_HAZARD || o.k == K_SOLID) && o.r.x1 > pos.x) {
				G.nextObjId = o.id; G.nextObjDist = o.r.x0 - pos.x; break;
			}
		}
	}
	G.danger = classifyDanger(G.plan, h, G.lat);

	// ---- execute the committed plan --------------------------------------------
	bool act = false;
	float predNY = pos.y;
	int clickLeft = -1;
	bool fromPlan = false;
	if (G.plan.valid && G.plan.idx < G.plan.acts.size()) {
		clickLeft = ticksToClick(G.plan);
		size_t i = G.plan.idx++;
		act = G.plan.acts[i] == 1;
		G.predX = G.plan.px[i];
		G.predY = G.plan.py[i];
		predNY = G.predY;
		G.havePred = true;
		fromPlan = true;
		if (act && !G.holding) G.lastPredClickX = i > 0 ? G.plan.px[i - 1] : G.plan.x0;
	}
	if (G.reason == R_NONE) {
		if (G.plan.valid && G.plan.doom) G.reason = R_BEST_EFFORT;
		else if (G.plan.valid && !G.plan.full) G.reason = R_DEADEND;
		else if (act && !G.holding) G.reason = R_CLICK;
		else if (clickLeft > 0) G.reason = R_WAIT;
		else G.reason = R_RUN;
	}
	G.reasonA = clickLeft;
	G.reasonB = G.plan.tolTicks;

	// ---- trial-and-error experiments (timing learned from real deaths) ---------
	if (C.learn && !G.exps.empty()) {
		bool overridden = false;
		for (auto& ex : G.exps) {
			if (ex.dead) continue;
			if (ex.fired) {
				if (!ex.solved && pos.x > ex.xt + 400.f) {
					ex.solved = true;
					log::info("[AutoBot] experiment at x={} solved the section (click x={}): timing is preserved for later attempts",
						(double)ex.base, (double)ex.xt);
				}
				continue;
			}
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
				G.lastPredClickX = pos.x;
			}
		}
		if (overridden) { G.havePred = false; G.reason = R_EXPERIMENT; }
	}
	(void)fromPlan;

	bool wasHolding = G.holding;
	bool clean = act && !wasHolding && p->m_isOnGround && std::abs(vN) < 0.5f && !padOrOrbNear();
	setHold(gl, act, p, vN, clean);
	G.qHist = (uint8_t)(((G.qHist << 1) | (act ? 1u : 0u)) & 0xFu);

	Rec& r = G.ring[G.ringN % 48];
	r = {pos.x, pos.y, vN, predNY, stepErr, p->m_isOnGround, act};
	++G.ringN;
	if ((G.ticks % 3) == 0) { G.trail[G.trailN % 96] = CCPoint{pos.x, pos.y}; ++G.trailN; }

	loadUpdate(planMs, Clock::now());
	(void)tickStart;
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

static char const* reasonText(char* buf, size_t n) {
	switch (G.reason) {
		case R_RUN:        std::snprintf(buf, n, "run: no jump needed in the planned horizon"); break;
		case R_WAIT:       std::snprintf(buf, n, "wait: next click in %d ticks (window tolerance %d)", G.reasonA, G.reasonB); break;
		case R_CLICK:      std::snprintf(buf, n, "click now: inside safe window, tolerance %d ticks", G.reasonB); break;
		case R_EXPERIMENT: std::snprintf(buf, n, "learning experiment overrides the plan"); break;
		case R_DEADEND:    std::snprintf(buf, n, "no survivable path found (conservative hitboxes)"); break;
		case R_BEST_EFFORT:std::snprintf(buf, n, "EMERGENCY best-effort click: sim says lethal anyway"); break;
		case R_JUMPFAIL:   std::snprintf(buf, n, "reassess: last jump did not register"); break;
		case R_INVALID:    std::snprintf(buf, n, "plan invalid from real state: replanned"); break;
		default:           std::snprintf(buf, n, "-"); break;
	}
	return buf;
}

class $modify(BotPlay, PlayLayer) {
	void setupHasCompleted() {
		PlayLayer::setupHasCompleted();
		refreshCfg();

		G.objs.clear();
		G.unsup.clear();
		G.exps.clear();
		G.plan = Plan();
		G.alt = Plan();
		G.floorY = 90.f; G.ceilY = 390.f; G.ceilLearned = false;
		G.sinceDiv = 1000000; G.divLogged = 0;
		G.qHist = 0; G.noReplan = 0; G.rung = 0;
		G.in = InCtl();
		G.danger = D_SAFE; G.probeTick = -1000; G.attempt = 0;
		G.depthScale = 1.f; G.loadMsAcc = 0.0;
		G.trailN = 0;
		G.perf = Perf();
		G.env.loc.reserve(512);

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
		G.trailNode = CCDrawNode::create();
		m_objectLayer->addChild(G.trailNode, 1002);
		G.drawVer = 0; G.drawnVer = -1; G.pathVer = -1; G.lastPathIdx = -1;

		G.hud = CCLabelBMFont::create("AutoBot", "bigFont.fnt");
		G.hud->setScale(0.3f);
		G.hud->setOpacity(190);
		G.hud->setAnchorPoint({0.f, 1.f});
		auto win = CCDirector::get()->getWinSize();
		G.hud->setPosition({6.f, win.height - 6.f});
		this->addChild(G.hud, 1000);
		G.lastHud.clear();

		log::info("[AutoBot] v0.9.0 ready: {} objects, {} unsupported, {} start hazards ignored. One decision per engine tick; sim step follows the real tick delta.",
			G.objs.size(), G.unsup.size(), skippedStart);
	}

	void onQuit() {
		releaseBtn(this, true);
		G.debug = nullptr;
		G.pathNode = nullptr;
		G.trailNode = nullptr;
		G.hud = nullptr;
		G.plan = Plan();
		G.prevX = -1.f;
		PlayLayer::onQuit();
	}

	void resetLevel() {
		// Never carry a held button across a respawn; force a game-side release too.
		releaseBtn(this, true);
		PlayLayer::resetLevel();
		for (auto& o : G.objs) o.used = false;
		for (auto& ex : G.exps) ex.fired = false;
		++G.attempt;
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
		G.sincePress = 1000000;
		G.qHist = 0;
		G.in.st = J_IDLE;
		G.in.sentTick = -1;
		G.danger = D_SAFE;
		G.probeTick = -1000;
		G.trailN = 0;
		++G.drawVer;
	}

	void destroyPlayer(PlayerObject* player, GameObject* object) {
		bool wasDead = m_player1 ? m_player1->m_isDead : true;
		PlayLayer::destroyPlayer(player, object);
		// Only count it if the player really died (GD calls this for harmless start-spike touches too).
		if (player == m_player1 && !wasDead && m_player1->m_isDead && G.active && C.enabled) {
			releaseBtn(this, true);
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
		float pressAgeFrames = (float)G.sincePress * G.lastH;
		long sinceSend = G.in.tick - G.in.sentTick;
		bool unregistered = G.in.sentTick >= 0 && sinceSend <= 12 && G.in.st != J_CONFIRMED;

		int cls;
		char const* why;
		if (G.aheadId >= 0 && G.aheadDist < 90)      { cls = F_UNSUPPORTED;     why = "an unsupported object/mechanic is right ahead"; }
		else if (killer && knownKind < 0)            { cls = F_GEOMETRY;        why = "the killer object is not in the sim's object table"; }
		else if (unregistered)                       { cls = F_NOT_REGISTERED;  why = "the last press never produced a jump in telemetry"; }
		else if (G.sinceDiv < 120)                   { cls = F_PHYSICS;         why = "the sim diverged from the game shortly before death"; }
		else if (simSawDoom)                         { cls = F_DEAD_END;        why = "no survivable path existed with conservative hitboxes"; }
		else if (!lastGr && pressAgeFrames < 100.f) {
			if (knownKind == (int)K_SOLID)           { cls = F_JUMP_COLLISION;  why = "the jump carried the cube into a block"; }
			else if (vLast < -1.f)                   { cls = F_TOO_EARLY;       why = "descending onto a hazard: the click came too early"; }
			else                                     { cls = F_LOW_HEIGHT;      why = "still rising when the hazard was hit: click too late / not enough height"; }
		}
		else if (!lastGr)                            { cls = F_UNSAFE_LANDING;  why = "airborne long after the last click: unsafe landing"; }
		else                                         { cls = F_TOO_LATE;        why = "on the ground at death with no timely click"; }

		// ---- death report --------------------------------------------------------------
		int nearCnt = 0;
		for (auto it = lowerBoundX(G.objs, pos.x - 80.f); it != G.objs.end() && it->r.x0 < pos.x + 80.f; ++it) ++nearCnt;
		float lastErr = G.ringN > 0 ? G.ring[(G.ringN - 1) % 48].err : 0.f;

		auto now = Clock::now();
		bool dump = std::chrono::duration<double>(now - G.lastDeathLogT).count() > 3.0;
		log::info("[AutoBot] ===== DEATH #{} (attempt {}) at x={} y={} =====", G.deaths, G.attempt, (double)pos.x, (double)pos.y);
		log::info("[AutoBot]   cause: {} - {}", failName(cls), why);
		log::info("[AutoBot]   killerId={} rect=[{}, {}, {}, {}] simKnowsIt={} objectsWithin80u={} | wallId={} wallDist={}",
			kid, (double)kr.x0, (double)kr.y0, (double)kr.x1, (double)kr.y1, knownKind, nearCnt, G.aheadId, G.aheadDist);
		log::info("[AutoBot]   state: vy={} onGround={} speed={} danger={} | plan steps={} full={} bestEffort={} margin={} tol={} window=[{}..{}] bounded={}",
			(double)vLast, lastGr, (double)G.speed, dangerName(G.danger), G.plan.acts.size(), G.plan.full, G.plan.doom,
			(double)G.usedMargin, G.plan.tolTicks, G.plan.winMin, G.plan.winMax, G.plan.winBounded);
		log::info("[AutoBot]   input: lastRequestedClickX={} predictedClickX={} (diff {}) | jumpState={} lastConfirmedClickX={} | ticksSinceSend={} latency={} (conf {:.0f}%) | confirms={} fails={}",
			(double)G.pressX, (double)G.lastPredClickX, (double)(G.pressX - G.lastPredClickX), jName(G.in.st),
			(double)G.in.lastConfirmedX, sinceSend, G.lat, (double)(latConf() * 100.f), G.in.confirms, G.in.fails);
		log::info("[AutoBot]   prediction: lastStepErr={} maxErr={} avgErr={:.3f} sinceDivergence={} ticks | calib g={:.4f} jump={:.3f} posScale={:.3f} | replans={} lastSearch={}ms",
			(double)lastErr, (double)G.maxErr, (double)G.calibErr, G.sinceDiv, (double)G.ph.g, (double)G.ph.jump,
			(double)G.ph.posScale, G.replans, G.lastMs);

		if (dump) {
			G.lastDeathLogT = now;
			int n = std::min(G.ringN, 12);
			for (int i = n; i >= 1; --i) {
				Rec const& rr = G.ring[(G.ringN - i) % 48];
				log::info("[AutoBot]   t-{} x={} y={} vN={} gnd={} act={} predY={} err={}",
					i, (double)rr.x, (double)rr.y, (double)rr.vN, rr.gr, rr.act, (double)rr.predY, (double)rr.err);
			}
		}

		DeathRec& dr = G.deathHist[G.deathN % 16];
		dr = DeathRec();
		dr.attempt = G.attempt; dr.cls = cls; dr.killerId = kid; dr.danger = G.danger;
		dr.x = pos.x; dr.y = pos.y; dr.vy = vLast; dr.lastClickX = G.pressX; dr.predClickX = G.lastPredClickX;
		dr.err = lastErr; dr.grounded = lastGr; dr.registered = G.in.st == J_CONFIRMED;
		++G.deathN;

		if (!C.learn) return;

		// A registration failure is not a timing problem: moving clicks around would hide it.
		if (cls == F_NOT_REGISTERED) {
			++G.in.consecFails;
			log::info("[AutoBot] learning: not shifting any click timing for this death (registration problem, latency estimate stays under review)");
			return;
		}

		// 1) Was the click that killed us one that an experiment forced? Then that timing failed: try the next.
		Exp* resp = nullptr;
		for (auto& ex : G.exps) {
			if (ex.dead || !ex.fired) continue;
			if (G.sincePress < 280 && std::abs(G.pressX - ex.xt) < 8.f) { resp = &ex; break; }
		}
		if (resp) {
			if (resp->nTried < 32) resp->tried[resp->nTried++] = resp->xt;
			resp->solved = false;
			++resp->n;
			expTarget(*resp);
			if (expExhausted(*resp)) {
				resp->dead = true;
				log::info("[AutoBot] experiment at x={} gave up after {} tries (planner alone from here)",
					(double)resp->base, resp->n);
			} else {
				log::info("[AutoBot] experiment at x={}: click x={} died ({}), next untried click x={} (try {})",
					(double)resp->base, (double)resp->tried[resp->nTried > 0 ? resp->nTried - 1 : 0],
					failName(cls), (double)resp->xt, resp->n + 1);
			}
			return;
		}

		// 2) New culprit. Died in the air soon after a click: move that click. Otherwise a click is missing.
		if (G.exps.size() >= MAX_EXPS) return;
		Exp ex;
		ex.F = std::max(1.f, G.speed / 60.f);
		ex.cause = cls;
		if (!lastGr && G.sincePress < 400) {
			ex.shift = true;
			ex.base = G.pressX;
			ex.air = G.pressAir;
			if (cls == F_TOO_EARLY)                                  ex.dir = 1;
			else if (cls == F_LOW_HEIGHT || cls == F_TOO_LATE)       ex.dir = -1;
			else                                                     ex.dir = (vLast < -1.f) ? 1 : -1;
		} else {
			ex.shift = false;
			ex.base = lastGr ? pos.x - 5.f : G.lastGroundX;
		}
		for (auto const& o : G.exps) {
			if (o.shift == ex.shift && std::abs(o.base - ex.base) < 30.f) {
				log::info("[AutoBot] already experimented near x={} (dead={} solved={}): not repeating", (double)ex.base, o.dead, o.solved);
				return;
			}
		}
		expTarget(ex);
		G.exps.push_back(ex);
		if (ex.shift) {
			log::info("[AutoBot] experiment ({}): click at x={} killed us in the air. Next attempt clicks {} at x={}",
				failName(cls), (double)ex.base, ex.dir < 0 ? "earlier" : "later", (double)ex.xt);
		} else {
			log::info("[AutoBot] experiment ({}): no click before dying near x={}. Next attempt clicks at x={}, then earlier each try",
				failName(cls), (double)pos.x, (double)ex.xt);
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

		int activeExp = 0, solvedExp = 0;
		for (auto const& e : G.exps) { if (!e.dead) ++activeExp; if (e.solved) ++solvedExp; }

		float tickMs = G.lastH * (1000.f / 60.f);
		char why[128];
		reasonText(why, sizeof(why));

		char buf[900];
		int n = std::snprintf(buf, sizeof(buf),
			"AutoBot %s | %s%s | %.0f ticks/s h=%.3f | %s | clk %d dth %d\n"
			"plan %d%s tol %d win[%d..%d] pick %d | score %d/%d | jump %s (c%d f%d) lat %d %.0f%% | margin %.1f\n"
			"err %.2f/%.2f avg %.2f | g %.3f jv %.2f ps %.3f | search %.1fms (%ld nodes) depth x%.2f | exp %d/%d solved\n"
			"ttc %.0fms fall %.0fms ceil %.0fms | next obj %d in %.0fu | %s",
			G.active ? "ON" : "OFF", G.modeStr, (G.active && !isCube(m_player1)) ? " (cube only)" : "",
			(double)(60.f / G.lastH), (double)G.lastH, dangerName(G.danger), G.clicks, G.deaths,
			(int)(G.plan.valid ? G.plan.acts.size() - std::min(G.plan.idx, G.plan.acts.size()) : 0),
			(G.plan.valid && G.plan.doom) ? " BEST-EFFORT" : ((G.plan.valid && !G.plan.full) ? " DEADEND" : ""),
			G.plan.tolTicks, G.plan.winMin, G.plan.winMax, G.plan.winPick,
			G.planScore.ok, G.planScore.total, jName(G.in.st), G.in.confirms, G.in.fails, G.lat, (double)(latConf() * 100.f),
			(double)G.usedMargin,
			(double)G.err, (double)G.maxErr, (double)G.calibErr, (double)G.ph.g, (double)G.ph.jump, (double)G.ph.posScale,
			G.lastMs, (long)G.plan.nodes, (double)G.depthScale, activeExp, solvedExp,
			G.probe.ttc >= 0 ? (double)(G.probe.ttc * tickMs) : -1.0,
			G.probe.fall >= 0 ? (double)(G.probe.fall * tickMs) : -1.0,
			G.probe.ceil >= 0 ? (double)(G.probe.ceil * tickMs) : -1.0,
			G.nextObjId, (double)G.nextObjDist, why);
		std::string s = buf;
		(void)n;
		if (G.aheadId >= 0 && G.aheadDist < 600) {
			char b2[64];
			std::snprintf(b2, sizeof(b2), "\nunsupported id %d in %d u", G.aheadId, G.aheadDist);
			s += b2;
		}
		if (s == G.lastHud) return;
		G.lastHud = s;
		G.hud->setString(s.c_str());
	}

	void drawDebug(Clock::time_point now) {
		auto d = G.debug;
		auto pn = G.pathNode;
		auto tn = G.trailNode;
		if (!d || !pn || !tn) return;
		if (!G.active || !C.debugDraw) {
			if (G.drawnVer != -2) { d->clear(); pn->clear(); tn->clear(); G.drawnVer = -2; G.lastPathIdx = -1; G.pathVer = -1; }
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
			for (auto const& o : G.env.loc) {
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

		// Planned path: green = confirmed safe (robust), yellow = survivable but fragile/uncertain,
		// red = lethal (no survivable path), white-cyan = the committed next ~0.25 s, white = click.
		int idx = (int)G.plan.idx;
		if ((idx != G.lastPathIdx || G.pathVer != G.drawVer) &&
		    std::chrono::duration<double>(now - G.lastPathT).count() >= (1.0 / 60.0)) {
			G.lastPathT = now;
			G.lastPathIdx = idx;
			G.pathVer = G.drawVer;
			pn->clear();
			if (G.plan.valid && G.plan.idx < G.plan.px.size()) {
				bool robust = G.planScore.total > 0 && G.planScore.ok == G.planScore.total &&
				              (G.plan.tolTicks < 0 || G.plan.tolTicks >= 3);
				ccColor4F base = !G.plan.full ? ccColor4F{1.f, 0.15f, 0.15f, 1.f}
				               : robust ? ccColor4F{0.2f, 1.f, 0.4f, 1.f}
				                        : ccColor4F{1.f, 0.85f, 0.15f, 1.f};
				ccColor4F commit{0.7f, 1.f, 1.f, 1.f};
				size_t i0 = G.plan.idx;
				size_t n = G.plan.px.size();
				size_t commitN = (size_t)std::max(1.f, 15.f / G.planH);
				size_t stride = (size_t)std::max(1.f, 3.f / G.planH);
				CCPoint prev = {px, m_player1->getPositionY()};
				for (size_t i = i0; i < n; ++i) {
					bool click = G.plan.acts[i] != 0;
					if (!click && ((i - i0) % stride) != stride - 1 && i + 1 != n) continue;
					CCPoint cur = {G.plan.px[i], G.plan.py[i]};
					ccColor4F col = click ? ccColor4F{1.f, 1.f, 1.f, 1.f} : ((i - i0) < commitN ? commit : base);
					pn->drawSegment(prev, cur, 1.2f, col);
					prev = cur;
				}
				if (G.plan.winMin >= 0) {
					pn->drawDot({G.plan.wMin.x, G.plan.wMin.y}, 3.f, {0.3f, 0.6f, 1.f, 1.f});    // earliest safe click
					pn->drawDot({G.plan.wMax.x, G.plan.wMax.y}, 3.f, {1.f, 0.5f, 0.1f, 1.f});    // latest safe click
					pn->drawDot({G.plan.wPick.x, G.plan.wPick.y}, 2.f, {1.f, 1.f, 1.f, 1.f});    // chosen click
				}
			}
		}

		// Actual trajectory (what the game really did), refreshed at 30 Hz.
		if (std::chrono::duration<double>(now - G.lastTrailT).count() >= (1.0 / 30.0)) {
			G.lastTrailT = now;
			tn->clear();
			int cnt = std::min(G.trailN, 96);
			for (int i = cnt - 1; i >= 1; --i) {
				CCPoint a = G.trail[(G.trailN - i - 1) % 96];
				CCPoint b = G.trail[(G.trailN - i) % 96];
				tn->drawSegment(a, b, 1.f, {0.2f, 0.8f, 1.f, 0.9f});
			}
		}
	}
};
