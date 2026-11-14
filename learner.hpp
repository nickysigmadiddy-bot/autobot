#pragma once
// ===========================================================================
// AutoBot adaptive jump-timing learner (header-only, pure C++20, no Geode dependency).
//
// Timing is measured as d = (obstacle front x) - (click x), in world units. Bigger d = EARLIER click.
//   * died TOO LATE  at d  -> every d' <= d is late too   -> lateBound  = max(lateBound, d)
//   * died TOO EARLY at d  -> every d' >= d is early too  -> earlyBound = min(earlyBound, d)
//   * a verified clear at d -> d is preserved; any bound contradicting it is dropped
// The safe window is the open interval (lateBound, earlyBound). The search:
//   only lateBound known  -> go earlier with a doubling step (coarse exploration)
//   only earlyBound known -> go later with a doubling step   (coarse exploration)
//   both known            -> bisect                          (fine refinement)
//   direction unknown     -> alternate earlier / later around the first timing
//   bounds collapse      -> contradictory evidence (noise / physics mismatch): reset and re-explore
// A timing is never tried twice (tolerance = one engine tick of travel), so there is no failure loop.
// Registration / physics failures never move the timing the first time; only a repeat at the same
// timing is escalated to an "unknown direction" experiment so the learner cannot get stuck either.
// Everything is deterministic (no RNG). One source of truth: Site::target.
// ===========================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace learn {

constexpr float NONE = -1.f;
constexpr float D_MAX = 420.f;       // never click more than this far before an obstacle

enum Kind : int { K_EARLY, K_LATE, K_AMBIG, K_REG, K_PHYS };       // what the death classifier says
enum Res : int { R_CLEAR = 0, R_LATE = 1, R_EARLY = 2, R_UNKNOWN = 3, R_LANDING = 4 };
enum Mode : int { M_NONE, M_COARSE, M_REFINE, M_REEXPLORE, M_LOCKED, M_PRESERVE };

inline char const* resName(int r) {
	static char const* n[] = {"CLEARED", "TOO LATE", "TOO EARLY", "UNKNOWN DIRECTION", "BAD LANDING / NEXT HAZARD"};
	return n[std::clamp(r, 0, 4)];
}
inline char const* modeName(int m) {
	static char const* n[] = {"NO DATA", "COARSE EXPLORATION", "FINE REFINEMENT", "RE-EXPLORING", "LOCKED (reliable)", "PRESERVING CLEAR"};
	return n[std::clamp(m, 0, 5)];
}
inline uint64_t fnv(std::string const& s, uint64_t h = 1469598103934665603ull) {
	for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
	return h;
}
inline uint64_t mix(uint64_t h, uint64_t v) { h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); return h; }

struct Cluster { int key; float x0, x1; };
struct Try { float d; int res; float prog; };
struct Clear { float d; int ok = 1, bad = 0; };

struct Site {
	int key = 0;
	float x0 = 0.f, x1 = 0.f;
	std::vector<Try> tried;
	std::vector<Clear> clears;
	float late = NONE, early = NONE, step = 0.f, target = 0.f, seed = NONE;
	int attempts = 0, collapses = 0, pend = 0, identical = 0, escalations = 0, mode = M_NONE;
	float pendD = NONE, lastDeathX = -1e9f;
	bool noJump = false, haveTarget = false, exhausted = false, improved = false;
	int lastRes = -1;
	float lastD = NONE, lastProg = NONE, prevProg = NONE;
	std::string cause;
};

struct Action { bool suppress = false, force = false; };
struct Report { std::string text; int key = -1; };

struct Learner {
	std::vector<Cluster> cl;
	std::map<int, Site> sites;
	uint64_t fp = 0;
	float levelLen = 1.f;
	float lastF = 5.19f;
	bool dirty = false;

	struct Att { float d = NONE; bool clicked = false, cleared = false, forced = false; };
	std::map<int, Att> att;
	size_t passIdx = 0;

	// ---- setup ------------------------------------------------------------------------
	// hz = (x0, x1) of every hazard. Hazards closer than 130 units form one obstacle site.
	void build(std::vector<std::pair<float, float>> hz, float len, uint64_t seed) {
		cl.clear(); sites.clear(); att.clear(); passIdx = 0; dirty = false;
		levelLen = std::max(1.f, len);
		std::sort(hz.begin(), hz.end());
		for (auto const& h : hz) {
			if (cl.empty() || h.first - cl.back().x1 > 130.f) cl.push_back({(int)std::lround(h.first), h.first, h.second});
			else cl.back().x1 = std::max(cl.back().x1, h.second);
		}
		fp = seed;
		for (auto const& c : cl) {
			fp = mix(fp, (uint64_t)(uint32_t)c.key);
			fp = mix(fp, (uint64_t)(uint32_t)std::lround(c.x1));
			Site s; s.key = c.key; s.x0 = c.x0; s.x1 = c.x1;
			sites[c.key] = s;
		}
	}
	void beginAttempt() { att.clear(); passIdx = 0; }

	Cluster const* ahead(float x) const {
		auto it = std::partition_point(cl.begin(), cl.end(), [&](Cluster const& c) { return c.x1 <= x; });
		return it == cl.end() ? nullptr : &*it;
	}
	int indexOf(int key) const {
		for (size_t i = 0; i < cl.size(); ++i) if (cl[i].key == key) return (int)i;
		return -1;
	}

	// ---- helpers ----------------------------------------------------------------------
	static bool anyFail(Site const& s) {
		for (auto const& t : s.tried) if (t.res != R_CLEAR) return true;
		return false;
	}
	static float nearestUntried(Site const& s, float c, float tol, float lo, float hi, int prefer) {
		auto ok = [&](float v) {
			if (v < 0.f || v > D_MAX) return false;
			if (lo >= 0.f && v <= lo) return false;
			if (hi >= 0.f && v >= hi) return false;
			for (auto const& t : s.tried) if (std::fabs(t.d - v) < tol) return false;
			return true;
		};
		int N = (int)(D_MAX / tol) + 2;
		for (int i = 0; i <= N; ++i) {
			float off = (float)i * tol;
			if (prefer > 0) { if (ok(c + off)) return c + off; }
			else if (prefer < 0) { if (ok(c - off)) return c - off; }
			else {
				if (ok(c + off)) return c + off;
				if (i && ok(c - off)) return c - off;
			}
		}
		return NONE;
	}

	// Choose the next timing for a site. This is the only place a target is decided.
	void retarget(Site& s, float F, float u) {
		s.exhausted = false;
		// 1) A timing that already cleared is preserved until evidence (2+ later failures) says otherwise.
		Clear const* best = nullptr;
		float mid = (s.late >= 0.f && s.early >= 0.f) ? (s.late + s.early) * 0.5f : NONE;
		for (auto const& c : s.clears) {
			if (c.bad >= 2 && c.bad >= c.ok) continue;
			if (!best) { best = &c; continue; }
			int a = c.ok - c.bad, b = best->ok - best->bad;
			if (a > b || (a == b && mid >= 0.f && std::fabs(c.d - mid) < std::fabs(best->d - mid))) best = &c;
		}
		if (best) {
			s.target = best->d;
			s.mode = (best->ok >= 2 && best->bad == 0) ? M_LOCKED : M_PRESERVE;
			return;
		}
		// 2) Search.
		float tol = std::max(0.9f * u, 0.5f);
		for (int iter = 0; iter < 3; ++iter) {
			bool lo = s.late >= 0.f, hi = s.early >= 0.f;
			float c = 0.f;
			int prefer = 0;
			bool collapsed = false;
			if (lo && hi) {
				if (s.early - s.late < 1.5f * u) collapsed = true;
				else { c = (s.late + s.early) * 0.5f; s.mode = M_REFINE; }
			} else if (lo) {
				if (s.step <= 0.f) s.step = 2.f * F;
				c = s.late + s.step; s.step = std::min(s.step * 2.f, 240.f);
				prefer = 1; s.mode = s.collapses ? M_REEXPLORE : M_COARSE;
			} else if (hi) {
				if (s.step <= 0.f) s.step = 2.f * F;
				c = s.early - s.step; s.step = std::min(s.step * 2.f, 240.f);
				prefer = -1; s.mode = s.collapses ? M_REEXPLORE : M_COARSE;
			} else {
				// direction unknown: test earlier AND later, alternating, growing around the first timing
				float st = s.step > 0.f ? s.step : 2.f * F;
				int n = (int)s.tried.size();
				int k = n / 2 + 1;
				float sgn = (n % 2 == 0) ? 1.f : -1.f;
				float base = s.seed >= 0.f ? s.seed : 40.f;
				c = base + sgn * (float)k * st;
				s.mode = s.collapses ? M_REEXPLORE : M_COARSE;
			}
			if (!collapsed) {
				c = std::clamp(c, 0.f, D_MAX);
				float v = nearestUntried(s, c, tol, lo ? s.late : NONE, hi ? s.early : NONE, prefer);
				if (v >= 0.f) { s.target = v; return; }
				if (!(lo && hi)) break;        // nothing left to try on this side
				collapsed = true;
			}
			// contradictory bounds: the evidence disagrees with itself -> forget bounds, explore wider
			++s.collapses;
			s.late = s.early = NONE;
			s.step = 2.f * F * (float)(1 << std::min(s.collapses, 3));
			s.mode = M_REEXPLORE;
		}
		s.exhausted = true;
	}

	void record(Site& s, float d, int res, float prog, float x, float F, float u, std::string const& cause) {
		++s.attempts;
		if (s.tried.empty()) s.seed = d;
		s.tried.push_back({d, res, prog});
		if (s.tried.size() > 64) s.tried.erase(s.tried.begin());
		s.prevProg = s.lastProg; s.lastProg = prog; s.lastD = d; s.lastRes = res; s.cause = cause;
		s.improved = res == R_CLEAR || (s.prevProg >= 0.f && prog > s.prevProg + 0.002f);
		if (res == R_LATE) s.late = std::max(s.late, d);
		if (res == R_EARLY) s.early = s.early < 0.f ? d : std::min(s.early, d);
		if (res != R_CLEAR) {
			for (auto& c : s.clears) if (std::fabs(c.d - d) < std::max(0.9f * u, 0.5f)) ++c.bad;
			// repeated, nearly identical deaths: the timing change was too small to matter -> widen the step
			if (std::fabs(x - s.lastDeathX) < 1.5f) ++s.identical; else s.identical = 0;
			s.lastDeathX = x;
			if (s.identical >= 2) { s.step = std::min(240.f, std::max(s.step, 2.f * F) * 2.f); ++s.escalations; s.identical = 0; }
		}
		s.haveTarget = true;
		retarget(s, F, u);
		dirty = true;
	}

	void recordClear(Site& s, float d, float prog, float F, float u) {
		float tol = std::max(0.9f * u, 0.5f);
		bool found = false;
		for (auto& c : s.clears) if (std::fabs(c.d - d) < tol) { ++c.ok; found = true; break; }
		if (!found) s.clears.push_back({d, 1, 0});
		++s.attempts;
		s.tried.push_back({d, R_CLEAR, prog});
		if (s.tried.size() > 64) s.tried.erase(s.tried.begin());
		s.prevProg = s.lastProg; s.lastProg = prog; s.lastD = d; s.lastRes = R_CLEAR; s.improved = true; s.cause = "cleared and landed safely";
		if (s.late >= d) s.late = NONE;                       // evidence beats an inconsistent bound
		if (s.early >= 0.f && s.early <= d) s.early = NONE;
		s.noJump = false;
		if (s.haveTarget || anyFail(s)) { s.haveTarget = true; retarget(s, F, u); }
		dirty = true;
	}

	// ---- runtime hooks ------------------------------------------------------------------
	// Called once per engine tick, AFTER the planner decided. plannerPress = planner wants a NEW click now.
	Action tick(float x, bool grounded, bool plannerPress, bool holding, float F, float u) {
		Action a;
		lastF = F;
		Cluster const* c = ahead(x);
		if (!c) return a;
		Site& s = sites[c->key];
		if (s.noJump) return a;
		if (!s.haveTarget && anyFail(s)) { s.haveTarget = true; retarget(s, F, u); }
		if (!s.haveTarget || s.exhausted) return a;
		Att& at = att[c->key];
		if (at.forced) return a;
		float xt = c->x0 - s.target;
		if (x < xt) {
			if (grounded && plannerPress && c->x0 - x < D_MAX + 10.f) a.suppress = true;   // hold the planner's click back
			return a;
		}
		if (grounded && !holding && x < c->x0 - 8.f) { a.force = true; at.forced = true; }
		else if (x >= c->x0 - 8.f) at.forced = true;            // too late to apply this attempt
		return a;
	}

	// Every real ground click. The first click before an obstacle defines that attempt's timing.
	void noteClick(float pressX, bool air) {
		if (air) return;
		for (auto const& c : cl) {
			if (c.x0 < pressX - 5.f) continue;
			float d = c.x0 - pressX;
			if (d > D_MAX) return;
			Att& at = att[c.key];
			if (!at.clicked) { at.clicked = true; at.d = std::max(0.f, d); }
			return;
		}
	}

	// An obstacle counts as cleared only when the player is 100u past it, alive and landed.
	void noteProgress(float x, bool grounded) {
		while (passIdx < cl.size() && cl[passIdx].x1 + 100.f < x) {
			if (!grounded) break;
			Cluster const& c = cl[passIdx];
			Att& at = att[c.key];
			Site& s = sites[c.key];
			if (at.clicked && !at.cleared) {
				recordClear(s, at.d, std::min(1.f, x / levelLen), lastF, lastF);
				at.cleared = true;
			} else if (!at.clicked && s.tried.empty() && s.clears.empty() && !s.noJump) {
				s.noJump = true; dirty = true;                  // the planner passes this site without jumping
			}
			++passIdx;
		}
	}

	std::string describe(Site const& s, int idx) const {
		char b[512];
		float fr = std::max(1.f, lastF);
		if (s.exhausted) {
			std::snprintf(b, sizeof(b), "obstacle #%02d: last %.1f frames (%s); every untried timing in range is used up -> planner decides",
				idx, s.lastD / fr, resName(s.lastRes));
		} else {
			float adj = (s.target - s.lastD) / fr;
			std::snprintf(b, sizeof(b), "obstacle #%02d: %.1f frames before it was %s; next %.1f frames (%.1f %s) | mode %s | window (%.1f, %.1f) frames | attempts %d",
				idx, s.lastD / fr, resName(s.lastRes), s.target / fr, std::fabs(adj), adj >= 0.f ? "EARLIER" : "LATER",
				modeName(s.mode), s.late >= 0.f ? s.late / fr : -1.f, s.early >= 0.f ? s.early / fr : -1.f, s.attempts);
		}
		return b;
	}

	// ---- death attribution --------------------------------------------------------------
	Report onDeath(float x, bool grounded, float vy, int kind, float F, float u) {
		(void)vy;
		lastF = F;
		Report rep;
		int si = -1;
		for (int i = (int)cl.size() - 1; i >= 0; --i) if (cl[i].x0 - 45.f <= x) { si = i; break; }
		if (si < 0 || x > cl[si].x1 + 400.f) { rep.text = "death is not near a tracked obstacle: nothing learned"; return rep; }
		Cluster const& S = cl[si];
		Site& s = sites[S.key];
		Att& at = att[S.key];
		float prog = std::min(1.f, x / levelLen);
		rep.key = S.key;
		int idx = si + 1;

		if (at.clicked) {
			float d = at.d;
			if (kind == K_REG || kind == K_PHYS) {
				// Not evidence about timing. Only a repeat at the same timing is escalated.
				if (s.pendD >= 0.f && std::fabs(s.pendD - d) < std::max(0.9f * u, 0.5f)) ++s.pend; else { s.pendD = d; s.pend = 1; }
				if (s.pend >= 2) {
					s.pend = 0;
					record(s, d, R_UNKNOWN, prog, x, F, u, "repeated non-timing failure at the same timing");
					rep.text = "repeated registration/physics failure at the same timing -> exploring a different one. " + describe(s, idx);
				} else {
					rep.text = "registration/physics failure: timing NOT changed (first occurrence at this timing)";
				}
				return rep;
			}
			int res;
			char const* why;
			if (kind == K_EARLY)                 { res = R_EARLY;   why = "descending onto the hazard / landed in front of it"; }
			else if (kind == K_LATE && grounded) { res = R_EARLY;   why = "jump already finished before the obstacle"; }
			else if (kind == K_LATE)             { res = R_LATE;    why = "still rising / not enough height at the front"; }
			else                                 { res = R_UNKNOWN; why = "collision with ambiguous direction (block / landing / dead end)"; }
			s.pend = 0;
			record(s, d, res, prog, x, F, u, why);
			rep.text = describe(s, idx);
			return rep;
		}

		// No click recorded for this obstacle.
		if (kind == K_REG || kind == K_PHYS) { rep.text = "no click for this obstacle and not a timing failure: nothing learned"; return rep; }
		if (!grounded) {
			// Airborne without a click here: the jump over an earlier obstacle ended on this one (planning failure).
			for (int i = si - 1; i >= 0; --i) {
				auto it = att.find(cl[i].key);
				if (it == att.end() || !it->second.clicked) continue;
				if (S.x0 - cl[i].x1 > 500.f) break;
				Site& p = sites[cl[i].key];
				record(p, it->second.d, R_LANDING, prog, x, F, u, "the jump cleared this obstacle but ended on the next hazard");
				rep.text = "planning failure: " + describe(p, i + 1);
				return rep;
			}
			rep.text = "airborne death with no jump recorded for any nearby obstacle: nothing learned";
			return rep;
		}
		// On the ground, never jumped: any click at d <= (distance to the front now) would also be too late.
		s.noJump = false;
		float d = std::max(0.f, S.x0 - x);
		record(s, d, R_LATE, prog, x, F, u, "no click before the obstacle (target too late)");
		rep.text = "missing click: " + describe(s, idx);
		return rep;
	}

	// ---- HUD ---------------------------------------------------------------------------
	std::string hud(float x) const {
		auto it = std::partition_point(cl.begin(), cl.end(), [&](Cluster const& c) { return c.x1 <= x; });
		if (it == cl.end()) return "REINFORCEMENT LEARNING\nno obstacles ahead";
		int idx = (int)(it - cl.begin()) + 1;
		Site const& s = sites.at(it->key);
		float fr = std::max(1.f, lastF);
		char b[640];
		int n = std::snprintf(b, sizeof(b), "REINFORCEMENT LEARNING\nObstacle: #%02d  Attempts: %d  Learning mode: %s\n", idx, s.attempts, modeName(s.mode));
		if (s.noJump) {
			std::snprintf(b + n, sizeof(b) - n, "No jump needed here");
		} else if (s.exhausted) {
			std::snprintf(b + n, sizeof(b) - n, "Search exhausted: planner timing in use");
		} else if (!s.haveTarget) {
			std::snprintf(b + n, sizeof(b) - n, "Planner timing (no failures recorded yet)");
		} else {
			float best = NONE;
			for (auto const& c : s.clears) if (c.ok - c.bad > 0) { best = c.d; break; }
			float adj = (s.target - s.lastD) / fr;
			int m = n;
			if (s.lastD >= 0.f)
				m += std::snprintf(b + m, sizeof(b) - m, "Previous: %.1f f (%s)  Current: %.1f f  Adjustment: %.1f f %s\n",
					s.lastD / fr, resName(s.lastRes), s.target / fr, std::fabs(adj), adj >= 0.f ? "EARLIER" : "LATER");
			char bb[24];
			if (best >= 0.f) std::snprintf(bb, sizeof(bb), "%.1f f", best / fr); else std::snprintf(bb, sizeof(bb), "none yet");
			std::snprintf(b + m, sizeof(b) - m, "Failure: %s | Progress %.1f%% %s | Best known: %s",
				s.cause.empty() ? "-" : s.cause.c_str(), std::max(0.f, s.lastProg) * 100.f,
				s.improved ? "(IMPROVED)" : "(no improvement)", bb);
		}
		return b;
	}

	// ---- persistence (integers only, so save(load(x)) == x) --------------------------------
	std::string save() const {
		std::ostringstream o;
		char hx[32];
		std::snprintf(hx, sizeof(hx), "%016llx", (unsigned long long)fp);
		o << "AUTOBOT-LEARN 1 " << hx << "\n";
		for (auto const& kv : sites) {
			Site const& s = kv.second;
			if (s.tried.empty() && s.clears.empty() && !s.noJump && s.attempts == 0) continue;
			o << "S " << s.key << ' ' << std::lround(s.late * 100.f) << ' ' << std::lround(s.early * 100.f) << ' '
			  << std::lround(s.step * 100.f) << ' ' << s.attempts << ' ' << s.collapses << ' ' << (s.noJump ? 1 : 0) << ' '
			  << s.tried.size() << ' ' << s.clears.size() << "\n";
			for (auto const& t : s.tried) o << "T " << std::lround(t.d * 100.f) << ' ' << t.res << ' ' << std::lround(t.prog * 1000.f) << "\n";
			for (auto const& c : s.clears) o << "C " << std::lround(c.d * 100.f) << ' ' << c.ok << ' ' << c.bad << "\n";
		}
		o << "END\n";
		return o.str();
	}

	// Rejects data from another level, edited layout, other game version/physics (fingerprint), or garbage.
	bool load(std::string const& str) {
		std::istringstream in(str);
		std::string tag, hx;
		int ver = 0;
		if (!(in >> tag >> ver >> hx) || tag != "AUTOBOT-LEARN" || ver != 1) return false;
		char want[32];
		std::snprintf(want, sizeof(want), "%016llx", (unsigned long long)fp);
		if (hx != want) return false;
		std::map<int, Site> fresh = sites;
		for (auto& kv : fresh) { Site s; s.key = kv.second.key; s.x0 = kv.second.x0; s.x1 = kv.second.x1; kv.second = s; }
		for (;;) {
			std::string t;
			if (!(in >> t)) return false;
			if (t == "END") break;
			if (t != "S") return false;
			long key, late, early, step, attempts, collapses, nj, nT, nC;
			if (!(in >> key >> late >> early >> step >> attempts >> collapses >> nj >> nT >> nC)) return false;
			auto it = fresh.find((int)key);
			if (it == fresh.end() || nT < 0 || nT > 64 || nC < 0 || nC > 64) return false;
			Site& s = it->second;
			s.late = (float)late / 100.f; s.early = (float)early / 100.f; s.step = (float)step / 100.f;
			s.attempts = (int)attempts; s.collapses = (int)collapses; s.noJump = nj != 0;
			for (long i = 0; i < nT; ++i) {
				std::string k; long d, r, p;
				if (!(in >> k >> d >> r >> p) || k != "T" || r < 0 || r > 4) return false;
				s.tried.push_back({(float)d / 100.f, (int)r, (float)p / 1000.f});
			}
			for (long i = 0; i < nC; ++i) {
				std::string k; long d, ok, bad;
				if (!(in >> k >> d >> ok >> bad) || k != "C") return false;
				s.clears.push_back({(float)d / 100.f, (int)ok, (int)bad});
			}
			// derived state
			if (!s.tried.empty()) {
				s.seed = s.tried.front().d;
				s.lastD = s.tried.back().d; s.lastRes = s.tried.back().res; s.lastProg = s.tried.back().prog;
				if (s.tried.size() > 1) s.prevProg = s.tried[s.tried.size() - 2].prog;
				s.improved = s.lastRes == R_CLEAR || (s.prevProg >= 0.f && s.lastProg > s.prevProg + 0.002f);
			}
		}
		sites = fresh;        // target is recomputed lazily on the first tick (needs the live frame size)
		dirty = false;
		return true;
	}
};

} // namespace learn
