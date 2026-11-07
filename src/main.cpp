#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

using namespace geode::prelude;

// ===========================================================================
// AutoBot v0.4 - lookahead planner with self-calibrating physics.
//
// Every frame the bot copies the player's state into a small physics sim,
// tries many hold/release sequences for the next ~1 s (beam search), drops
// every sequence that crashes, and keeps the one that stays furthest from
// danger. It applies only the first action, then re-plans next frame.
//
// New in v0.4:
//  * Ship physics are learned live, separately for "thrust while rising",
//    "thrust while falling", "gravity while falling" and "gravity while
//    rising", plus the real speed limits. The sim matches the real ship
//    better the longer you fly.
//  * Ship paths are scored for clearance in a wider band and prefer calm
//    (low vertical speed) states, so the ship settles in the middle of gaps.
//  * The ship ceiling is learned as well as the floor.
//  * Death learning: when the bot dies, nearby hazards get a bigger safety
//    margin for the next attempt, and the death is logged with the numbers
//    needed to tune the sim.
// ===========================================================================

// ---------------------------------------------------------------------------
// Gamemode detection
// ---------------------------------------------------------------------------
enum class Mode { Cube, Ship, Ball, UFO, Wave, Robot, Spider, Swing };

static Mode getMode(PlayerObject* p) {
	if (p->m_isShip)   return Mode::Ship;
	if (p->m_isBall)   return Mode::Ball;
	if (p->m_isBird)   return Mode::UFO;
	if (p->m_isDart)   return Mode::Wave;
	if (p->m_isRobot)  return Mode::Robot;
	if (p->m_isSpider) return Mode::Spider;
	if (p->m_isSwing)  return Mode::Swing;
	return Mode::Cube;
}

static char const* modeName(Mode m) {
	switch (m) {
		case Mode::Cube:   return "cube";
		case Mode::Ship:   return "ship";
		case Mode::Ball:   return "ball";
		case Mode::UFO:    return "ufo";
		case Mode::Wave:   return "wave";
		case Mode::Robot:  return "robot";
		case Mode::Spider: return "spider";
		case Mode::Swing:  return "swing";
	}
	return "?";
}

static bool supported(Mode m) {
	return m == Mode::Cube || m == Mode::Robot || m == Mode::Ship;
}

// ---------------------------------------------------------------------------
// Physics constants. Units are "level units per 1/60 s frame".
// ---------------------------------------------------------------------------
constexpr float BASE_SPEED = 311.58f;   // units/s at normal speed
namespace phys {
	constexpr float HALF = 15.f;            // player half size (30x30 hitbox)
	constexpr float GRAVITY = 0.958199f;    // cube gravity per frame^2
	constexpr float JUMP = 11.180032f;      // cube jump velocity
	constexpr float FALL_MAX = 15.f;        // cube terminal velocity
	constexpr float STEP = 0.5f;            // frames per substep
	constexpr int SUBSTEPS = 2;             // substeps per planned frame
	constexpr float SPEEDS[5] = {251.16f, 311.58f, 387.42f, 468.0f, 576.0f};
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------
struct Rc { float x0, y0, x1, y1; };

static Rc toRc(CCRect const& r) {
	return {r.getMinX(), r.getMinY(), r.getMaxX(), r.getMaxY()};
}
// Mirror vertically when gravity is flipped, so the sim always works "y-up".
static inline Rc mir(Rc const& r, float sgn) {
	return sgn > 0.f ? r : Rc{r.x0, -r.y1, r.x1, -r.y0};
}
static inline bool hit(Rc const& a, Rc const& b) {
	return a.x0 < b.x1 && a.x1 > b.x0 && a.y0 < b.y1 && a.y1 > b.y0;
}

// ---------------------------------------------------------------------------
// Level objects the bot cares about:
//   HAZARD  - spikes (all sizes), saw blades, thorns, any "deadly" object
//   SOLID   - blocks / platforms: land on top, die on the sides, bonk underside
//   ORB     - yellow, pink, red rings (need a click while touching)
//   PAD     - yellow, pink, red pads (fire on touch)
//   PORTAL  - cube/ship, gravity up/down, and the five speed portals
// Not handled yet: slopes, blue/green/black orbs, blue pads, ball/UFO/wave/
// robot/spider/swing portals, dual mode, mini size, moving objects.
// ---------------------------------------------------------------------------
enum Kind : uint8_t { K_SOLID, K_HAZARD, K_ORB, K_PAD, K_PORTAL };

struct Obj {
	Rc r;
	Kind k;
	int id = 0;
	int ix = -1;         // slot in the sim's "already triggered" bitmask
	bool used = false;   // orb/pad/portal already consumed in the real game
	float extra = 0.f;   // extra hazard margin learned from deaths
};

static bool isOrbId(int id)    { return id == 36 || id == 141 || id == 1333; }
static bool isPadId(int id)    { return id == 35 || id == 140 || id == 1332; }
static bool isPortalId(int id) {
	switch (id) {
		case 10: case 11:                                   // gravity normal / inverted
		case 12: case 13:                                   // cube / ship
		case 200: case 201: case 202: case 203: case 1334:  // speeds
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
	float y = 0.f;       // normalized: gravity always pulls toward -y
	float vy = 0.f;      // normalized as well
	float sgn = 1.f;     // +1 normal gravity, -1 flipped
	float speed = BASE_SPEED;
	bool ship = false;
	bool ground = false;
	bool held = false;
	uint64_t mask = 0;   // interactables already triggered
};

struct Env {
	std::vector<Obj> loc;   // objects near the player, sorted by x0
	float floorY = 90.f;
	float ceilY = 390.f;
	// Ship accelerations (units/frame^2). Index 1 = currently moving up.
	float up[2]   = {0.8f, 0.8f};   // while holding
	float down[2] = {0.8f, 0.8f};   // while released (magnitude, pulls down)
	float vUp = 8.f, vDown = 6.4f;  // ship velocity limits
	float margin = 2.f;             // hazard inflation = safety margin
	bool useOrbs = true;
};

static inline Rc box(Sim const& s, float h) {
	return {s.x - h, s.y - h, s.x + h, s.y + h};
}

static void flipGravity(Sim& s) {
	s.sgn = -s.sgn;
	s.y = -s.y;
	s.vy = -s.vy;
}

static void applyPortal(Sim& s, int id) {
	switch (id) {
		case 12: s.ship = false; break;
		case 13: s.ship = true; break;
		case 10: if (s.sgn < 0.f) flipGravity(s); break;
		case 11: if (s.sgn > 0.f) flipGravity(s); break;
		case 200: s.speed = phys::SPEEDS[0]; break;
		case 201: s.speed = phys::SPEEDS[1]; break;
		case 202: s.speed = phys::SPEEDS[2]; break;
		case 203: s.speed = phys::SPEEDS[3]; break;
		case 1334: s.speed = phys::SPEEDS[4]; break;
	}
}

// Advance one 1/60 s frame with the button held (or not). false = crashed.
static bool stepFrame(Sim& s, bool hold, Env const& e) {
	using namespace phys;
	bool edge = hold && !s.held;   // a fresh press: the only thing orbs react to
	s.held = hold;

	for (int i = 0; i < SUBSTEPS; ++i) {
		s.x += s.speed * (STEP / 60.f);
		float py = s.y;
		bool canJump = s.ground;
		s.ground = false;

		if (s.ship) {
			int ri = s.vy > 0.f ? 1 : 0;
			float a = hold ? e.up[ri] : -e.down[ri];
			s.vy = std::clamp(s.vy + a * STEP, -e.vDown, e.vUp);
		} else {
			if (canJump && hold) s.vy = JUMP;       // also re-jumps if held on landing
			s.vy = std::max(s.vy - GRAVITY * STEP, -FALL_MAX);
		}
		s.y += s.vy * STEP;

		Rc pr = box(s, HALF);
		for (auto const& o : e.loc) {
			if (o.r.x1 < s.x - 20.f) continue;
			if (o.r.x0 > s.x + 20.f) break;
			Rc r = mir(o.r, s.sgn);

			if (o.k == K_HAZARD) {
				float m = e.margin + o.extra;
				Rc h{r.x0 - m, r.y0 - m, r.x1 + m, r.y1 + m};
				if (hit(pr, h)) return false;
			} else if (o.k == K_SOLID) {
				if (!hit(pr, r)) continue;
				if (py - HALF >= r.y1 - 3.f && s.vy <= 0.f) {        // landed on top
					s.y = r.y1 + HALF;
					s.vy = 0.f;
					s.ground = true;
				} else if (py + HALF <= r.y0 + 3.f && s.vy >= 0.f) { // bonked underside
					s.y = r.y0 - HALF;
					s.vy = 0.f;
				} else {
					return false;                                    // hit a wall
				}
				pr = box(s, HALF);
			} else {
				if (o.ix < 0 || ((s.mask >> o.ix) & 1ull) || !hit(pr, r)) continue;
				uint64_t bit = 1ull << o.ix;
				if (o.k == K_PAD) {
					if (!s.ship) { s.vy = padVel(o.id); s.ground = false; s.mask |= bit; }
				} else if (o.k == K_ORB) {
					if (!s.ship && edge && e.useOrbs) {
						s.vy = orbVel(o.id);
						s.ground = false;
						s.mask |= bit;
						edge = false;
					}
				} else { // portal
					float before = s.sgn;
					applyPortal(s, o.id);
					s.mask |= bit;
					if (before != s.sgn) { py = -py; pr = box(s, HALF); }
				}
			}
		}

		// Floor / ceiling (in normalized space).
		float fl = s.sgn > 0.f ? e.floorY : -e.ceilY;
		float ce = s.sgn > 0.f ? e.ceilY : -e.floorY;
		if (s.y - HALF <= fl) {
			s.y = fl + HALF;
			if (s.vy < 0.f) s.vy = 0.f;
			s.ground = true;
		}
		if (s.ship && s.y + HALF >= ce) {
			s.y = ce - HALF;
			if (s.vy > 0.f) s.vy = 0.f;
		}
	}
	return true;
}

// Distance (units) from the player to the nearest thing that could hurt it.
// Hazards always count; solids and floor/ceiling only count for the ship,
// because a cube standing on a block is *supposed* to touch it.
static float clearance(Sim const& s, Env const& e) {
	float cap = s.ship ? 45.f : 30.f;
	float c = cap;
	Rc pr = box(s, phys::HALF);
	for (auto const& o : e.loc) {
		if (o.r.x1 < s.x - 60.f) continue;
		if (o.r.x0 > s.x + 60.f) break;
		if (!(o.k == K_HAZARD || (s.ship && o.k == K_SOLID))) continue;
		Rc r = mir(o.r, s.sgn);
		float dx = std::max({r.x0 - pr.x1, pr.x0 - r.x1, 0.f});
		float dy = std::max({r.y0 - pr.y1, pr.y0 - r.y1, 0.f});
		c = std::min(c, std::max(dx, dy));
	}
	if (s.ship) {
		float fl = s.sgn > 0.f ? e.floorY : -e.ceilY;
		float ce = s.sgn > 0.f ? e.ceilY : -e.floorY;
		c = std::min(c, std::min(s.y - phys::HALF - fl, ce - (s.y + phys::HALF)));
	}
	return std::max(c, 0.f);
}

static bool touchingOrb(Sim const& s, Env const& e) {
	Rc pr = box(s, phys::HALF);
	for (auto const& o : e.loc) {
		if (o.r.x1 < s.x - 20.f) continue;
		if (o.r.x0 > s.x + 20.f) break;
		if (o.k != K_ORB || o.ix < 0 || ((s.mask >> o.ix) & 1ull)) continue;
		if (hit(pr, mir(o.r, s.sgn))) return true;
	}
	return false;
}

// ---------------------------------------------------------------------------
// Beam search over hold/release sequences
// ---------------------------------------------------------------------------
struct BeamNode {
	Sim s;
	float clr = 60.f;      // worst clearance along this path
	int toggles = 0;
	int holds = 0;
	int first = -1;        // first action of this path
	float y1 = 0.f;        // predicted world y after the first frame
	uint64_t acts = 0;     // action bitmask, bit i = frame i
};

struct Plan {
	bool hold = false;
	int alive = 0;         // frames the best path survived
	float y1 = 0.f;
	uint64_t acts = 0;
};

static float scoreOf(BeamNode const& n) {
	// Bigger worst-case clearance first (centres jumps and ship paths), then
	// calm ship states, then fewer clicks, and wait rather than click early.
	float calm = n.s.ship ? std::abs(n.s.vy) * 0.3f : 0.f;
	return std::floor(n.clr) * 100.f - n.toggles * 6.f - n.holds * 0.2f - calm
	     - (n.first == 1 ? 1.f : 0.f);
}

static uint64_t keyOf(Sim const& s) {
	uint64_t h = 1469598103934665603ull;
	auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ull; };
	mix((uint32_t)(int32_t)std::lround(s.y * 1.5f));
	mix((uint32_t)(int32_t)std::lround(s.vy * 2.f));
	mix((s.ground ? 1u : 0u) | (s.held ? 2u : 0u) | (s.ship ? 4u : 0u) | (s.sgn > 0.f ? 8u : 0u));
	mix(s.mask);
	mix((uint32_t)(int32_t)std::lround(s.speed));
	return h;
}

static Plan search(Sim const& start, Env const& e, int H, int W) {
	static std::vector<BeamNode> cur, nxt;
	static std::unordered_map<uint64_t, size_t> seen;
	cur.clear();
	BeamNode root;
	root.s = start;
	cur.push_back(root);

	BeamNode best = root;
	int depth = 0;

	for (int f = 0; f < H; ++f) {
		nxt.clear();
		seen.clear();
		for (auto const& n : cur) {
			// Cube: only ground / orb frames offer a real choice.
			bool both = n.s.ship || n.s.ground || touchingOrb(n.s, e);
			for (int a = 0; a < (both ? 2 : 1); ++a) {
				BeamNode m = n;
				if (!stepFrame(m.s, a == 1, e)) continue;
				m.clr = std::min(m.clr, clearance(m.s, e));
				if ((a == 1) != n.s.held) ++m.toggles;
				if (a == 1) ++m.holds;
				if (f == 0) { m.first = a; m.y1 = m.s.y * m.s.sgn; }
				m.acts |= (uint64_t)a << f;

				uint64_t key = keyOf(m.s);
				auto it = seen.find(key);
				if (it == seen.end()) {
					seen[key] = nxt.size();
					nxt.push_back(m);
				} else if (scoreOf(m) > scoreOf(nxt[it->second])) {
					nxt[it->second] = m;
				}
			}
		}
		if (nxt.empty()) break; // every path crashed; keep the longest survivor
		std::sort(nxt.begin(), nxt.end(), [](BeamNode const& a, BeamNode const& b) {
			return scoreOf(a) > scoreOf(b);
		});
		if ((int)nxt.size() > W) nxt.resize(W);
		cur.swap(nxt);
		best = cur[0];
		depth = f + 1;
	}

	Plan p;
	p.alive = depth;
	p.hold = best.first == 1;
	p.y1 = best.y1;
	p.acts = best.acts;
	return p;
}

// ===========================================================================
// The mod
// ===========================================================================
class $modify(BotPlayLayer, PlayLayer) {
	struct Fields {
		std::vector<Obj> objs;   // sorted by x0
		bool holding = false;
		bool pressedLast = false;
		int clicks = 0;
		int deaths = 0;

		// telemetry
		float prevX = -1.f, prevY = 0.f, prevVN = 0.f;
		Mode prevMode = Mode::Cube;
		float speed = BASE_SPEED;

		// ship parameters learned while flying
		float up[2]   = {0.8f, 0.8f};
		float down[2] = {0.8f, 0.8f};
		float vUp = 8.f, vDown = 6.4f;
		float floorY = 90.f;
		float ceilY = 390.f;
		bool ceilLearned = false;

		// planner readout
		bool havePred = false;
		float predY = 0.f;
		float err = 0.f;
		int alive = 0;

		CCDrawNode* debug = nullptr;
		CCLabelBMFont* hud = nullptr;
	};

	void setupHasCompleted() {
		PlayLayer::setupHasCompleted();

		auto& f = m_fields;
		f->objs.clear();
		if (m_objects) {
			for (auto obj : CCArrayExt<GameObject*>(m_objects)) {
				int id = obj->m_objectID;
				Obj o;
				o.id = id;
				o.r = toRc(obj->getObjectRect());
				if (isPortalId(id))                                   o.k = K_PORTAL;
				else if (isOrbId(id))                                 o.k = K_ORB;
				else if (isPadId(id))                                 o.k = K_PAD;
				else if (obj->m_objectType == GameObjectType::Hazard) o.k = K_HAZARD;
				else if (obj->m_objectType == GameObjectType::Solid)  o.k = K_SOLID;
				else continue;
				f->objs.push_back(o);
			}
		}
		std::sort(f->objs.begin(), f->objs.end(), [](Obj const& a, Obj const& b) {
			return a.r.x0 < b.r.x0;
		});

		f->debug = CCDrawNode::create();
		m_objectLayer->addChild(f->debug, 1000);

		f->hud = CCLabelBMFont::create("AutoBot", "bigFont.fnt");
		f->hud->setScale(0.35f);
		f->hud->setOpacity(190);
		f->hud->setAnchorPoint({0.f, 1.f});
		auto win = CCDirector::get()->getWinSize();
		f->hud->setPosition({6.f, win.height - 6.f});
		this->addChild(f->hud, 1000);
	}

	void resetLevel() {
		PlayLayer::resetLevel();
		auto& f = m_fields;
		for (auto& o : f->objs) o.used = false;
		f->holding = false;
		f->pressedLast = false;
		f->prevX = -1.f;
		f->havePred = false;
	}

	// Death learning: log what happened and harden the hazards around the
	// crash so the next attempt gives them more room.
	void destroyPlayer(PlayerObject* player, GameObject* object) {
		if (player == m_player1 && !m_player1->m_isDead && Mod::get()->getSettingValue<bool>("enabled")) {
			onDeath(object);
		}
		PlayLayer::destroyPlayer(player, object);
	}

	void onDeath(GameObject* killer) {
		auto& f = m_fields;
		auto pos = m_player1->getPosition();
		++f->deaths;

		log::info(
			"[AutoBot] death #{} x={} y={} mode={} speed={} err={} plan={}f killerId={} up={}/{} down={}/{} vUp={} vDown={} floor={}",
			f->deaths, (double)pos.x, (double)pos.y, modeName(getMode(m_player1)), (double)f->speed,
			(double)f->err, f->alive, killer ? killer->m_objectID : -1,
			(double)f->up[0], (double)f->up[1], (double)f->down[0], (double)f->down[1],
			(double)f->vUp, (double)f->vDown, (double)f->floorY
		);

		if (!Mod::get()->getSettingValue<bool>("learn-from-deaths")) return;
		Rc zone{pos.x - 60.f, pos.y - 60.f, pos.x + 60.f, pos.y + 60.f};
		for (auto it = lowerBoundX(f->objs, pos.x - 160.f); it != f->objs.end() && it->r.x0 < pos.x + 60.f; ++it) {
			if (it->k == K_HAZARD && hit(zone, it->r)) it->extra = std::min(it->extra + 1.5f, 8.f);
		}
	}

	void postUpdate(float dt) {
		PlayLayer::postUpdate(dt);

		auto mod = Mod::get();
		auto& f = m_fields;
		auto p = m_player1;
		Mode mode = getMode(p);

		bool on = mod->getSettingValue<bool>("enabled");
		if (mod->getSettingValue<bool>("practice-only") && !m_isPracticeMode) on = false;
		if (!on || p->m_isDead) {
			release();
			f->prevX = -1.f;
			f->havePred = false;
			if (f->debug) f->debug->clear();
			updateHud(false, mode);
			return;
		}

		auto pos = p->getPosition();
		float sgn = p->m_isUpsideDown ? -1.f : 1.f;

		// ---- telemetry --------------------------------------------------
		bool fresh = f->prevX < 0.f;
		float v = static_cast<float>(p->m_yVelocity);
		if (!fresh && (pos.y - f->prevY) * v < -0.05f) v = -v; // sign convention guard
		float vN = v * sgn;

		if (fresh) {
			f->speed = BASE_SPEED * static_cast<float>(p->m_playerSpeed);
		} else if (dt > 1e-4f) {
			// Measure the real speed and snap it to the nearest known speed.
			float sp = (pos.x - f->prevX) / dt;
			for (float t : phys::SPEEDS) {
				if (std::abs(sp - t) < t * 0.06f) { f->speed = t; break; }
			}
			// Prediction error of last frame's plan (world y, in units).
			if (f->havePred && dt > 0.010f && dt < 0.025f) f->err = pos.y - f->predY;

			// Learn ship acceleration and limits from what the game really does.
			if (mode == Mode::Ship && f->prevMode == Mode::Ship && !p->m_isOnGround
				&& dt > 0.008f && dt < 0.03f) {
				float a = (vN - f->prevVN) / (dt * 60.f);
				bool saturated = vN >= f->vUp - 0.05f || vN <= -f->vDown + 0.05f;
				int ri = f->prevVN > 0.f ? 1 : 0;
				if (!saturated && std::abs(a) < 3.f) {
					if (f->holding && a > 0.05f)   f->up[ri]   += (a - f->up[ri]) * 0.15f;
					if (!f->holding && a < -0.05f) f->down[ri] += (-a - f->down[ri]) * 0.15f;
				}
				bool still = std::abs(vN - f->prevVN) < 0.01f;
				if (f->holding && vN > 1.f && still)   f->vUp   += (vN - f->vUp) * 0.5f;
				if (!f->holding && vN < -1.f && still) f->vDown += (-vN - f->vDown) * 0.5f;
				f->vUp   = std::min(12.f, std::max(f->vUp, vN));
				f->vDown = std::min(12.f, std::max(f->vDown, -vN));
			}
		}
		f->havePred = false;

		auto solidIn = [&](Rc q) {
			for (auto it = lowerBoundX(f->objs, pos.x - 200.f); it != f->objs.end() && it->r.x0 < pos.x + 40.f; ++it) {
				if (it->k == K_SOLID && hit(q, it->r)) return true;
			}
			return false;
		};

		// Learn the ship floor (resting on the real floor, not a block) and
		// the ceiling (rising fast, then stopped dead with nothing above).
		if (mode == Mode::Ship && sgn > 0.f) {
			if (p->m_isOnGround) {
				Rc below{pos.x - 14.f, pos.y - 19.f, pos.x + 14.f, pos.y - 14.5f};
				if (!solidIn(below)) f->floorY = pos.y - phys::HALF;
			}
			if (f->holding && f->prevMode == Mode::Ship && f->prevVN > 2.f && vN < 0.05f) {
				Rc above{pos.x - 14.f, pos.y + 14.5f, pos.x + 14.f, pos.y + 19.f};
				if (!solidIn(above)) { f->ceilY = pos.y + phys::HALF; f->ceilLearned = true; }
			}
		}

		// Mark pads/portals we are touching, and orbs we just clicked, as used.
		{
			Rc pr{pos.x - 15.f, pos.y - 15.f, pos.x + 15.f, pos.y + 15.f};
			for (auto it = lowerBoundX(f->objs, pos.x - 200.f); it != f->objs.end() && it->r.x0 < pos.x + 40.f; ++it) {
				if (it->used || !hit(pr, it->r)) continue;
				if (it->k == K_PAD || it->k == K_PORTAL) it->used = true;
				else if (it->k == K_ORB && f->pressedLast) it->used = true;
			}
		}
		f->pressedLast = false;

		f->prevX = pos.x;
		f->prevY = pos.y;
		f->prevVN = vN;
		f->prevMode = mode;

		// ---- plan ----------------------------------------------------------
		if (!supported(mode)) {
			release();
			if (f->debug) f->debug->clear();
			updateHud(true, mode);
			return;
		}

		int H = std::clamp<int>(static_cast<int>(mod->getSettingValue<int64_t>("horizon")), 10, 62);
		int W = std::clamp<int>(static_cast<int>(mod->getSettingValue<int64_t>("beam-width")), 8, 400);

		Env env;
		env.floorY = f->floorY;
		env.ceilY = f->ceilLearned ? f->ceilY : f->floorY + 300.f;
		for (int i = 0; i < 2; ++i) { env.up[i] = f->up[i]; env.down[i] = f->down[i]; }
		env.vUp = f->vUp; env.vDown = f->vDown;
		env.margin = static_cast<float>(mod->getSettingValue<double>("hazard-margin"));
		env.useOrbs = mod->getSettingValue<bool>("use-orbs");

		Sim s0;
		s0.x = pos.x;
		s0.sgn = sgn;
		s0.y = pos.y * sgn;
		s0.vy = vN;
		s0.speed = f->speed;
		s0.ship = mode == Mode::Ship;
		s0.ground = p->m_isOnGround;
		s0.held = f->holding;

		float reach = f->speed * (H / 60.f) + 90.f;
		int slot = 0;
		for (auto it = lowerBoundX(f->objs, pos.x - 200.f); it != f->objs.end() && it->r.x0 <= pos.x + reach; ++it) {
			if (it->r.x1 < pos.x - 60.f) continue;
			Obj o = *it;
			if (o.k == K_ORB || o.k == K_PAD || o.k == K_PORTAL) {
				if (slot < 64) {
					o.ix = slot++;
					if (o.used) s0.mask |= 1ull << o.ix;
				}
			}
			env.loc.push_back(o);
		}

		Plan pl = search(s0, env, H, W);
		f->alive = pl.alive;
		f->predY = pl.y1;
		f->havePred = pl.alive > 0;

		setHold(pl.hold);
		updateHud(true, mode);
		drawDebug(s0, env, pl);
	}

	// ---- Input -------------------------------------------------------------

	void setHold(bool want) {
		auto& f = m_fields;
		if (want == f->holding) return;
		f->holding = want;
		if (want) { ++f->clicks; f->pressedLast = true; }
		this->handleButton(want, 1, true); // 1 = jump, last arg = player 1
	}

	void release() {
		if (m_fields->holding) setHold(false);
	}

	// ---- HUD and debug overlay ---------------------------------------------

	void updateHud(bool active, Mode mode) {
		auto& f = m_fields;
		if (!f->hud) return;
		char buf[192];
		std::snprintf(buf, sizeof(buf), "AutoBot %s | %s%s | clicks %d | deaths %d",
			active ? "ON" : "OFF", modeName(mode),
			(active && !supported(mode)) ? " (unsupported)" : "", f->clicks, f->deaths);
		std::string s = buf;
		if (active && supported(mode)) {
			std::snprintf(buf, sizeof(buf), " | plan %df | err %.1f", f->alive, (double)f->err);
			s += buf;
			if (mode == Mode::Ship) {
				std::snprintf(buf, sizeof(buf), " | up %.2f/%.2f dn %.2f/%.2f v %.1f/%.1f",
					(double)f->up[0], (double)f->up[1], (double)f->down[0], (double)f->down[1],
					(double)f->vUp, (double)f->vDown);
				s += buf;
			}
		}
		f->hud->setString(s.c_str());
	}

	void drawDebug(Sim const& s0, Env const& env, Plan const& pl) {
		auto d = m_fields->debug;
		if (!d) return;
		d->clear();
		if (!Mod::get()->getSettingValue<bool>("debug-draw")) return;

		auto drawRect = [&](Rc const& r, ccColor4F col) {
			CCPoint v[4] = {
				{r.x0, r.y0}, {r.x1, r.y0}, {r.x1, r.y1}, {r.x0, r.y1}
			};
			d->drawPolygon(v, 4, {0, 0, 0, 0}, 1.f, col);
		};
		for (auto const& o : env.loc) {
			if (o.r.x1 < s0.x) continue;
			ccColor4F col = o.k == K_HAZARD ? ccColor4F{1.f, 0.2f, 0.2f, 1.f}
			              : o.k == K_SOLID  ? ccColor4F{0.3f, 0.6f, 1.f, 0.8f}
			              : o.k == K_PORTAL ? ccColor4F{0.8f, 0.4f, 1.f, 1.f}
			                                : ccColor4F{1.f, 0.9f, 0.2f, 1.f};
			drawRect(o.r, col);
		}

		// Re-simulate the chosen plan and draw the path the bot expects to take.
		Sim s = s0;
		CCPoint prev = {s.x, s.y * s.sgn};
		for (int i = 0; i < pl.alive; ++i) {
			if (!stepFrame(s, ((pl.acts >> i) & 1ull) != 0, env)) break;
			CCPoint cur = {s.x, s.y * s.sgn};
			d->drawSegment(prev, cur, 1.2f, {0.2f, 1.f, 0.4f, 1.f});
			prev = cur;
		}
	}
};
