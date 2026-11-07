#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

using namespace geode::prelude;

// ---------------------------------------------------------------------------
// Gamemode detection: the player object exposes flags, so no icon guessing.
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

// Orb object IDs: yellow, blue (gravity), pink, green, black, red.
static bool isOrbId(int id) {
	return id == 36 || id == 84 || id == 141 || id == 1022 || id == 1330 || id == 1333;
}

// Base horizontal speed at 1x in units per second (multiplied by m_playerSpeed).
constexpr float BASE_SPEED = 311.58f;

struct Orb {
	CCRect rect;
	bool used = false;
};

class $modify(BotPlayLayer, PlayLayer) {
	struct Fields {
		std::vector<CCRect> hazards;  // sorted by minX
		std::vector<CCRect> solids;   // sorted by minX
		std::vector<Orb> orbs;        // sorted by minX
		bool holding = false;
		int pulse = 0;                // frames left on an orb click
		int clicks = 0;
		CCDrawNode* debug = nullptr;
		CCLabelBMFont* hud = nullptr;
	};

	// Build the object caches once the level has finished loading.
	void setupHasCompleted() {
		PlayLayer::setupHasCompleted();

		auto& f = m_fields;
		f->hazards.clear();
		f->solids.clear();
		f->orbs.clear();

		if (m_objects) {
			for (auto obj : CCArrayExt<GameObject*>(m_objects)) {
				if (obj->m_objectType == GameObjectType::Hazard) {
					f->hazards.push_back(obj->getObjectRect());
				} else if (obj->m_objectType == GameObjectType::Solid) {
					f->solids.push_back(obj->getObjectRect());
				} else if (isOrbId(obj->m_objectID)) {
					f->orbs.push_back(Orb{obj->getObjectRect(), false});
				}
			}
		}
		auto byX = [](CCRect const& a, CCRect const& b) {
			return a.getMinX() < b.getMinX();
		};
		std::sort(f->hazards.begin(), f->hazards.end(), byX);
		std::sort(f->solids.begin(), f->solids.end(), byX);
		std::sort(f->orbs.begin(), f->orbs.end(), [](Orb const& a, Orb const& b) {
			return a.rect.getMinX() < b.rect.getMinX();
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
		for (auto& o : f->orbs) o.used = false;
		f->holding = false;
		f->pulse = 0;
	}

	// Runs every frame after the game's own update.
	void postUpdate(float dt) {
		PlayLayer::postUpdate(dt);

		auto mod = Mod::get();
		bool on = mod->getSettingValue<bool>("enabled");
		if (mod->getSettingValue<bool>("practice-only") && !m_isPracticeMode) on = false;
		if (!on || m_player1->m_isDead) {
			release();
			updateHud(false, getMode(m_player1));
			return;
		}

		Mode mode = getMode(m_player1);
		bool want = false;

		switch (mode) {
			case Mode::Cube:
			case Mode::Robot: // robot hold length is not handled yet
				want = decideCube();
				break;
			case Mode::Ship:
				want = decideShip();
				break;
			default: // TODO: UFO, Wave, Ball, Spider, Swing
				want = false;
				break;
		}

		// Orbs (cube/robot/ship only for now).
		if (supported(mode) && mod->getSettingValue<bool>("use-orbs")) {
			if (checkOrbs()) m_fields->pulse = 3;
		}
		if (m_fields->pulse > 0) {
			--m_fields->pulse;
			want = true;
		}

		setHold(want);
		updateHud(true, mode);
		drawDebug();
	}

	// ---- Controllers ------------------------------------------------------

	// Cube: click when a hazard's hitbox is about to enter the jump window.
	// Approximation, NOT a physics simulation yet.
	bool decideCube() {
		auto p = m_player1;
		if (!p->m_isOnGround) return false;

		float speed = BASE_SPEED * p->m_playerSpeed;
		float lead = static_cast<float>(Mod::get()->getSettingValue<double>("lead-time"));
		float window = speed * lead;

		auto pos = p->getPosition();
		for (auto const& r : m_fields->hazards) {
			if (r.getMaxX() < pos.x) continue;
			if (r.getMinX() - pos.x > window) break;
			if (std::abs(r.getMidY() - pos.y) < 45.f) return true;
		}
		return false;
	}

	// Ship: find the floor and ceiling just ahead, then hold/release to steer
	// toward the middle of the gap (or hold altitude if there is no gap).
	bool decideShip() {
		auto p = m_player1;
		auto pos = p->getPosition();
		float speed = BASE_SPEED * p->m_playerSpeed;
		float look = speed * 0.35f;

		constexpr float NONE = std::numeric_limits<float>::infinity();
		float floorY = -NONE;
		float ceilY = NONE;

		auto scan = [&](std::vector<CCRect> const& v) {
			for (auto const& r : v) {
				if (r.getMaxX() < pos.x - 15.f) continue;
				if (r.getMinX() - pos.x > look) break;
				if (r.getMidY() < pos.y) floorY = std::max(floorY, r.getMaxY());
				else ceilY = std::min(ceilY, r.getMinY());
			}
		};
		scan(m_fields->solids);
		scan(m_fields->hazards);

		float target;
		bool hasFloor = floorY > -NONE;
		bool hasCeil = ceilY < NONE;
		if (hasFloor && hasCeil) target = (floorY + ceilY) * 0.5f;
		else if (hasFloor)       target = floorY + 45.f;
		else if (hasCeil)        target = ceilY - 45.f;
		else                     target = pos.y;

		float predicted = pos.y + static_cast<float>(p->m_yVelocity) * 0.1f;
		bool rise = predicted < target;
		return p->m_isUpsideDown ? !rise : rise;
	}

	// True once when the player first touches an unused orb.
	bool checkOrbs() {
		auto pr = m_player1->getObjectRect();
		float px = m_player1->getPositionX();
		for (auto& o : m_fields->orbs) {
			if (o.rect.getMaxX() < px - 40.f) continue;
			if (o.rect.getMinX() > px + 40.f) break;
			if (!o.used && pr.intersectsRect(o.rect)) {
				o.used = true;
				return true;
			}
		}
		return false;
	}

	// ---- Input helpers ----------------------------------------------------

	void setHold(bool want) {
		auto& f = m_fields;
		if (want == f->holding) return;
		f->holding = want;
		if (want) ++f->clicks;
		this->handleButton(want, 1, true); // 1 = jump, last arg = player 1
	}

	void release() {
		if (m_fields->holding) setHold(false);
	}

	// ---- HUD and debug overlay -------------------------------------------

	void updateHud(bool active, Mode mode) {
		auto hud = m_fields->hud;
		if (!hud) return;
		std::string s = active ? "AutoBot ON" : "AutoBot OFF";
		s += " | ";
		s += modeName(mode);
		if (active && !supported(mode)) s += " (unsupported)";
		s += " | clicks " + std::to_string(m_fields->clicks);
		hud->setString(s.c_str());
	}

	void drawDebug() {
		auto d = m_fields->debug;
		if (!d) return;
		d->clear();
		if (!Mod::get()->getSettingValue<bool>("debug-draw")) return;

		float px = m_player1->getPositionX();
		auto draw = [&](CCRect const& r, ccColor4F col) {
			CCPoint v[4] = {
				{r.getMinX(), r.getMinY()}, {r.getMaxX(), r.getMinY()},
				{r.getMaxX(), r.getMaxY()}, {r.getMinX(), r.getMaxY()}
			};
			d->drawPolygon(v, 4, {0, 0, 0, 0}, 1.f, col);
		};
		for (auto const& r : m_fields->hazards) {
			if (r.getMaxX() < px) continue;
			if (r.getMinX() > px + 600.f) break;
			draw(r, {1.f, 0.2f, 0.2f, 1.f});
		}
		for (auto const& o : m_fields->orbs) {
			if (o.rect.getMaxX() < px) continue;
			if (o.rect.getMinX() > px + 600.f) break;
			draw(o.rect, {1.f, 0.9f, 0.2f, 1.f});
		}
	}
};
