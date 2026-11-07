#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <algorithm>
#include <cmath>
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

// Base horizontal speed at 1x in units per second (multiplied by m_playerSpeed).
constexpr float BASE_SPEED = 311.58f;

class $modify(BotPlayLayer, PlayLayer) {
	struct Fields {
		std::vector<CCRect> hazards;   // spikes, saws, etc. sorted by minX
		bool holding = false;
		CCDrawNode* debug = nullptr;
	};

	// Build the hazard cache once the level has finished loading.
	void setupHasCompleted() {
		PlayLayer::setupHasCompleted();

		auto& hz = m_fields->hazards;
		hz.clear();

				if (m_objects) {
			for (auto obj : CCArrayExt<GameObject*>(m_objects)) {
				if (obj->m_objectType == GameObjectType::Hazard) {
					// Use the real hitbox, not the sprite bounds.
					hz.push_back(obj->getObjectRect());
				}
			}
		}
		std::sort(hz.begin(), hz.end(), [](CCRect const& a, CCRect const& b) {
			return a.getMinX() < b.getMinX();
		});

		m_fields->debug = CCDrawNode::create();
		m_objectLayer->addChild(m_fields->debug, 1000);
	}

	// Runs every frame after the game's own update.
	void postUpdate(float dt) {
		PlayLayer::postUpdate(dt);

		auto mod = Mod::get();
		if (!mod->getSettingValue<bool>("enabled")) return release();
		if (mod->getSettingValue<bool>("practice-only") && !m_isPracticeMode) return release();
		if (m_player1->m_isDead) return release();

		bool want = false;
		switch (getMode(m_player1)) {
			case Mode::Cube:
			case Mode::Robot: // robot is cube-like; hold length is not handled yet
				want = decideCube();
				break;
			// TODO: Ship, UFO, Wave, Ball, Spider, Swing controllers.
			default:
				want = false;
				break;
		}
		setHold(want);
		drawDebug();
	}

	// ---- Controllers ------------------------------------------------------

	// First-pass cube controller: click when a hazard's hitbox is about to
	// enter the jump window. Approximation, NOT a physics simulation yet.
	bool decideCube() {
		auto p = m_player1;
		if (!p->m_isOnGround) return false; // already airborne: release

		float speed = BASE_SPEED * p->m_playerSpeed;
		float lead = static_cast<float>(Mod::get()->getSettingValue<double>("lead-time"));
		float window = speed * lead;

		auto pos = p->getPosition();
		float px = pos.x;
		float py = pos.y;

		for (auto const& r : m_fields->hazards) {
			if (r.getMaxX() < px) continue;          // behind us
			float dx = r.getMinX() - px;
			if (dx > window) break;                  // sorted: nothing closer remains
			float cy = r.getMidY();
			// Hazard sits in our lane (works upside-down too).
			if (std::abs(cy - py) < 45.f) return true;
		}
		return false;
	}

	// ---- Input helpers ----------------------------------------------------

	void setHold(bool want) {
		auto& f = m_fields;
		if (want == f->holding) return;
		f->holding = want;
		// button 1 = jump, last arg = player 1
		this->handleButton(want, 1, true);
	}

	void release() {
		if (m_fields->holding) setHold(false);
	}

	// ---- Debug overlay ----------------------------------------------------

	void drawDebug() {
		auto d = m_fields->debug;
		if (!d) return;
		d->clear();
		if (!Mod::get()->getSettingValue<bool>("debug-draw")) return;

		float px = m_player1->getPositionX();
		for (auto const& r : m_fields->hazards) {
			if (r.getMaxX() < px) continue;
			if (r.getMinX() > px + 600.f) break;
			CCPoint v[4] = {
				{r.getMinX(), r.getMinY()}, {r.getMaxX(), r.getMinY()},
				{r.getMaxX(), r.getMaxY()}, {r.getMinX(), r.getMaxY()}
			};
			d->drawPolygon(v, 4, {0, 0, 0, 0}, 1.f, {1.f, 0.2f, 0.2f, 1.f});
		}
	}
};
