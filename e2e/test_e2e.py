"""Full user journey through the real UI (headless Chromium, python Playwright).

Run via ./run_e2e.sh, which starts the backend (port 8004) and a production build of the
frontend (port 5176). Screenshots land in docs/screenshots/.
"""
from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

from playwright.sync_api import Page, expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get("E2E_BASE_URL", "http://127.0.0.1:5176")
SHOTS = ROOT / "docs" / "screenshots"
TESTSET = ROOT / "backend" / "ml" / "testset" / "images"
FIXTURES = ROOT / "backend" / "tests" / "fixtures"


def photo(name: str, fixture: str) -> Path:
    p = TESTSET / name
    return p if p.exists() else FIXTURES / fixture


def shot(page: Page, name: str) -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    expect(page.locator(".toast")).to_have_count(0, timeout=8000)  # no stale toasts in screenshots
    page.wait_for_timeout(350)  # let bar transitions finish
    page.screenshot(path=str(SHOTS / f"{name}.png"), full_page=False)


def login(page: Page, email: str, password: str) -> None:
    page.goto(f"{BASE}/login")
    page.get_by_label("Email").fill(email)
    page.get_by_label("Password").fill(password)
    page.get_by_role("button", name="Sign in").click()
    expect(page.get_by_role("heading", name="What’s on your plate?")).to_be_visible()


def test_full_journey():
    email = f"e2e_{uuid.uuid4().hex[:8]}@example.com"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))

        # 1. register -> lands on profile setup
        page.goto(f"{BASE}/register")
        expect(page.get_by_role("heading", name="Create your account")).to_be_visible()
        page.get_by_label("Name").fill("Priya")
        page.get_by_label("Email").fill(email)
        page.get_by_label("Password").fill("secret123")
        page.get_by_role("button", name="Create account").click()
        expect(page.get_by_role("heading", name="Set your daily target")).to_be_visible()

        # 2. complete profile -> targets with the formula
        page.get_by_label("Age").fill("21")
        page.get_by_label("Height").fill("165")
        page.get_by_label("Weight").fill("58")
        page.get_by_role("button", name="Female").click()
        page.get_by_label("Activity").select_option("moderate")
        page.get_by_role("button", name="Maintain").click()
        # Mifflin-St Jeor: 10*58 + 6.25*165 - 5*21 - 161 = 1345.25; x1.55 = 2085 -> 2090
        expect(page.get_by_test_id("target-kcal")).to_have_text("2,090")
        expect(page.get_by_test_id("targets")).to_contain_text("10 x 58 kg + 6.25 x 165 cm - 5 x 21 y - 161")
        page.get_by_test_id("save-profile").click()
        expect(page.get_by_test_id("save-profile")).to_contain_text("Saved")

        # 3. first save redirects to Snap; upload a real pizza photo
        expect(page.get_by_role("heading", name="What’s on your plate?")).to_be_visible(timeout=5000)
        page.get_by_test_id("file-input").set_input_files(str(photo("pizza/pizza_1.jpg", "pizza.jpg")))
        matches = page.get_by_role("group", name="Top matches")
        expect(matches).to_be_visible(timeout=30000)
        expect(matches.get_by_role("button")).to_have_count(5)
        pizza_chip = matches.get_by_role("button", name=re.compile(r"^Margherita Pizza"))
        expect(pizza_chip).to_be_visible()
        pizza_chip.click()
        expect(page.get_by_test_id("picked-food")).to_have_text("Margherita Pizza")

        # the result explains itself: provenance + a decision/confidence line
        expect(page.get_by_test_id("nutrition-source")).to_contain_text("USDA verified")

        # 4. choose portion: 2 slices (100 g each, 266 kcal/100 g from USDA FDC 2708616)
        page.get_by_label("Serving").select_option("1 slice")
        page.get_by_role("button", name="More").click()  # steps are half servings
        page.get_by_role("button", name="More").click()
        expect(page.get_by_test_id("qty")).to_have_value("2")
        expect(page.get_by_test_id("portion-kcal")).to_have_text("532")
        page.get_by_role("group", name="Meal").get_by_role("button", name="Dinner").click()
        shot(page, "snap-desktop")
        page.get_by_test_id("save-meal").click()
        expect(page.get_by_test_id("saved")).to_contain_text("Logged Margherita Pizza")

        # 5. diary shows the meal; edit the portion
        page.get_by_role("link", name="View diary").click()
        row = page.get_by_test_id("meal-row").filter(has_text="Margherita Pizza")
        expect(row).to_be_visible()
        expect(row.get_by_test_id("meal-kcal")).to_have_text("532")
        expect(page.get_by_test_id("day-kcal")).to_have_text("532")
        row.click()
        page.get_by_role("button", name="More").click()
        expect(page.get_by_test_id("update-meal")).to_contain_text("665")
        page.get_by_test_id("update-meal").click()
        expect(row.get_by_test_id("meal-kcal")).to_have_text("665")
        expect(row).to_contain_text("2.5 × 1 slice · 250 g")

        # 6. trends renders with that day
        page.get_by_role("link", name="Trends").first.click()
        expect(page.get_by_test_id("avg-kcal")).to_contain_text("665")
        expect(page.get_by_role("img", name=re.compile("Calories per day"))).to_be_visible()
        expect(page.get_by_text("Margherita Pizza", exact=True)).to_be_visible()

        # 6b. delete goes through a confirm dialog
        page.get_by_role("link", name="Diary").first.click()
        row = page.get_by_test_id("meal-row").filter(has_text="Margherita Pizza")
        row.click()
        page.get_by_role("button", name="Remove").click()
        dialog = page.get_by_role("alertdialog")
        expect(dialog).to_be_visible()
        dialog.get_by_role("button", name="Cancel").click()
        expect(dialog).to_have_count(0)
        expect(row).to_be_visible()

        # 6c. model card page
        page.get_by_role("link", name="Model", exact=True).first.click()
        expect(page.get_by_role("heading", name="How recognition works")).to_be_visible()
        expect(page.get_by_text("not trained by this project")).to_be_visible()
        expect(page.get_by_text("Measured quality")).to_be_visible()
        shot(page, "model-desktop")

        # 7. screenshots with the seeded demo account (a month of data)
        page.get_by_role("button", name="Sign out").click()
        page.get_by_role("button", name="Use demo account").click()
        expect(page.get_by_label("Email")).to_have_value("demo@nutrisnap.app")
        page.get_by_role("button", name="Sign in").click()
        expect(page.get_by_role("heading", name="What’s on your plate?")).to_be_visible()
        expect(page.get_by_role("note")).to_contain_text("generated sample data")
        page.get_by_test_id("file-input").set_input_files(str(photo("dosa/masala-dosa_1.jpg", "dosa.jpg")))
        expect(page.get_by_role("group", name="Top matches")).to_be_visible(timeout=30000)
        expect(page.get_by_test_id("result-state")).to_be_visible()
        shot(page, "snap-uncertain-desktop")
        page.get_by_role("link", name="Diary").first.click()
        page.get_by_role("button", name="Previous day").click()
        expect(page.get_by_test_id("meal-row").first).to_be_visible()
        shot(page, "diary-desktop")
        page.get_by_role("link", name="Trends").first.click()
        page.get_by_role("button", name="30 days").click()
        expect(page.get_by_test_id("avg-kcal")).to_be_visible()
        shot(page, "trends-desktop")
        assert not errors, errors
        ctx.close()

        # 8. mobile (375 px): snap result, diary, trends
        m = browser.new_context(viewport={"width": 375, "height": 812}, device_scale_factor=2, is_mobile=True, has_touch=True)
        mp = m.new_page()
        login(mp, "demo@nutrisnap.app", "demo1234")
        assert mp.evaluate("document.documentElement.scrollWidth") <= 375
        mp.get_by_test_id("file-input").set_input_files(str(photo("pani-puri/pani-puri_6.jpg", "pizza.jpg")))
        expect(mp.get_by_role("group", name="Top matches")).to_be_visible(timeout=30000)
        expect(mp.get_by_test_id("portion-kcal")).to_be_visible()
        mp.evaluate("window.scrollTo(0, 170)")
        shot(mp, "snap-mobile")
        assert mp.evaluate("document.documentElement.scrollWidth") <= 375
        mp.get_by_role("navigation", name="Main").get_by_role("link", name="Diary").click()
        mp.get_by_role("button", name="Previous day").click()
        expect(mp.get_by_test_id("meal-row").first).to_be_visible()
        shot(mp, "diary-mobile")
        mp.get_by_role("navigation", name="Main").get_by_role("link", name="Trends").click()
        expect(mp.get_by_test_id("avg-kcal")).to_be_visible()
        shot(mp, "trends-mobile")
        assert mp.evaluate("document.documentElement.scrollWidth") <= 375
        m.close()
        browser.close()
