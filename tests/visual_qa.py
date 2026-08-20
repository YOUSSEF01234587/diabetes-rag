"""Playwright visual QA — Clinical Evidence Copilot (corrected selectors)."""
import asyncio
import os
from playwright.async_api import async_playwright

BASE = "http://localhost:3000"
SCREENSHOTS = os.path.join(os.path.dirname(__file__), "screenshots")
os.makedirs(SCREENSHOTS, exist_ok=True)

VIEWPORTS = {
    "desktop_1440": {"width": 1440, "height": 900},
    "desktop_1280": {"width": 1280, "height": 800},
    "desktop_1024": {"width": 1024, "height": 768},
    "tablet_768":   {"width": 768, "height": 1024},
    "mobile_390":   {"width": 390, "height": 844},
    "mobile_375":   {"width": 375, "height": 812},
}

RESULTS = {"pass": 0, "fail": 0, "warn": 0, "info": 0,
           "console_errors": [], "screenshots": [], "issues": []}

def log(status, msg):
    tag = {"pass": "PASS", "fail": "FAIL", "warn": "WARN", "info": "INFO"}[status]
    RESULTS[status] = RESULTS.get(status, 0) + 1
    if status == "fail":
        RESULTS["issues"].append(msg)
    print(f"  [{tag}] {msg}")

async def screenshot(page, name):
    path = os.path.join(SCREENSHOTS, f"{name}.png")
    await page.screenshot(path=path, full_page=False)
    RESULTS["screenshots"].append(path)
    print(f"  -> {name}.png")
    return path

async def check_overflow(page, label):
    issues = await page.evaluate("""() => {
        const issues = [];
        if (document.body.scrollWidth > document.body.clientWidth + 2)
            issues.push('body: sw=' + document.body.scrollWidth + ' cw=' + document.body.clientWidth);
        if (document.documentElement.scrollWidth > document.documentElement.clientWidth + 2)
            issues.push('html: sw=' + document.documentElement.scrollWidth + ' cw=' + document.documentElement.clientWidth);
        document.querySelectorAll('.answer-text,.answer-card,.message-content,.refusal-card,.message-body,.welcome-screen,.chat-container,.input-wrapper').forEach(function(el, i) {
            if (el.scrollWidth > el.clientWidth + 2)
                issues.push(el.tagName + '.' + el.className.split(' ')[0] + '[' + i + ']: sw=' + el.scrollWidth + ' cw=' + el.clientWidth);
        });
        return issues;
    }""")
    if issues:
        log("fail", f"Horizontal overflow {label}: {issues}")
    else:
        log("pass", f"No horizontal overflow {label}")
    return issues

async def test_landing(page, vp_name):
    print(f"\n{'='*60}")
    print(f"TEST: Landing Page - {vp_name}")
    print(f"{'='*60}")

    await page.goto(BASE, wait_until="networkidle", timeout=15000)
    await page.wait_for_timeout(500)

    header = await page.query_selector(".app-header")
    log("pass" if header else "fail", "Header " + ("exists" if header else "missing"))

    brand = await page.query_selector(".brand-text h1")
    if brand:
        log("pass", f"Brand title: '{await brand.inner_text()}'")
    else:
        log("fail", "Brand title missing")

    sub = await page.query_selector(".brand-sub")
    if sub:
        log("pass", f"Brand subtitle present")
    else:
        log("fail", "Brand subtitle missing")

    sidebar = await page.query_selector(".sidebar")
    log("pass" if sidebar else "fail", "Sidebar " + ("exists" if sidebar else "missing"))

    toggle = await page.query_selector(".sidebar-toggle")
    if toggle:
        display = await toggle.evaluate("el => getComputedStyle(el).display")
        vp_w = int(vp_name.split("_")[1])
        if vp_w <= 1024:
            log("pass" if display != "none" else "warn", f"Sidebar toggle on {vp_name}: display={display}")
        else:
            log("pass" if display == "none" else "warn", f"Sidebar toggle on {vp_name}: display={display}")

    welcome = await page.query_selector("#welcome-screen")
    if welcome:
        visible = await welcome.evaluate("el => el.style.display !== 'none'")
        log("pass" if visible else "fail", "Welcome screen " + ("visible" if visible else "hidden"))
    else:
        log("fail", "Welcome screen missing")

    icon = await page.query_selector(".welcome-icon svg")
    log("pass" if icon else "fail", "Welcome icon SVG " + ("exists" if icon else "missing"))

    h2 = await page.query_selector(".welcome-hero h2")
    if h2:
        log("pass", f"Welcome title: '{await h2.inner_text()}'")
    else:
        log("fail", "Welcome title missing")

    desc = await page.query_selector(".welcome-desc")
    if desc:
        log("pass", f"Welcome description present ({len(await desc.inner_text())} chars)")
    else:
        log("fail", "Welcome description missing")

    flow_steps = await page.query_selector_all(".flow-step")
    log("pass" if len(flow_steps) >= 4 else "fail", f"Flow steps: {len(flow_steps)}")

    examples = await page.query_selector_all(".example-btn")
    log("pass" if len(examples) >= 4 else "fail", f"Example buttons: {len(examples)}")
    for btn in examples:
        label = await btn.query_selector(".example-label")
        desc_el = await btn.query_selector(".example-desc")
        if label:
            lt = await label.inner_text()
            dt = await desc_el.inner_text() if desc_el else ""
            print(f"    - {lt}: {dt}")

    input_el = await page.query_selector("#user-input")
    log("pass" if input_el else "fail", "Input " + ("exists" if input_el else "missing"))

    send_btn = await page.query_selector("#send-btn")
    log("pass" if send_btn else "fail", "Send button " + ("exists" if send_btn else "missing"))

    chat = await page.query_selector(".chat-container")
    log("pass" if chat else "fail", "Chat container " + ("exists" if chat else "missing"))

    drawer = await page.query_selector("#evidence-drawer")
    log("pass" if drawer else "fail", "Evidence drawer " + ("exists" if drawer else "missing"))

    overlay = await page.query_selector("#drawer-overlay")
    log("pass" if overlay else "fail", "Drawer overlay " + ("exists" if overlay else "missing"))

    status = await page.query_selector("#engine-status")
    if status:
        log("pass", f"Engine status: '{(await status.inner_text()).strip()}'")
    else:
        log("fail", "Engine status missing")

    pipeline = await page.query_selector("#pipeline-view")
    if pipeline:
        steps = await pipeline.query_selector_all(".pipeline-step")
        log("pass", f"Pipeline view: {len(steps)} steps")
    else:
        log("fail", "Pipeline view missing")

    disclaimer = await page.query_selector(".sidebar-disclaimer")
    log("pass" if disclaimer else "warn", "Disclaimer " + ("present" if disclaimer else "missing"))

    diff = await page.query_selector(".welcome-different")
    log("pass" if diff else "warn", "Decision Path section " + ("exists" if diff else "missing"))

    await screenshot(page, f"01_landing_{vp_name}")
    await check_overflow(page, f"landing {vp_name}")

async def test_drawer_closed(page):
    print(f"\n{'='*60}")
    print("TEST: Drawer Closed State")
    print(f"{'='*60}")

    layout = await page.query_selector(".app-layout")
    classes = await layout.get_attribute("class")
    log("pass", f"Layout classes: {classes}")

    drawer = await page.query_selector("#evidence-drawer")
    drawer_classes = await drawer.evaluate("el => el.className")
    has_open = "open" in drawer_classes
    log("pass" if not has_open else "fail", f"Drawer has 'open': {has_open}")

    grid = await page.evaluate("() => getComputedStyle(document.querySelector('.app-layout')).gridTemplateColumns")
    log("pass", f"Grid template: {grid}")

    dw = await page.evaluate("() => document.querySelector('#evidence-drawer').getBoundingClientRect().width")
    log("pass" if dw == 0 else "warn", f"Drawer width: {dw}px")

    await screenshot(page, "02_drawer_closed")

async def test_chat(page):
    question = "What fasting plasma glucose threshold is used to diagnose diabetes?"
    print(f"\n{'='*60}")
    print(f"TEST: Chat - {question}")
    print(f"{'='*60}")

    input_el = await page.query_selector("#user-input")
    await input_el.click()
    await input_el.fill(question)
    await page.wait_for_timeout(200)

    before = await page.query_selector_all(".message.user")
    log("pass", f"User messages before submit: {len(before)}")

    await page.keyboard.press("Enter")
    await page.wait_for_selector(".message.user", timeout=5000)
    user_msgs = await page.query_selector_all(".message.user")
    log("pass", f"User message appeared (total: {len(user_msgs)})")

    await page.wait_for_timeout(500)
    loading = await page.query_selector(".loading-message")
    log("pass" if loading else "warn", "Loading state " + ("appeared" if loading else "not found"))

    try:
        await page.wait_for_selector(".message.assistant .answer-card", timeout=90000)
        log("pass", "Assistant response appeared")
    except Exception as e:
        log("fail", f"No response within 90s: {e}")
        await screenshot(page, "03_chat_timeout")
        return False

    await page.wait_for_timeout(1000)

    answer_card = await page.query_selector(".message.assistant .answer-card")
    log("pass" if answer_card else "fail", "Answer card " + ("exists" if answer_card else "missing"))
    if not answer_card:
        return False

    answer_label = await page.query_selector(".answer-label")
    if answer_label:
        log("pass", f"Answer label: '{await answer_label.inner_text()}'")
    else:
        refusal = await page.query_selector(".refusal-card")
        log("info", "Response is a refusal" if refusal else "No answer-label, no refusal-card")

    answer_text = await page.query_selector(".answer-text")
    if answer_text:
        text = await answer_text.inner_text()
        log("pass", f"Answer text: {len(text)} chars")
        raw_md = any(m in text for m in ["**", "## ", "### "])
        log("pass" if not raw_md else "fail", f"Raw markdown: {raw_md}")
        log("pass" if any(c.isdigit() for c in text) else "warn", f"Numeric values: {any(c.isdigit() for c in text)}")
    else:
        log("fail", "No .answer-text found")

    refusal = await page.query_selector(".refusal-card")
    log("info" if refusal else "pass", "Refusal card " + ("present" if refusal else "absent (answered)"))

    badges = await page.query_selector_all(".meta-badges .badge")
    if badges:
        badge_texts = [await b.inner_text() for b in badges]
        log("pass", f"Meta badges: {badge_texts}")
    else:
        log("warn", "No meta badges")

    sources_used = await page.query_selector(".evidence-sources-used")
    if sources_used:
        source_cards = await page.query_selector_all(".source-used-card")
        log("pass", f"Sources Used: {len(source_cards)} cards")
        for card in source_cards:
            name_el = await card.query_selector(".source-used-name")
            if name_el:
                print(f"    - Source: {await name_el.inner_text()}")
    else:
        log("warn", "No Sources Used section")

    citations = await page.query_selector_all(".citation-ref")
    log("pass" if len(citations) > 0 else "warn", f"Citation refs: {len(citations)}")

    trust = await page.query_selector(".trust-section")
    if trust:
        v = await page.query_selector(".trust-section .verification-status")
        c = await page.query_selector(".trust-section .confidence-section")
        s = await page.query_selector(".trust-section .safety-notice")
        log("pass" if v else "warn", f"  Verification: {'present' if v else 'missing'}")
        log("pass" if c else "warn", f"  Confidence: {'present' if c else 'missing'}")
        log("pass" if s else "warn", f"  Safety: {'present' if s else 'missing'}")
    else:
        log("warn", "No trust section")

    advanced = await page.query_selector("details.advanced-details")
    if advanced:
        summary = await page.query_selector("details.advanced-details summary")
        if summary:
            await summary.click()
            await page.wait_for_timeout(300)
            expanded = await advanced.evaluate("el => el.hasAttribute('open')")
            log("pass" if expanded else "fail", f"Advanced expanded: {expanded}")
            body = await page.query_selector(".advanced-body")
            if body:
                ep = await page.query_selector(".advanced-body .evidence-panel")
                tr = await page.query_selector(".advanced-body .answer-trace")
                log("pass" if ep else "warn", f"  Evidence panel: {'present' if ep else 'missing'}")
                log("pass" if tr else "warn", f"  Answer trace: {'present' if tr else 'missing'}")
    else:
        log("warn", "No advanced details")

    why_btn = await page.query_selector(".btn-why")
    log("pass" if why_btn else "warn", "'Why' button " + ("present" if why_btn else "missing"))

    timings = await page.query_selector(".timing-info")
    if timings:
        log("pass", f"Timings: {await timings.inner_text()}")
    else:
        log("warn", "No timings")

    chain = await page.query_selector(".timing-chain")
    log("pass" if chain else "info", f"Provider chain: {await chain.inner_text() if chain else 'none'}")

    await screenshot(page, "03_chat_response")

    if advanced and summary:
        await summary.click()
        await page.wait_for_timeout(200)

    await check_overflow(page, "after response")
    return True

async def test_drawer_open(page):
    print(f"\n{'='*60}")
    print("TEST: Drawer Open State")
    print(f"{'='*60}")

    citation = await page.query_selector(".citation-ref")
    if citation:
        await citation.click()
        await page.wait_for_timeout(500)

    drawer = await page.query_selector("#evidence-drawer")
    is_open = await drawer.evaluate("el => el.classList.contains('open')")
    log("pass" if is_open else "warn", f"Drawer open: {is_open}")

    if is_open:
        body = await page.query_selector("#drawer-body")
        body_html = await body.inner_html()
        has_content = len(body_html) > 50 and "drawer-empty" not in body_html
        log("pass" if has_content else "warn", f"Drawer has evidence content: {has_content}")

        hdr = await page.query_selector(".drawer-header h3")
        if hdr:
            log("pass", f"Drawer header: '{await hdr.inner_text()}'")

        await screenshot(page, "04_drawer_open")
        await check_overflow(page, "drawer open")

        close_btn = await page.query_selector("#drawer-close")
        if close_btn:
            await close_btn.click()
            await page.wait_for_timeout(300)
            is_open_after = await drawer.evaluate("el => el.classList.contains('open')")
            log("pass" if not is_open_after else "fail", f"Drawer closed: {not is_open_after}")

        await screenshot(page, "04_drawer_closed_after")
    else:
        log("info", "Drawer did not open via citation")

async def test_refusals(page):
    print(f"\n{'='*60}")
    print("TEST: Refusal States")
    print(f"{'='*60}")

    refusals = [
        ("What is the capital of France?", "low_relevance"),
        ("How much metformin should I take?", "medical_advice"),
    ]
    for question, expected in refusals:
        print(f"\n  --- {question} ---")
        new_btn = await page.query_selector("#new-chat-btn")
        if new_btn:
            await new_btn.click()
            await page.wait_for_timeout(300)

        input_el = await page.query_selector("#user-input")
        await input_el.click()
        await input_el.fill(question)
        await page.keyboard.press("Enter")

        try:
            await page.wait_for_selector(".message.assistant .answer-card", timeout=90000)
            await page.wait_for_timeout(1000)
        except Exception as e:
            log("fail", f"Timeout: {e}")
            continue

        refusal = await page.query_selector(".refusal-card")
        if refusal:
            variant = await refusal.evaluate("el => el.className")
            log("pass", f"Refusal card: {variant}")

            title = await page.query_selector(".refusal-title")
            if title:
                log("pass", f"  Title: '{await title.inner_text()}'")
            body = await page.query_selector(".refusal-body")
            if body:
                bt = await body.inner_text()
                log("pass", f"  Body: '{bt[:80]}...'")
            sug = await page.query_selector(".refusal-suggestion")
            if sug:
                log("pass", f"  Suggestion present")
        else:
            log("fail", f"No refusal card for: {question}")

        badges = await page.query_selector_all(".meta-badges .badge")
        badge_texts = [await b.inner_text() for b in badges]
        log("pass" if badges else "warn", f"  Badges: {badge_texts}")

        name = question[:20].replace(" ", "_").replace("?", "")
        await screenshot(page, f"05_refusal_{name}")

async def test_drawer_keyboard(page):
    print(f"\n{'='*60}")
    print("TEST: Drawer Keyboard Access")
    print(f"{'='*60}")

    citation = await page.query_selector(".citation-ref")
    if citation:
        await citation.click()
        await page.wait_for_timeout(500)

    drawer = await page.query_selector("#evidence-drawer")
    is_open = await drawer.evaluate("el => el.classList.contains('open')")
    if is_open:
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(300)
        is_open_after = await drawer.evaluate("el => el.classList.contains('open')")
        log("pass" if not is_open_after else "fail", f"Escape closes drawer: {not is_open_after}")
    else:
        log("info", "Drawer not open, skipping Escape test")

async def test_recent_chats(page):
    print(f"\n{'='*60}")
    print("TEST: Recent Chats")
    print(f"{'='*60}")

    clear_all = await page.query_selector("#clear-all-chats-btn")
    if clear_all:
        page.on("dialog", lambda d: d.accept())
        await clear_all.click()
        await page.wait_for_timeout(500)

    recent = await page.query_selector("#recent-chats")
    if recent:
        text = await recent.inner_text()
        log("pass" if "No conversations" in text else "warn", f"Empty state: '{text.strip()[:50]}'")

    log("pass" if await page.query_selector("#new-chat-btn") else "fail", "New chat button exists")
    log("pass" if await page.query_selector("#clear-all-chats-btn") else "fail", "Clear all button exists")

async def test_sidebar_sources(page):
    print(f"\n{'='*60}")
    print("TEST: Sidebar Sources")
    print(f"{'='*60}")

    sources = await page.query_selector("#sidebar-sources")
    if sources:
        text = await sources.inner_text()
        log("pass", f"Sources: '{text.strip()[:60]}'")
    else:
        log("fail", "Sidebar sources missing")

async def main():
    print("=" * 60)
    print("CLINICAL EVIDENCE COPILOT - VISUAL QA")
    print("=" * 60)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        for vp_name, vp in VIEWPORTS.items():
            context = await browser.new_context(viewport=vp, device_scale_factor=1)
            page = await context.new_page()

            page.on("console", lambda msg: RESULTS["console_errors"].append(msg.text) if msg.type == "error" else None)

            await test_landing(page, vp_name)

            if vp_name == "desktop_1440":
                await test_drawer_closed(page)
                success = await test_chat(page)
                if success:
                    await test_drawer_open(page)
                    await test_drawer_keyboard(page)
                    await test_refusals(page)
                await test_recent_chats(page)
                await test_sidebar_sources(page)

            await context.close()

        await browser.close()

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"PASS: {RESULTS['pass']}")
    print(f"FAIL: {RESULTS['fail']}")
    print(f"WARN: {RESULTS['warn']}")
    print(f"INFO: {RESULTS['info']}")
    print(f"\nConsole errors: {len(RESULTS['console_errors'])}")
    for e in RESULTS["console_errors"]:
        print(f"  {e}")
    print(f"\nIssues: {len(RESULTS['issues'])}")
    for i in RESULTS["issues"]:
        print(f"  {i}")
    print(f"\nScreenshots: {len(RESULTS['screenshots'])}")
    for s in RESULTS["screenshots"]:
        print(f"  {s}")

if __name__ == "__main__":
    asyncio.run(main())