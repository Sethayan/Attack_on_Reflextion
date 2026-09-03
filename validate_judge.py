"""
Validate Judge — run the LLM judge on synthetic samples to check accuracy.

Contains 12 synthetic trip-plan outputs (mix of good and bad) with known
expected verdicts. Runs each through the LLM judge and prints a side-by-side
table for manual human review.

Usage:
    python3 validate_judge.py
"""

from evaluator import run_llm_judge

# ──────────────────────────────────────────────────────────────────────
#  Synthetic test samples
# ──────────────────────────────────────────────────────────────────────

SAMPLES = [
    # ── GOOD PLANS (all checks should pass) ──────────────────────────
    {
        "id": 1,
        "label": "good_basic",
        "task_input": {
            "origin": "Mumbai",
            "cities": "Paris, Rome, Barcelona",
            "date_range": "June 1-7, 2026",
            "interests": "art, food, history",
        },
        "output": """# 7-Day Paris Trip Plan

**Day 1 (June 1):** Arrive at Charles de Gaulle Airport. Check into Hotel Le Marais
in the 4th arrondissement. Evening walk along the Seine. Dinner at Le Comptoir
du Panthéon — classic French bistro cuisine.

**Day 2 (June 2):** Morning visit to the Louvre Museum (allocate 4 hours).
Lunch at Café Marly overlooking the pyramid. Afternoon at Musée d'Orsay
for Impressionist art. Evening: stroll through Saint-Germain-des-Prés.

**Day 3 (June 3):** Day trip to Versailles. Tour the palace and gardens
(full day). Return to Paris for dinner at Le Bouillon Chartier.

**Day 4 (June 4):** Morning at Notre-Dame area and Île de la Cité.
Lunch at L'As du Fallafel in Le Marais. Afternoon: Centre Pompidou
for modern art. Evening: Montmartre and Sacré-Cœur.

**Day 5 (June 5):** Visit the Eiffel Tower (book skip-the-line tickets).
Picnic lunch at Champ de Mars. Afternoon: Musée Rodin and Les Invalides.
Evening: Seine river cruise.

**Day 6 (June 6):** Morning at Père Lachaise Cemetery. Lunch in Belleville.
Afternoon: free time for shopping at Galeries Lafayette.
Evening: farewell dinner at Le Train Bleu at Gare de Lyon.

**Day 7 (June 7):** Morning checkout. Last croissant at a local bakery.
Transfer to CDG for departure.

**Budget Breakdown:**
- Flights: $650
- Hotel (6 nights): $900
- Food: $350
- Activities: $200
- Transport: $100
**Total Estimated Cost: $2,200**
""",
    },
    {
        "id": 2,
        "label": "good_tokyo",
        "task_input": {
            "origin": "New York",
            "cities": "Tokyo, Seoul, Bangkok",
            "date_range": "March 10-17, 2026",
            "interests": "technology, street food",
        },
        "output": """# 8-Day Tokyo Adventure

**Day 1 (March 10):** Arrive at Narita Airport. Take Narita Express to
Shinjuku. Check into Hotel Gracery Shinjuku. Evening: explore Kabukicho
and try ramen at Fuunji.

**Day 2 (March 11):** Morning: Akihabara for electronics and anime culture.
Lunch: street food at Ameyoko Market near Ueno. Afternoon: teamLab Borderless
digital art museum. Evening: Shibuya Crossing and dinner at Ichiran Ramen.

**Day 3 (March 12):** Day trip to Odaiba. Visit Miraikan (National Museum
of Emerging Science). Lunch at DiverCity Tokyo. Afternoon: Toyota Mega Web.
Evening: Rainbow Bridge views.

**Day 4 (March 13):** Tsukiji Outer Market for breakfast. Meiji Shrine and
Harajuku. Takeshita Street for crepes. Afternoon: Omotesando for architecture.
Evening: Roppongi nightlife.

**Day 5 (March 14):** Day trip to Hakone. See Mount Fuji views. Hot springs
experience. Return to Tokyo by evening.

**Day 6 (March 15):** Asakusa and Senso-ji Temple. Lunch: street yakitori.
Afternoon: Sumida River cruise. Evening: Tokyo Skytree observation deck.

**Day 7 (March 16):** Nakameguro area. Afternoon: Shimokitazawa vintage
shopping. Evening: farewell dinner at Gonpachi (the "Kill Bill" restaurant).

**Day 8 (March 17):** Morning checkout and departure from Narita.

**Estimated Budget: $3,500**
""",
    },
    {
        "id": 3,
        "label": "good_short_trip",
        "task_input": {
            "origin": "London",
            "cities": "Lisbon, Athens, Istanbul",
            "date_range": "September 1-5, 2026",
            "interests": "beaches, nightlife",
        },
        "output": """# 5-Day Lisbon Beach & Nightlife Guide

**Day 1 (Sep 1):** Arrive at Lisbon Humberto Delgado Airport. Check into
The Lumiares Hotel in Bairro Alto. Afternoon: Praia de Carcavelos beach.
Evening: Bar crawl through Bairro Alto — start at Pensão Amor, then
Pavilhão Chinês.

**Day 2 (Sep 2):** Morning surf lesson at Costa da Caparica. Beach day.
Lunch: grilled fish at Ponto Final across the river. Evening: LX Factory
for drinks and live music at Bravo.

**Day 3 (Sep 3):** Day trip to Cascais and Guincho Beach. Lunch: seafood
at Casa da Guia. Afternoon: explore Cascais old town. Evening: Dock's Club
in Santos for nightlife.

**Day 4 (Sep 4):** Morning: Praia da Ursa (hidden beach, hike required).
Afternoon: Belém — Pastéis de Belém for custard tarts. Sunset drinks at
PARK bar (rooftop). Night: Lux Frágil, Lisbon's most famous club.

**Day 5 (Sep 5):** Morning checkout. Last pastéis de nata. Departure.

**Estimated Budget: $1,800**
""",
    },
    {
        "id": 4,
        "label": "good_europe_cluster",
        "task_input": {
            "origin": "Berlin",
            "cities": "Amsterdam, Prague, Vienna",
            "date_range": "April 15-20, 2026",
            "interests": "museums, beer, architecture",
        },
        "output": """# 6-Day Prague Cultural Itinerary

**Day 1 (April 15):** Arrive at Václav Havel Airport Prague. Check into
Hotel Josef in Old Town. Afternoon: walk across Charles Bridge.
Evening: traditional Czech dinner at Lokál Dlouhááá. Try Pilsner Urquell.

**Day 2 (April 16):** Morning: Prague Castle complex (St. Vitus Cathedral,
Old Royal Palace, Golden Lane). Lunch at Klášterní pivovar Strahov
(monastery brewery). Afternoon: Strahov Library.

**Day 3 (April 17):** Jewish Quarter — Old Jewish Cemetery, Spanish Synagogue.
Lunch: Café Louvre. Afternoon: National Gallery at Trade Fair Palace.
Evening: craft beer tasting at BeerGeek Bar.

**Day 4 (April 18):** Day trip to Kutná Hora (Sedlec Ossuary, Italian Court).
Return to Prague. Evening: rooftop dinner at Terasa U Zlaté studně.

**Day 5 (April 19):** Morning: Petřín Hill and observation tower.
Lunch at Café Savoy. Afternoon: Museum of Decorative Arts and Cubist
architecture at House of the Black Madonna. Evening: U Fleků brewery
(founded 1499).

**Day 6 (April 20):** Morning checkout. Last trdelník at Old Town Square.
Departure.

**Budget: $1,600**
""",
    },

    # ── BAD PLANS (should trigger failures) ──────────────────────────
    {
        "id": 5,
        "label": "bad_wrong_city",
        "task_input": {
            "origin": "Mumbai",
            "cities": "Paris, Rome, Barcelona",
            "date_range": "June 1-7, 2026",
            "interests": "art, food, history",
        },
        "output": """# 7-Day Tokyo Trip Plan

Day 1: Arrive in Tokyo. Visit Shibuya Crossing.
Day 2: Akihabara electronics district.
Day 3: Tsukiji fish market.
Day 4: Day trip to Mount Fuji.
Day 5: Harajuku and Meiji Shrine.
Day 6: Odaiba and teamLab.
Day 7: Departure from Narita.
""",
        # Should fail: constraints_ok (wrong city — Tokyo not in options)
    },
    {
        "id": 6,
        "label": "bad_impossible_travel",
        "task_input": {
            "origin": "London",
            "cities": "Lisbon, Athens, Istanbul",
            "date_range": "September 1-5, 2026",
            "interests": "beaches, nightlife",
        },
        "output": """# 5-Day Multi-City Sprint

Day 1: Morning flight London to Lisbon. Afternoon beach at Cascais.
Evening fly to Athens.
Day 2: Morning at the Acropolis. Lunch in Plaka. Afternoon flight to Istanbul.
Evening: Grand Bazaar shopping.
Day 3: Morning flight Istanbul to Lisbon. Beach at Carcavelos.
Afternoon flight back to Athens for nightlife.
Day 4: Morning in Athens. Fly to Istanbul for lunch.
Fly to Lisbon for dinner.
Day 5: Fly back to London from wherever you ended up.
""",
        # Should fail: feasible (multiple flights per day, impossible logistics)
    },
    {
        "id": 7,
        "label": "bad_hallucinated_venues",
        "task_input": {
            "origin": "Berlin",
            "cities": "Amsterdam, Prague, Vienna",
            "date_range": "April 15-20, 2026",
            "interests": "museums, beer, architecture",
        },
        "output": """# 6-Day Prague Plan

Day 1: Visit the Grand Zeppelin Museum of Prague (the largest aviation
museum in Central Europe). Lunch at McBrothsky's Traditional Czech Tavern.
Day 2: Tour the Prague Royal Pyramid (built in 1342 by Emperor Zoltan IV).
Afternoon: The Quantum Beer Experience at FizzTron Brewery.
Day 3: Visit Café Nebulosa for floating pancakes. Tour the Crystal Palace
of Wenceslas (underground glass castle).
Day 4: Day trip to the Enchanted Forest of Moravia theme park.
Day 5: Morning at the Blockchain Museum of Prague. Lunch at Rover's
Autonomous Kitchen (robot-served restaurant).
Day 6: Departure.
""",
        # Should fail: no_hallucination (fabricated venues)
    },
    {
        "id": 8,
        "label": "bad_missing_days",
        "task_input": {
            "origin": "New York",
            "cities": "Tokyo, Seoul, Bangkok",
            "date_range": "March 10-17, 2026",
            "interests": "technology, street food",
        },
        "output": """# Tokyo Trip

Day 1: Arrive in Tokyo.
Day 2: Visit Akihabara.
Day 5: Visit Shibuya.
Day 8: Departure.

Budget: $2,000
""",
        # Should fail: days_ok (missing days 3, 4, 6, 7; only 4 of 8)
    },
    {
        "id": 9,
        "label": "bad_duplicate_venues",
        "task_input": {
            "origin": "Mumbai",
            "cities": "Paris, Rome, Barcelona",
            "date_range": "June 1-7, 2026",
            "interests": "art, food, history",
        },
        "output": """# 7-Day Paris Plan

Day 1: Visit the **Louvre Museum**. Lunch at **Café Marly**. Evening at
**Eiffel Tower**.
Day 2: Morning at **Musée d'Orsay**. Afternoon: **Louvre Museum** again for
the Egyptian wing. Dinner at **Le Comptoir**.
Day 3: **Eiffel Tower** sunrise visit. Then **Versailles** day trip.
Day 4: **Notre-Dame** and **Louvre Museum** for sculpture hall.
Day 5: **Montmartre** and **Sacré-Cœur**. Evening: **Eiffel Tower** lights.
Day 6: **Père Lachaise** and **Café Marly** for lunch again.
Day 7: Departure.
""",
        # Should fail: no_duplicates (Louvre Museum x3, Eiffel Tower x3, Café Marly x2)
    },
    {
        "id": 10,
        "label": "bad_constraint_violation",
        "task_input": {
            "origin": "Sydney",
            "cities": "Queenstown, Bali, Fiji",
            "date_range": "December 20-28, 2026",
            "interests": "adventure, nature",
            "constraints": "vegetarian diet only, no extreme heights (fear of heights)",
        },
        "output": """# 9-Day Queenstown Adventure

Day 1: Arrive in Queenstown. Check into Hilton.
Day 2: Morning: Bungee jumping at Kawarau Bridge (134m drop!).
Lunch: lamb BBQ at Flame Bar & Grill.
Day 3: Skydiving over Queenstown at 15,000 feet. Steak dinner at Botswana Butchery.
Day 4: Nevis Swing (highest swing in Australasia, 160m).
Lunch: venison burger at Fergburger.
Day 5: Paragliding off Coronet Peak. Seafood platter at The Boatshed.
Day 6: Canyon swing and zipline. BBQ ribs at Atlas Beer Cafe.
Day 7: Helicopter ride over Milford Sound. Roast duck dinner.
Day 8: Morning hike to Ben Lomond summit. Farewell lamb roast.
Day 9: Departure.
""",
        # Should fail: constraints_ok (fear of heights + vegetarian both violated)
    },
    {
        "id": 11,
        "label": "good_adventure",
        "task_input": {
            "origin": "Sydney",
            "cities": "Queenstown, Bali, Fiji",
            "date_range": "December 20-28, 2026",
            "interests": "adventure, nature",
        },
        "output": """# 9-Day Queenstown Nature & Adventure

Day 1 (Dec 20): Arrive at Queenstown Airport. Check into Sofitel Queenstown.
Afternoon: walk along Lake Wakatipu. Dinner at Rata restaurant.

Day 2 (Dec 21): Morning jet boat ride on Shotover River. Afternoon: hike
the Queenstown Hill Track. Evening: Fergburger for dinner.

Day 3 (Dec 22): Full-day Milford Sound cruise. Stunning fiord scenery,
waterfalls, and wildlife. Packed lunch included.

Day 4 (Dec 23): Morning: Routeburn Track day hike (first section).
Afternoon: Arrowtown historic gold mining village. Evening: Flame Bar & Grill.

Day 5 (Dec 24): Glenorchy day trip. Paradise Valley. Kayaking on Lake Wakatipu.
Christmas Eve dinner at The Grille by Eichardt's.

Day 6 (Dec 25): Christmas Day — relaxed morning. Afternoon: Skyline Gondola
and luge rides. Christmas dinner at Botswana Butchery.

Day 7 (Dec 26): Morning: Gibbston Valley winery tour and tasting.
Afternoon: AJ Hackett Bungy at Kawarau Bridge. Evening: The Boatshed.

Day 8 (Dec 27): Full-day Doubtful Sound expedition. More remote and
less crowded than Milford. Overnight cruise option.

Day 9 (Dec 28): Morning checkout. Last walk through Queenstown Gardens.
Departure.

**Estimated Budget: $4,200**
""",
    },
    {
        "id": 12,
        "label": "bad_mixed_issues",
        "task_input": {
            "origin": "Berlin",
            "cities": "Amsterdam, Prague, Vienna",
            "date_range": "April 15-20, 2026",
            "interests": "museums, beer, architecture",
        },
        "output": """# 6-Day Vienna Plan

Day 1: Arrive in Vienna. Visit St. Stephen's Cathedral.
Day 2: Schönbrunn Palace.
Day 3: Take a quick flight to Sydney to see the Opera House and then
fly back to Vienna for dinner.
Day 6: Visit the Museum of Invisible Art (admission: $500).
Departure.
""",
        # Should fail: feasible + no_hallucination + days_ok
    },
]


# ──────────────────────────────────────────────────────────────────────
#  Run validation
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 100)
    print("LLM JUDGE VALIDATION — Review each verdict and fill in human_verdict")
    print("=" * 100)
    print()

    header = f"{'#':>3} | {'Label':<25} | {'constraints':>11} | {'feasible':>8} | {'halluc.':>8} | {'human_verdict':>14} | {'agree?':>6}"
    print(header)
    print("-" * len(header))

    results = []
    for sample in SAMPLES:
        judge = run_llm_judge(sample["task_input"], sample["output"])

        c = judge.get("constraints_ok")
        f = judge.get("feasible")
        h = judge.get("no_hallucination")
        notes = judge.get("judge_notes", "")

        c_str = str(c) if c is not None else "null"
        f_str = str(f) if f is not None else "null"
        h_str = str(h) if h is not None else "null"

        row = f"{sample['id']:>3} | {sample['label']:<25} | {c_str:>11} | {f_str:>8} | {h_str:>8} | {'_____':>14} | {'_____':>6}"
        print(row)
        if notes:
            print(f"      notes: {notes}")

        results.append({
            "id": sample["id"],
            "label": sample["label"],
            "constraints_ok": c,
            "feasible": f,
            "no_hallucination": h,
            "notes": notes,
        })

    print()
    print("-" * len(header))
    print()
    print("OUTPUT SNIPPETS FOR REFERENCE:")
    print("-" * 80)
    for sample in SAMPLES:
        snippet = sample["output"].replace("\n", " ")[:120]
        print(f"  #{sample['id']:>2} ({sample['label']}): {snippet}...")
    print()
    print("INSTRUCTIONS:")
    print("  1. Review each judge verdict against the output snippet above")
    print("  2. Fill in 'human_verdict' column (pass/fail for each criterion)")
    print("  3. Mark 'agree?' as Y/N")
    print("  4. Count agreement rate. Target: ≥80% agreement.")
    print(f"  5. Total samples: {len(SAMPLES)}")
    print()


if __name__ == "__main__":
    main()
