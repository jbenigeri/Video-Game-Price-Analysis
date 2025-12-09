### Correlation Heatmap Analysis

#### Strong Correlations (|r| > 0.5)

**1. Price (€) <-> Original Price (€): 0.651**
- **Makes Perfect Sense**: Current prices closely track original launch prices
- Even when games go on sale, they maintain relative positioning (€60 AAA games -> €30-40 on sale, €10 indies -> €5-7 on sale)
- Price tiers are sticky: premium games stay premium, budget games stay budget

**2. Linux <-> MacOS: 0.625**
- **Very Insightful**: Multi-platform support comes in bundles
- If a developer supports Linux, they almost always support MacOS too (and vice versa)
- **Why?** 
  - Both are Unix-based systems -> easier simultaneous porting
  - Cross-platform engines (Unity, Unreal) export to both at once
  - Indie devs who care about accessibility support both non-Windows platforms
  - It's an "all or nothing" decision: either support Windows-only (AAA), or support all three

---

#### Moderate Correlations (0.3 < |r| ≤ 0.5)

**3. Discount% <-> Price (€): 0.465**
- **Business Strategy Insight**: Higher-priced games have steeper discounts
- €60 game at 70% off -> €18 (still profitable)
- €5 indie at 70% off -> €1.50 (not sustainable)
- **Psychological pricing**: "Save €42!" is more compelling than "Save €2!"
- Premium games use deep discounts during Steam sales to drive volume

---

#### Interesting Weak Patterns

**4. Rating <-> Price: -0.111 (slight negative)**
- **Surprising**: Cheaper games are rated slightly HIGHER!
- Possible explanations:
  - Indie games = more innovation, fewer bugs, more charm
  - Lower expectations -> pleasant surprises -> better ratings
  - AAA games face more scrutiny and criticism
  - Price/value perception affects ratings ("great for €5!" vs "disappointing for €60")

**5. Original Price <-> Discount%: -0.256 (negative)**
- More expensive games have SMALLER discount percentages
- Could reflect different sales strategies:
  - Indie games frequently on deep sale
  - AAA games protect brand value with smaller discounts
  - Premium pricing = less need to discount aggressively

**6. Price/Original Price <-> Linux/MacOS: negative correlations (-0.15 to -0.22)**
- **Clear Market Segmentation**:
  - **AAA games (expensive)**: Windows-only, maximize largest market
  - **Indie games (cheap)**: Cross-platform engines, broader accessibility
- Porting AAA games to Mac/Linux is expensive with limited ROI

---

#### Surprisingly Weak/No Correlation

**7. #Reviews <-> Current Price: 0.006 (zero!)**
- Review count is independent of current price
- Reviews accumulate over a game's lifetime regardless of sales
- A game launched at €60, now €10, still has reviews from original €60 buyers

**8. Rating <-> Platform Support: ~0 correlation**
- Game quality has nothing to do with how many platforms it's on
- Cross-platform = business decision, not quality indicator

**9. #Reviews <-> Discount%: -0.151 (weak negative)**
- Popular games (more reviews) have slightly smaller discounts
- Bestsellers don't need deep discounts to move units

---

#### Key Takeaways

1. **Market Segmentation is Real**: Premium (Windows-only, high price) vs Indie (multi-platform, low price) are distinct categories
2. **Pricing is Sticky**: Games maintain their price tier identity even through sales
3. **Platform Support Bundles**: Linux + MacOS go together (indie strategy) or neither (AAA strategy)
4. **Quality ≠ Price**: Cheaper indie games often deliver better value/ratings than expensive AAA titles
5. **Discounting Strategy**: Premium games can afford (and use) deeper percentage discounts