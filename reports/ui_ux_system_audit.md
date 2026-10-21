# Comprehensive UI/UX & System Design Audit (30-Point Framework)

Project: Counterfactual Prerequisite Impact Modeling for Personalized Learning Path Generation (CFLP)

| # | Consideration | Decision | Justification & Architectural Strategy |
| :---: | :--- | :---: | :--- |
| 1 | Harsh gradients | **AVOID** | Academic research dashboards must prioritize data legibility, scientific clarity, and clean neutral palettes over high-saturation decorative gradients. |
| 2 | Lucide icons | **ADAPT** | Use clear, semantic SVG icons (status, arrows, graph nodes) with accessible ARIA labels for intuitive navigation without extraneous decoration. |
| 3 | Pure white backgrounds | **AVOID** | Use calibrated neutral backgrounds (`#f8fafc` in light mode, `#0f172a` in dark mode) to reduce visual fatigue during analytical workflows. |
| 4 | Rainbow coloring | **AVOID** | Restrict palette to an intentional educational/research token system: slate neutrals, focused indigo accent (`#4f46e5`), subtle emerald positive delta (`#059669`), and amber cautionary/prerequisite indicators. |
| 5 | Drop shadows | **ADAPT** | Use restrained, diffused micro-shadows (`0 1px 3px rgba(0,0,0,0.06)`) purely for card elevation and visual hierarchy, avoiding deep blurry skeuomorphic drops. |
| 6 | Three feature cards per row | **ADAPT** | Use a flexible responsive CSS grid adapting dynamically from 1 column on mobile to 2 or 4 columns based on screen width and data density. |
| 7 | Emojis | **AVOID** | Maintain academic and engineering rigor. Replace whimsical emojis with precise semantic badges, status pills, and SVG icons. |
| 8 | Liquid-glass effects | **AVOID** | Avoid GPU-heavy backdrop filters or blurry frosted glass that hinder readability of numbers and scientific equations. |
| 9 | Correct em dash usage | **APPLY** | Use proper grammatical punctuation: em dash (—) with appropriate spacing for parenthetical thought, en dash (–) for ranges. |
| 10 | Inter, Geist, or Space Grotesk typography | **APPLY** | Implement clean modern typography via system-fallback font stack prioritizing Inter / -apple-system / Segoe UI with tabular figures (`font-variant-numeric: tabular-nums`) for numerical alignment. |
| 11 | Colored left stripes | **AVOID** | Avoid decorative left borders that clutter dense analytical tables; use subtle badges and contextual border highlights. |
| 12 | Fake testimonials | **AVOID** | Zero fabrication. Display exclusively genuine model predictions, verified dataset metrics, and authentic XES3G5M student sequence data. |
| 13 | Bento grids | **ADAPT** | Organize research workflow modules (Dataset, Student State, Target Selector, Impact Ranking, Learning Path) into a structured modular grid with purposeful layout hierarchy. |
| 14 | Terminal-style windows | **AVOID** | Avoid gimmicky decorative fake-terminal windows; render actual logs and outputs in cleanly styled code blocks. |
| 15 | Repetitive "It's not X, it's Y" copy | **AVOID** | Use direct, objective technical exposition explaining research formulation, causal caveats, and model mechanics. |
| 16 | Checkmark bullets | **AVOID** | Use clean typographic hierarchy, numbered ordered lists for pedagogical step progression, and standard definition lists. |
| 17 | Three pricing tiers | **AVOID** | Not a commercial SaaS; this is an academic research prototype. Exclude any mock commercial pricing structures. |
| 18 | Absence of real product demos | **AVOID** | Provide working interactive demonstrations executing real model inference against the live FastAPI backend on real student test data. |
| 19 | Soft corner radius | **APPLY** | Maintain consistent, professional design tokens (`--radius-sm: 4px`, `--radius-md: 6px`, `--radius-lg: 8px`), avoiding exaggerated pill shapes. |
| 20 | Purple-and-black palettes | **AVOID** | Steer clear of cliché AI "cyberpunk" purple-black palettes. Use an academic slate, indigo, and neutral research palette. |
| 21 | Absence of skeleton loaders | **AVOID** | Implement animated skeleton placeholders for asynchronous model inference (target prediction, impact ranking, path generation) to ensure clear UX feedback. |
| 22 | Radial gradients and orbs | **AVOID** | Avoid distracting floating ambient orbs or glowing decorative blobs. |
| 23 | Dot grids | **AVOID** | Avoid decorative background noise patterns that reduce visual contrast. |
| 24 | Sparkle icons | **AVOID** | Do not use AI sparkle icons (`✨`) that imply magical ungrounded claims. |
| 25 | Animated arrows | **AVOID** | Use crisp, static directional indicators representing graph dependency and prerequisite order. |
| 26 | Absence of Terms of Service | **ADAPT** | Include explicit Research Use Guidelines and Open-Source academic license terms. |
| 27 | Absence of a Privacy Policy | **ADAPT** | Include an explicit Data Governance & Educational Privacy Policy detailing that no PII is retained and student interactions are de-identified. |
| 28 | Hover animations | **ADAPT** | Use subtle, accessible micro-transitions (`transition: transform 0.15s ease, box-shadow 0.15s ease`) with `prefers-reduced-motion` compliance. |
| 29 | Neon colors | **AVOID** | Use WCAG AA-compliant accessible color contrast ratios for all states. |
| 30 | Basic pastel colors | **AVOID** | Avoid low-contrast washed-out pastels; use distinct, accessible palette tokens. |
