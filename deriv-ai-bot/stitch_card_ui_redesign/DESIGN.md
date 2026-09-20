---
name: Quant Terminal Precision
colors:
  surface: '#0f131c'
  surface-dim: '#0f131c'
  surface-bright: '#353943'
  surface-container-lowest: '#0a0e17'
  surface-container-low: '#181b25'
  surface-container: '#1c1f29'
  surface-container-high: '#262a34'
  surface-container-highest: '#31353f'
  on-surface: '#dfe2ef'
  on-surface-variant: '#c7c4d7'
  inverse-surface: '#dfe2ef'
  inverse-on-surface: '#2c303a'
  outline: '#908fa0'
  outline-variant: '#464554'
  surface-tint: '#c0c1ff'
  primary: '#c0c1ff'
  on-primary: '#1000a9'
  primary-container: '#8083ff'
  on-primary-container: '#0d0096'
  inverse-primary: '#494bd6'
  secondary: '#4cd7f6'
  on-secondary: '#003640'
  secondary-container: '#03b5d3'
  on-secondary-container: '#00424e'
  tertiary: '#4edea3'
  on-tertiary: '#003824'
  tertiary-container: '#00885d'
  on-tertiary-container: '#000703'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#e1e0ff'
  primary-fixed-dim: '#c0c1ff'
  on-primary-fixed: '#07006c'
  on-primary-fixed-variant: '#2f2ebe'
  secondary-fixed: '#acedff'
  secondary-fixed-dim: '#4cd7f6'
  on-secondary-fixed: '#001f26'
  on-secondary-fixed-variant: '#004e5c'
  tertiary-fixed: '#6ffbbe'
  tertiary-fixed-dim: '#4edea3'
  on-tertiary-fixed: '#002113'
  on-tertiary-fixed-variant: '#005236'
  background: '#0f131c'
  on-background: '#dfe2ef'
  surface-variant: '#31353f'
  profit-emerald: '#10B981'
  profit-emerald-muted: rgba(16, 185, 129, 0.12)
  loss-rose: '#F43F5E'
  loss-rose-muted: rgba(244, 63, 94, 0.12)
  warning-amber: '#F59E0B'
  warning-amber-muted: rgba(245, 158, 11, 0.12)
  agent-cyan: '#06B6D4'
  agent-indigo: '#6366F1'
  surface-base: '#090D16'
  surface-card: '#111827'
  surface-elevated: '#161F30'
  surface-overlay: '#1E293B'
  border-subtle: '#1F293D'
  border-strong: '#2D3B55'
  text-primary: '#F8FAFC'
  text-secondary: '#94A3B8'
  text-muted: '#64748B'
typography:
  headline-lg:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '700'
    lineHeight: 32px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '600'
    lineHeight: 20px
    letterSpacing: -0.005em
  body-md:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  label-lg:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.02em
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.04em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '500'
    lineHeight: 12px
    letterSpacing: 0.05em
  data-tabular-lg:
    fontFamily: JetBrains Mono
    fontSize: 16px
    fontWeight: '700'
    lineHeight: 20px
    letterSpacing: -0.01em
  data-tabular-md:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.00em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 0.75rem
  margin: 1rem
  space-xs: 0.25rem
  space-sm: 0.375rem
  space-md: 0.75rem
  space-lg: 1rem
  space-xl: 1.5rem
---

## Brand & Style

This design system embodies high-density quantitative intelligence: mission-critical, cold-blooded, mathematically rigorous, and hyper-responsive. Designed for institutional quantitative traders, autonomous multi-agent monitoring, and high-frequency risk surveillance, the aesthetic blends the ruthless utility of a Bloomberg Terminal with the refined ergonomics of Linear and the reactive feedback of TradingView.

The design movement is **Technical Minimalism with Micro-Border Precision**. The interface prioritizes tabular throughput, monospaced numerical clarity, and instant visual triage over gratuitous ornamentation. Visual hierarchies are driven by crisp 1px borders, subtle tonal stratification, and high-contrast semantic indicators that let anomalous algorithmic behavior, trade outcomes, and hard veto states register instantaneously across peripheral vision.

## Colors

The palette operates under an uncompromising dark mode canvas built on dark slate and zinc foundations.

- **Primary (`#6366F1`) & Secondary (`#06B6D4`)**: Reserved for machine cognition, agent identity badges, consensus status, and telemetry highlights.
- **Profit Emerald (`#10B981`)**: Signals execution success, winning win-rates, active runs, and affirmative risk checks. Background tints are strictly capped at 12% opacity.
- **Loss Rose (`#F43F5E`)**: Communicates veto interventions, circuit breakers, drawdown warnings, and loss-side trade results. Demands immediate focus without visual vibration.
- **Warning Amber (`#F59E0B`)**: Designates cooldown modes, calibration drift, and borderline confidence scores.
- **Surfaces**: Three strict tiers—`surface-base` (`#090D16`) for root layouts, `surface-card` (`#111827`) for standard modular data cells, and `surface-elevated` (`#161F30`) for hovered or active panels.

## Typography

Typography enforces a strict bifurcated system between structural interface labels and financial telemetry:

1. **System Interface (`Inter`)**: Used for titles, narrative logs, tooltips, and high-level section names. Clean, neutral, and devoid of distracting quirks.
2. **Computational Data (`JetBrains Mono`)**: Used exclusively for monetary figures, ticker symbols (e.g., `R_100`, `BOOM500`), percentages, execution speeds, and timestamps. Always rendered with `font-variant-numeric: tabular-nums` to guarantee aligned decimal points across rapid data updates.
3. All micro status tags (`HARD VETO`, `ACTIVE SCANNING`, `CALL`, `PUT`) must use uppercase sizing with tracked letter spacing (`letterSpacing: 0.04em` to `0.05em`).

## Layout & Spacing

The layout employs a high-density, multi-panel institutional console grid designed to pack maximal actionable context above the fold.

- **Grid Architecture**: 12-column fluid grid system with compact 12px (`0.75rem`) gutters and 16px (`1rem`) viewport margins.
- **Breakpoints**:
  - `Desktop (>= 1440px)`: Full 4-pane workstation view (Left: Portfolio/Risk Sentinel, Center: Live Charts & Probability Engines, Right: Agent Fleet Leaderboard, Bottom: Consolidated Telemetry Feed).
  - `Laptop (1024px - 1439px)`: 2-column stacked layout; telemetry docks collapse to tabbed panels.
  - `Mobile (< 1024px)`: Single column feed, pinned persistent kill-switch/veto bar at the top, scrollable sub-tables.
- **Density Controls**: Vertical line items inside tables use a fixed 28px height; dense tables use a 24px height with `space-xs` inline padding.

## Elevation & Depth

Visual depth is communicated exclusively through **Low-Contrast Outlines** and subtle **Tonal Layers**, strictly avoiding heavy blur shadows that cause visual mud on dense multi-monitor setups.

- **Layer 0 (Canvas Base)**: `#090D16` flat background.
- **Layer 1 (Card/Container Panels)**: `#111827` with a 1px continuous perimeter border of `border-subtle` (`#1F293D`).
- **Layer 2 (Elevated Overlays / Hover States)**: `#161F30` with `border-strong` (`#2D3B55`).
- **Accent Elevation (Alert / Focus)**: A single inner-glow hairline border using the semantic color at 30% alpha (e.g., `box-shadow: inset 0 0 0 1px rgba(244, 63, 94, 0.35)` for veto interventions).

## Shapes

The interface embraces an engineered, razor-sharp visual language:

- **Global Roundedness**: Soft (`0.25rem` / `4px`). This maintains maximum usable pixel real-estate within tight data tables and charts.
- **Micro-Badges & Telemetry Pills**: Small items use `2px` or `4px` maximum corner radius. Avoid circular pill containers (`rounded-full`) except for live ping/pulse status indicators (`●`).
- **Dividers & Table Cells**: Crisp `1px` lines utilizing `#1F293D`.

## Components

### Buttons & Operational Controls
- **System Actions**: Minimum height 30px, `space-sm` vertical, `space-md` horizontal padding. Font: `label-md`.
- **Emergency Halt / Hard Veto**: High-contrast outline with `#F43F5E` background, pure white label, accompanied by a subtle red border glow.
- **Control Strip**: Clustered segmented button group (`Resume`, `Pause`, `Restart`) with `#111827` background and 1px borders between segments.

### Status Indicators & Badges
- **Status Dot**: 6px geometric circle with an optional CSS ping-animation for real-time polling (`Auto-refresh 15s`).
- **Agent Roles**: Monospaced tags with a subtle tinted background (e.g., Indigo for `CEO`, Cyan for `Specialist`, Rose for `Guardian`), rendered at `label-sm`.

### Data Tables & Registries
- **Row Styling**: Alternating transparent and 2% white hover layers. Border bottom 1px solid `#1F293D`.
- **Cell Alignment**: Text left-aligned; numerical values, percentages, and execution latency strictly right-aligned using `data-tabular-md`.

### Micro Telemetry & Progress Bars
- Compact 4px height track with `#1E293B` background. Fill track maps dynamically to status colors (Emerald for win-rates > 65%, Rose for < 50%, Cyan for AI advisor confidence).

### Cards & Modular Containers
- Constructed from `surface-card` with an explicit card header (height 36px), title in `headline-sm`, right-aligned monospaced parameter count or latency indicator, and a bottom 1px separator line.