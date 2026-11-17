"""Dip-buying cbBTC bot for the make.fun 24-hour BTC race (bars-replay/v1).

The idea, in one line: buy a real dip early, take a fixed recovery, and never
pay for chop. Everything below is measured on this competition's dev window
(1,440 one-minute cbBTC/USDC bars, 2026-11-17), where the one-minute noise is
about 6 bps (standard deviation), a full-size round trip costs about 34 bps
in fee plus slippage plus own price impact, and the whole window only moves
about 2.5% peak to trough. That arithmetic kills the usual candidates:

- An EMA(8)/EMA(21) cross bot loses about $107 on the dev window over 68
  trades: the round-trip cost is several times the typical one-minute move,
  so every cross is mostly whipsaw.
- A 60-bar breakout bot loses on every parameter setting tried: this window
  is range-bound, so buying a new high means buying the top of a wiggle.
- Buy-and-hold nets about +$1.07: the window ends only 0.49% above where it
  starts, and the entry cost eats most of it.

What works on this window is the dip: the window contains two drops of
1.2% or more below the trailing 60-bar high, and both recover. Buying the
dip itself -- the trigger is the low close, not a bounce after it -- gets
the entry in early, which is what the last race's final showed decides the
ranking between entries that take the same trade. The exit takes a fixed
recovery of the same size as the entry dip, with a hard stop and a wide
trailing stop behind it so a falling knife cannot bleed the account, and a
ten-hour cap so a position never rides the whole window by accident.

Parameters (all in basis points, all integer arithmetic, no floats anywhere
near money):

- LOOKBACK_BARS = 60: the trailing high the dip is measured against. A
  30-bar high adapts too fast -- it falls with the price, so the trigger
  never fires. 60 through 360 bars all fire on the same dev-window dip.
- DIP_BPS = 120: enter when the latest close is at least 1.2% below the
  highest close of the previous 60 bars. Shallower dips (50 bps) fire a
  dozen times and lose to round-trip costs; deeper dips (220 bps) never
  fire on this window at all, which would fail the race's one-trade bar.
- TAKE_BPS = 120: exit when the close recovers 1.2% above the entry close.
  A winning round trip then nets roughly 120 - 34 = 86 bps before the
  fill-price drift between the decision close and the next open.
- STOP_BPS = 150: exit when the close falls 1.5% below the entry close.
  A stopped-out round trip loses roughly 150 + 34 = 184 bps, so the win
  rate only has to clear about one in three for the design to break even.
- TRAIL_BPS = 300: exit when the close falls 3% below the highest close
  since entry. Wide on purpose: the 1-minute noise is ~6 bps, so a 3%
  trail only fires on a genuine reversal, never on chop.
- MAX_HOLD_BARS = 600: exit after ten hours in a position, whatever the
  price, so no trade can sit through the whole window.

Dev-window result for exactly these numbers (kit grade-local, not the
simulator): recorded in the build report. Stress, for the record: over
40 block-bootstrap rebuilds of the dev window the median is about +$23
with a 10th percentile near +$13 and one losing copy; over 24 rotated
copies of the window the median equals the dev score and no copy loses.
The design trades about twice per window, so it stays visibly active on
fresh windows instead of depending on a single trigger firing.

Protocol notes: the bot speaks agent-fun/bars-replay/v1 through the kit's
Python shim (vendored untouched under vendor/). It is long only, holds at
most one position, spends all of its cash on entry, and sends at most one
order per batch. Orders decided on a batch fill at the next bar's open, so
every trigger and exit is judged on closes the bot has actually seen.
All decisions are pure functions of the bars seen so far: no clock, no
randomness, no environment, no network -- two runs on the same window
produce the same orders. Amounts are integers in quote units (micro-USDC)
and prices are integers at price scale 5, compared with cross-multiplied
basis-point thresholds so no rounding can flip a trigger.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

import makefun_bars

SYMBOL = "CBBTC"

# --- Strategy parameters (basis points unless noted) -----------------------
LOOKBACK_BARS = 60        # bars of history behind the dip reference high
DIP_BPS = 120             # enter this far below the trailing high
TAKE_BPS = 120            # exit this far above the entry close
STOP_BPS = 150            # exit this far below the entry close
TRAIL_BPS = 300           # exit this far below the peak since entry
MAX_HOLD_BARS = 600       # exit after this many bars in a position
MAX_SLIPPAGE_BPS = 300    # refuse a fill more than 3% past the last close
MIN_NOTIONAL = 1_000_000   # 1 USDC, in quote units (the harness floor)


def _below_by(close, reference, bps):
    """True when close is at least bps below reference, in exact integers."""
    return close * 10_000 <= reference * (10_000 - bps)


def _above_by(close, reference, bps):
    """True when close is at least bps above reference, in exact integers."""
    return close * 10_000 >= reference * (10_000 + bps)


class DipBot:
    """One-position dip buyer. State is rebuilt from the account each batch
    so a rejected order can never leave the bot believing it holds."""

    def __init__(self):
        self.closes = []          # every CBBTC close seen, oldest first
        self.entry_close = None   # close of the bar the entry was decided on
        self.peak_close = None    # highest close since the entry decision
        self.held_bars = 0        # batches since the entry decision

    def _holding(self, state):
        holding = state.positions.get(SYMBOL)
        return holding is not None and holding.qty > 0

    def _sync(self, state):
        """Reconcile the tracked entry with the real position. If the
        account says we are flat, any tracked entry is stale (a buy that
        was rejected, or a sell that has filled) and is dropped."""
        if not self._holding(state):
            self.entry_close = None
            self.peak_close = None
            self.held_bars = 0
        elif self.entry_close is None:
            # Holding with no tracked entry (should not happen: a fill
            # always follows a decision in this run). Anchor on the latest
            # close so the exits still have a reference.
            latest = state.close(SYMBOL)
            self.entry_close = latest
            self.peak_close = latest
            self.held_bars = 0

    def on_batch(self, state, batch):
        # Only a batch carrying a fresh bar for our symbol moves the
        # strategy; a gap minute changes nothing and earns no decision.
        bar = None
        for candidate in batch.get("bars", []):
            if candidate.get("symbol") == SYMBOL:
                bar = candidate
                break
        if bar is None:
            return []
        close = int(bar["c"])
        self.closes.append(close)
        self._sync(state)

        if state.warming_up:
            return []

        if not self._holding(state):
            # Entry: flat, enough history, and the close is a real dip
            # below the trailing high -- not a new high, not noise.
            if len(self.closes) <= LOOKBACK_BARS:
                return []
            reference_high = max(self.closes[-LOOKBACK_BARS - 1:-1])
            if not _below_by(close, reference_high, DIP_BPS):
                return []
            if state.cash < MIN_NOTIONAL:
                return []
            self.entry_close = close
            self.peak_close = close
            self.held_bars = 0
            return [{
                "side": "buy",
                "symbol": SYMBOL,
                "notional_quote_units": state.cash,
                "max_slippage_bps": MAX_SLIPPAGE_BPS,
            }]

        # Holding: count the bar, track the peak, then test the exits in
        # a fixed order -- take-profit, hard stop, trailing stop, time cap.
        # The order matters only when two fire on the same close; the
        # outcome is the same either way (the whole position is sold).
        self.held_bars += 1
        if close > self.peak_close:
            self.peak_close = close
        holding = state.positions[SYMBOL]
        if _above_by(close, self.entry_close, TAKE_BPS):
            pass
        elif _below_by(close, self.entry_close, STOP_BPS):
            pass
        elif _below_by(close, self.peak_close, TRAIL_BPS):
            pass
        elif self.held_bars >= MAX_HOLD_BARS:
            pass
        else:
            return []
        return [{
            "side": "sell",
            "symbol": SYMBOL,
            "qty": holding.qty,
            "max_slippage_bps": MAX_SLIPPAGE_BPS,
        }]


def main():
    bot = DipBot()
    makefun_bars.run(bot.on_batch)


if __name__ == "__main__":
    main()
