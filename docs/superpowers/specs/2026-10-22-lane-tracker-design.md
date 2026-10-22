# Lane tracker and map-first 3D view — design

Date: 2026-10-22. Status: approved in conversation, to be reviewed as written.

## Purpose

Know which lane of our carriageway we are in, as a probability per lane that
carries its history forward, and draw the 3D page around the map with our car
in that lane.

First use is the 3D page. The same output is meant to steer the car later, so
it must say how sure it is, and be reproducible on recorded drives.

What the driver asked for:

- The world is drawn from the map; what openpilot knows is placed on it.
- Lane reasoning with history, in the spirit of Monte Carlo Localization:
  at a junction with turn lanes, no indicator now or recently means we are
  probably not in a turn lane; after a lane change from the right lane we are
  in the one left of it, and so on.
- Show the most probable lane.

## Approach

A discrete Bayes (histogram) filter over the lanes of our carriageway: the
exact form of MCL for a state with at most a handful of values. No sampling,
so results are deterministic. If lateral position across the road turns out
to matter, it can be extended to a particle filter later.

## State

`p[i]` for lane `i = 0..L-1`, counted from the left of our carriageway, in our
direction of travel. `L` comes from the matched map way (`lanes` for our
direction). When the map has no lane count, `L` is taken from what the model
sees (lane lines with probability above 0.3 bound lanes), at least 1. On a new
way with no usable mapping from the previous one, `p` is uniform.

## Predict (10 Hz)

1. Lane change.
   - openpilot's own lane change (`modelV2.meta.laneChangeState` going from
     `laneChangeStarting` back to `off`, with `laneChangeDirection`) moves
     probability one lane in that direction, with weight 0.9 (0.1 stays, for
     an aborted change).
   - A driver lane change: indicator on, and the model's nearest line on that
     side crossing under the car (its lateral offset passing through ~0 and
     the lines relabelling). Weight 0.9 once the crossing is seen; indicator
     alone without a crossing does not move `p`.
   - Without either, a slow diffusion: 0.5 % per second to each neighbour, so
     old evidence loses weight.
2. Map topology. When the matched way changes, or the lane count changes along
   the path, old lanes are mapped to new ones:
   - With `turn:lanes` on both sides, align by the arrows: lanes that gain a
     pure left (right) arrow at the left (right) end are new lanes on that
     side.
   - Without them: one lane more is assumed to be added on the side the road
     later turns off to (a left-turn lane before a junction where a way leaves
     to the left, an exit lane on the right before a link); if that cannot be
     told, the extra probability is split evenly between the two sides.
   - One lane fewer: the lost lane's probability goes to its neighbour (merge).
3. Turn lanes without indicating. On entering a stretch where some lanes are
   only turn lanes (their arrows do not include through), and no indicator
   towards that side has been on in the last 10 s, those lanes' probability is
   multiplied by 0.1 before normalising; an indicator towards that side
   multiplies them by 3 instead.

## Update (10 Hz)

Each a soft likelihood, multiplied in and normalised:

- Road edges from the model: a confident left edge within ~2.5 m favours lane
  0, likewise right for lane L-1. The distance to an edge divided by the lane
  width gives a soft lane estimate, weighted by edge confidence.
- Neighbouring lanes from the model: a confident outer lane line on a side
  means a lane exists there, so we are not outermost on that side (the
  signal behind `rightmost`/`leftmost` today).
- Radar: a vehicle moving our way beside us on the left means not lane 0;
  likewise right. Oncoming vehicles do not count.
- Uncertain model (lane line probabilities below ~0.3, typical in junctions)
  weakens the model terms.

## Where it runs, what it gives

- In `tavascan_map` (which has the map and GPS), raised from 2 Hz to 10 Hz;
  map matching stays at 2 Hz, the filter runs every tick. New subscriptions:
  `modelV2`, `carState`, `radarTracks`.
- The filter is a pure class in `tavascan_map/lanes.py`, fed plain values, so
  recorded drives can be replayed through it.
- Output added to `/dev/shm/tavascan_map.json` and the trace:
  `lane = {p: [...], best: i, of: L, confidence: max(p), why: "<last change>"}`.
- Read-only towards the car: nothing is published to cereal and nothing is
  sent on CAN.

## 3D page, map first

- Road, lanes, markings, turn arrows, junctions and signs are drawn from the
  map out to the horizon.
- Our car sits in the most probable lane; the road is laid out so that lane's
  centre is under us. The model no longer decides which lane we are in.
- Radar vehicles are drawn in the lane given by their lateral offset counted
  from our lane.
- The model's lane lines become an optional layer, off by default; they still
  turn the map onto the model's heading in the background.
- HUD: "lane 2 of 3 · 85 %".

## Testing

- Replay the filter over recorded drives with a script, as with the undertake
  guard, and print lane, confidence and reason over time.
- Check it against episodes whose answer is visible on camera: a driver
  lane change into a left-turn lane at a signalised junction, a motorway exit
  lane appearing, and a four-lane approach with turn arrows.
- Unit tests for the filter's predict and update steps, and for the lane
  mapping between ways.
- Then deploy (parked at home), log while driving, and review with the driver
  after a few drives.

## Out of scope for now

- Using the lane for control (the undertake guard, or anything else).
- Lateral position across the road and GPS lateral correction (particle
  filter).
- Signal state of traffic lights.
