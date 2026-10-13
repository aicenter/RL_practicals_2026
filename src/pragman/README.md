# Prag-man

A small, readable, Gymnasium-conformant Pac-Man-like gridworld, built as the running example for
the reinforcement learning course at CTU Prague. One environment carries a student from value
iteration in Week 4 to the last practical in Week 13, so that every new algorithm is compared against
the same maze rather than against a new one.

```
###########
#ooooooooo#          #  wall          o  pellet
#o#ooooo#o#          P  agent         X  goal
#o#o###o#o#          G  ghost
#oo#G.G#oo#
#oo#...#oo#          11 x 11, 51 pellets, up to four ghosts,
#oo#G.G#oo#          62 walkable cells, four actions
#o#o#.#o#o#
#o#ooooo#o#
#ooooPoooo#
###########
```

What makes it useful for teaching rather than for benchmarking:

- **It is a *known* MDP.** `env.model` hands you the exact `(P, R, rho_0)` arrays for the small
  worlds, and `pragman.successors(cfg, state, action)` gives the exact one-step distribution for a
  single state in any world. Learned values can therefore be compared with the true ones, not
  merely with each other.
- **Everything is a plain function of plain data.** States are frozen dataclasses (hashable, so
  `Q[state]` works), ghosts return distributions rather than actions, and the dynamics live in
  five pure functions you can read in one sitting.
- **Ten ready-made worlds**, one per teaching purpose, from a 25-state room to a maze whose state
  space deliberately does not fit in memory.
- **numpy and gymnasium only.** pygame, matplotlib and imageio are optional extras used for
  keyboard play and for figures.

---

## Using it in the practicals

Prag-man is part of the course repository: `uv sync` installs it together with everything else, so
any script run with `uv run` from the repository root can simply `import pragman`. The code lives
in `src/pragman/`; you are welcome to read it.

## Quickstart

```python
import pragman

env = pragman.make("classic_11x11", render_mode="ansi")
observation, info = env.reset(seed=0)

for _ in range(50):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(env.render())                  # a coloured board, one character per cell
    print(info["state"], info["pellets_left"], reward)
    if terminated or truncated:
        break
```

`pragman.make(name, **overrides)` is the one-liner: any `PragmanConfig` field can be passed as a
keyword, so `pragman.make("classic_11x11", ghost_policy="mix", slip_prob=0.2)` needs no new
factory. `import pragman` also registers Gymnasium ids, so `gym.make("Pragman/Classic11x11-v0")`
works — but `pragman.make` returns the environment unwrapped, which is what makes `env.model` and
`env.clone_state()` reachable without `env.unwrapped`.

## The three layers

The same dynamics are available at three levels of access, and a lab picks whichever it needs.

| Layer | Call | What you get |
|---|---|---|
| environment | `env.step(a)` | one sampled transition, as a Gymnasium five-tuple |
| pure core | `pragman.transition(cfg, s, a, rng)` | the same step as a function, no hidden state |
| | `pragman.successors(cfg, s, a)` | *every* outcome, with its exact probability |
| white-box model | `env.model` | `.P (S,A,S)`, `.R (S,A)`, `.rho0 (S,)`, `.states`, `.index` |

`env.model` builds by breadth-first enumeration and raises `StateSpaceTooLarge` above 1500
states — which is how Week 6 demonstrates, rather than asserts, that dynamic programming does not
scale. `successors` keeps working at any size, which is all that asynchronous DP, Dyna-Q and MCTS
ever need.

## The worlds, week by week

| Factory | Week | Ghosts | Observation | Notes |
|---|---|---|---|---|
| `empty_5x5` | 4 | 0 | index | 25 states, deterministic; hand-checkable DP |
| `slippery_5x5` | 4–5 | 0 | index | `slip_prob=0.2` |
| `maze_8x8` | 5–6, 10 | 0 | index | 52 states, `slip_prob=0.1`; also the Dyna-Q world |
| `haunted_5x5` | 5–6, 10 | 1 random | index | 576 states: DP and Dyna-Q *with* a ghost |
| `pellets_8x8` | 6–8 | 1 random | channels | state space explodes → motivates function approximation |
| `classic_11x11` | 7–9 | 2 random | channels | the workhorse; `max_steps=300` |
| `pixels_11x11` | 7 | 2 random | pixels | 88×88×3, CPU-trainable |
| `planning_11x11` | 10 | 2 chasers | channels | fully deterministic, injection on, for MCTS |
| `partial_11x11` | 13 | 2 proximity | channels | `view_radius=2` |
| `single_ghost_11x11` | 13 | 1 random | channels | for experiments with `ReplaceReward` |

```python
cfg  = pragman.configs.maze_8x8(slip_prob=0.3)   # the config, for the curious
env  = pragman.PragmanEnv(cfg)                   # full control
envs = pragman.make_vec("classic_11x11", 8)      # 8 copies, autoreset mode pinned
```

Observations come in four encodings, set by `cfg.observation`: `"index"` (`Discrete(S)`, the
tabular weeks), `"coords"` (`Box(int32, (2 + 2·n_ghosts,))`), `"channels"` (`Box(uint8, (H, W, C))`
binary planes, named in `env.channel_names`) and `"pixels"` (`Box(uint8, (H·k, W·k, 3))`). Both
array encodings are **channel-last**, so a convolutional network needs one
`x.permute(0, 3, 1, 2)` before its first `Conv2d`, and swapping `channels` for `pixels` is a
one-field change.

## Write your own layout

A layout is ASCII art, one character per cell.

| Char | Meaning |
|---|---|
| `#` | wall |
| `.` | empty floor |
| `o` | pellet |
| `P` | agent start — one or more; several mean ρ₀ is uniform over them |
| `G` | ghost spawn; the first `n_ghosts` in row-major order are used |
| `X` | goal cell |
| `O` | reserved for power pellets; rejected by the parser with a message |

```python
import pragman

env = pragman.make(pragman.PragmanConfig(
    layout="""
    #########
    #P..#..o#
    #.#.#.#.#
    #.#...#G#
    #...#..X#
    #########
    """,
    n_ghosts=1,
    ghost_policy="chaser",
    observation="index",
))
print(env.model.n_states)
```

Rows must all be the same width and there must be at least one `P`; everything else is optional,
including the surrounding wall (out of bounds counts as wall anyway). Parse errors name the
offending row and column. Connectivity is not required — an unreachable pocket is a legal and
occasionally useful map — but `layout.unreachable_cells()` will tell you about one. The shipped
maps are plain strings in `pragman.NAMED_LAYOUTS`, so the usual way to start is to copy one and
change two characters.

A *position* is the same alphabet used for a concrete situation rather than for a map:
`state.ascii(layout)` prints one, `PragmanState.from_ascii(text, layout)` reads one back, and
`env.reset(options={"state": "..."})` accepts one — which makes a hand-written test fixture
something a reader can see.

## Terminal output

Boards are drawn in colour when the terminal supports it; `pragman.ansi(state, layout)` is the
function behind them.

## Rendering, playing, recording

```python
picture = pragman.frame(state, env.layout, cell_pixels=24)   # (H·k, W·k, 3) uint8, numpy only
text    = pragman.ansi(state, env.layout)                    # coloured terminal string
env     = pragman.make("classic_11x11", render_mode="human") # a pygame window
```

`pragman.play(env)` gives keyboard control and returns the `Episode` you just played;
`pragman.record_episode(env, policy, path="clip.gif")` does the same for a policy and
optionally writes a GIF. An `Episode` carries its whole config, not merely its layout, so a
trajectory recorded today is replayable, re-renderable and re-encodable weeks later:
`episode.save(path)` / `Episode.load(path)` for one, `save_episodes` / `load_episodes` for a
dataset that stores each distinct world once.

For lecture figures, `pragman.viz` draws value heatmaps, policy arrows and visit counts with
matplotlib: `plot_values(env, V, states=model.states)`, `plot_policy(env, pi)`,
`plot_state_visits(env, counts)`.

## Tests

```bash
uv run pytest -q tests/pragman             # the fast suite
uv run pytest -q -m slow tests/pragman     # the calibration tests (throughput, learnability)
```

The tests are meant to be read: they are, for instance, where the agreement between the sampler
and the white-box model is pinned down.
