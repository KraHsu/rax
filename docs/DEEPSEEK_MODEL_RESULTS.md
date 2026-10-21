# DeepSeek Model Results

RAX with Deepseek-v4.1-flash reaches **66.0% success across all 11 DexJoCo
tasks** and **84.0% on the six single-arm tasks**, using the DeepSeek API through
OpenRouter. Online exploration produces a robot program that runs independently
of the language model during evaluation.

The all-task mean exceeds **ME-Dex's 65.3%**. Against representative VLA and world
action models, the gains are **11.9 percentage points over π0.5** and
**13.1 points over Fast-WAM**.
On single-arm manipulation, the respective gains are **14.3 and 24.9 points**;
four of the six tasks reach **100%** success.

## DexJoCo comparison

Success rates (%) on the official `rand-obj` evaluation layouts.

| Method | All tasks | Single arm |
| --- | ---: | ---: |
| **RAX with Deepseek-v4.1-flash** | **66.0** | **84.0** |
| RAX with Claude Opus 5.5 | 96.7 | 98.8 |
| π0.5 | 54.1 | 69.7 |
| Fast-WAM | 52.9 | 59.1 |
| GR00T N1.6 | 51.5 | 63.3 |
| ME-Dex | 65.3 | 79.0 |

Baseline results: [RAX official paper, Table 5](https://arxiv.org/html/2610.22496v1#S5.T5).
Means use each method's reported tasks; single arm covers six tasks.

The DeepSeek aggregate combines nine expert-demo sessions with a 64-episode
budget and the best completed API sessions for Mouse and iPad with a 16-episode
budget. Each selected session contributes all 150 evaluation episodes. This
gives **1089/1650** overall successes, **756/900** single-arm successes and
**333/750 (44.4%)** bimanual successes.

## Success by task

### Single-arm tasks

| Task | Profile | Exploration used / budget | Successes | Success rate |
| --- | --- | ---: | ---: | ---: |
| `hammer_nail` | Expert demos | 46/64 | 107/150 | 71.3% |
| `click_mouse` | Goal references | 10/16 | 150/150 | **100.0%** |
| `pick_bucket` | Expert demos | 59/64 | 150/150 | **100.0%** |
| `pinch_tongs` | Expert demos | 9/64 | 150/150 | **100.0%** |
| `fold_glasses` | Expert demos | 26/64 | 49/150 | 32.7% |
| `water_plant` | Expert demos | 18/64 | 150/150 | **100.0%** |
| **Mean** | | | **756/900** | **84.0%** |

### Bimanual tasks

| Task | Profile | Exploration used / budget | Successes | Success rate |
| --- | --- | ---: | ---: | ---: |
| `bimanual_unlock_ipad` | Goal references | 13/16 | 142/150 | 94.7% |
| `bimanual_hanoi` | Expert demos | 62/64 | 0/150 | 0.0% |
| `bimanual_assembly` | Expert demos | 64/64 | 0/150 | 0.0% |
| `bimanual_microwave_cook` | Expert demos | 60/64 | 135/150 | 90.0% |
| `bimanual_photograph` | Expert demos | 49/64 | 56/150 | 37.3% |
| **Mean** | | | **333/750** | **44.4%** |

Exploration includes interactive episodes and rehearsals. The budget is an
upper bound shared across the environments; a session can submit earlier.

## Experiment configuration

Both profiles use Claude Code as the agent harness and OpenRouter model
`deepseek/deepseek-v4.1-flash-1017`, the October 17 snapshot of DeepSeek V4.1
Flash. The main comparison uses native agent execution.

| Setting | Expert demos | Goal references |
| --- | --- | --- |
| Tasks | Nine tasks marked above | Mouse and iPad |
| Exploration budget | 64 episodes total | 16 episodes total |
| Exploration subagents | 4 | 0; coordinator only |
| Parallel exploration environments | 4 | 1 |
| Expert demonstrations | 3 complete episodes | 0 action demonstrations |
| Demonstration content | Video, stepwise actions, measured states, timestamps and raw records | Terminal-state reference images |
| Goal references | Three terminal-state reference images | Three terminal-state reference images |
| Observation profile | RGB-D | RGB-D |
| Model reasoning effort | `max` | `max` |
| External perception | Off | Off |
| Episode reviews | Off | Off |
| Task rules | Public | Public |
| Formal evaluation | Frozen robot program, code formal | Frozen robot program, code formal |
| Formal episodes | 150 official layouts per task | 150 official layouts per task |
| Parallel formal workers | 4 | 4 |

The Opus 5.5 reference uses Claude Code at `xhigh`, 32 exploration episodes,
one environment, no subagents and three goal images without expert actions or
videos, as described in the paper's DexJoCo protocol.

## API pricing

USD per **1 million tokens**, checked on **2026-10-21**:

| Model / provider | Uncached input | Output | Cached input |
| --- | ---: | ---: | ---: |
| DeepSeek V4.1 Flash 1017 / DeepSeek on OpenRouter | $0.15–$0.30 | $0.60–$1.20 | $0.003–$0.006 |
| Claude Opus 5.5 / Anthropic, standard mode | $4.00 | $20.00 | $0.20 |

Sources: [OpenRouter's DeepSeek provider catalog](https://openrouter.ai/api/v1/models/deepseek/deepseek-v4.1-flash-1017/endpoints)
and [Anthropic's Opus 5.5 pricing](https://www.anthropic.com/claude-opus-5-5#pricing).
The DeepSeek range reflects the provider's listed time-based rates.
At these rates, Opus 5.5 uncached input tokens cost **13.3–26.7×** as much and output
tokens cost **16.7–33.3×** as much. Session billing follows actual token use,
cache hits and provider routing.

## Click Mouse harness comparison

These additional native sessions use `click_mouse`, a 16-episode budget,
one agent and one environment, RGB-D observations, three terminal-state goal
images and no expert action demonstrations. Reasoning uses maximum effort;
external perception and episode reviews are off. Code formal evaluates the
submitted program on 150 official layouts.

| Agent harness | Model | Provider | Successes | Success rate |
| --- | --- | --- | ---: | ---: |
| Claude Code | DeepSeek V4.1 Flash 1017 | OpenRouter | 150/150 | **100.0%** |
| Claude Code | DeepSeek V4.1 Flash 1017 | Local vLLM, 4 H200 | 150/150 | **100.0%** |
| DeepSeek Harness | DeepSeek V4.1 Flash 1017 | OpenRouter | 128/150 | 85.3% |
| DeepSeek Harness | DeepSeek Flash (`deepseek-flash` account alias) | DeepSeek account | 114/150 | 76.0% |
| Kimi Code | Kimi K3 | OpenRouter | 145/150 | 96.7% |

The account run records the `deepseek-flash` alias with an unpinned revision.
Each harness/model configuration has one completed session. The local deployment
is described in the [DeepSeek serving guide](https://github.com/robot-agent-exploration/rax/blob/main/docs/DEEPSEEK_SERVING.md).

For the same task, the paper reports:

| Method | Mouse success rate |
| --- | ---: |
| π0.5 | 64.7% |
| Fast-WAM | 77.3% |
| GR00T N1.6 | 66.0% |
| ME-Dex | 76.0% |
| RAX with Claude Opus 5.5, 16 exploration episodes | 100.0% |

Both Claude Code/DeepSeek sessions match the paper's RAX-16 Mouse result.
